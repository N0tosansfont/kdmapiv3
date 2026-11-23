//! Optional, non-blocking background telemetry logger (`kdmapiv3.log`).
//!
//! When `telemetry_enabled = true` in `kdmapiv3.toml`, spawns a dedicated
//! low-priority background thread that samples lock-free atomic counters every
//! 1 second and writes a clean summary to `kdmapiv3.log` without ever touching
//! `SendDirectData`, `SendDirectDataNoBuf`, or the audio/consumer hot path.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::ringbuf::TelemetrySnapshot;

/// Period between non-blocking telemetry snapshots written to `kdmapiv3.log`.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// Snapshot returned by the lock-free runtime reader callback.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiagnosticSample {
    pub counters: TelemetrySnapshot,
    pub active_voices: u64,
    pub rendering_time_pct: f32,
}

/// Handle to the optional background telemetry writer thread.
pub struct TelemetryWorker {
    stop_signal: Arc<(Mutex<bool>, Condvar)>,
    handle: Option<JoinHandle<()>>,
}

impl TelemetryWorker {
    /// Spawn a low-priority telemetry thread sampling `reader` every 1 second
    /// and appending formatted summaries to `log_path`.
    pub fn spawn<F>(log_path: PathBuf, reader: F) -> Option<Self>
    where
        F: Fn() -> DiagnosticSample + Send + 'static,
    {
        let stop_signal = Arc::new((Mutex::new(false), Condvar::new()));
        let signal_clone = Arc::clone(&stop_signal);

        let handle = thread::Builder::new()
            .name("kdmapiv3-telemetry".to_string())
            .spawn(move || {
                platform::lower_thread_priority();
                run_logger_loop(log_path, signal_clone, reader);
            })
            .ok()?;

        Some(Self {
            stop_signal,
            handle: Some(handle),
        })
    }

    /// Wake the background telemetry thread immediately and wait for clean exit.
    pub fn stop(mut self) {
        let (lock, cvar) = &*self.stop_signal;
        if let Ok(mut stopped) = lock.lock() {
            *stopped = true;
            cvar.notify_all();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_logger_loop<F>(log_path: PathBuf, stop_signal: Arc<(Mutex<bool>, Condvar)>, reader: F)
where
    F: Fn() -> DiagnosticSample,
{
    if let Some(parent) = log_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let mut file = match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(f) => f,
        Err(_) => return,
    };

    let start_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let _ = writeln!(
        file,
        "# kdmapiv3 telemetry session started (unix_ts={})",
        start_epoch
    );
    let _ = file.flush();

    let mut prev = reader().counters;
    let mut last_tick = Instant::now();
    let (lock, cvar) = &*stop_signal;

    loop {
        let guard = match lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if *guard {
            break;
        }
        let wait_res = cvar.wait_timeout(guard, SAMPLE_INTERVAL);
        let should_stop = match wait_res {
            Ok((g, _)) => *g,
            Err(p) => *p.into_inner().0,
        };

        let now = Instant::now();
        let dt_secs = now.duration_since(last_tick).as_secs_f64().max(0.001);
        last_tick = now;

        let sample = reader();
        let cur = sample.counters;

        let d_in = cur.events_in.saturating_sub(prev.events_in);
        let d_out = cur.events_out.saturating_sub(prev.events_out);
        let d_drop = cur.events_dropped.saturating_sub(prev.events_dropped);
        prev = cur;

        let in_rate = (d_in as f64 / dt_secs) as u64;
        let out_rate = (d_out as f64 / dt_secs) as u64;

        let unix_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let _ = writeln!(
            file,
            "[ts={}] events_in={} ({}/s) events_out={} ({}/s) events_dropped={} (+{}) xrun_count={} voices={} render_pct={:.2}",
            unix_ts,
            cur.events_in,
            in_rate,
            cur.events_out,
            out_rate,
            cur.events_dropped,
            d_drop,
            cur.xrun_count,
            sample.active_voices,
            sample.rendering_time_pct,
        );
        let _ = file.flush();

        if should_stop {
            break;
        }
    }
}

/// Strictly isolated Linux low-priority helper for the telemetry thread.
#[cfg(target_os = "linux")]
mod platform {
    pub fn lower_thread_priority() {
        // SAFETY: `nice(10)` lowers scheduling priority for this non-RT logger thread.
        unsafe {
            let _ = libc::nice(10);
        }
    }
}

/// Strictly isolated Windows/Wine low-priority helper for the telemetry thread.
#[cfg(target_os = "windows")]
mod platform {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
    };

    pub fn lower_thread_priority() {
        // SAFETY: `GetCurrentThread()` returns a valid pseudo-handle for this thread.
        unsafe {
            let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        }
    }
}
