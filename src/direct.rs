//! In-process Rust direct API for embedding `kdmapiv3` into hosts (e.g., Wasabi)
//! without dynamic DLL loading, C FFI shims, temp `.csflist` files, or hot-path locks.
//!
//! # Real-Time Guarantees
//! - [`DirectMidiSender`] is `Clone + Send + Sync` and communicates strictly through
//!   the pre-allocated lock-free [`crate::ringbuf::LockFreeRingBuffer`] and atomics.
//! - `push_short`, `push_short_batch`, `push_short_iter`, `reset`, `voice_count`,
//!   `rendering_time_pct`, and `telemetry_snapshot` perform ZERO heap allocations,
//!   ZERO blocking locks, ZERO file I/O, and ZERO panics.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::audio::{AudioDriver, RtEngineHooks, RtEngineShared};
use crate::backend::{BackendError, SynthBackend, SynthConfig};
use crate::config::{MAX_RING_CAPACITY, MIN_RING_CAPACITY};
use crate::ringbuf::{MidiEvent, TelemetrySnapshot, DEFAULT_SHORT_RING_CAPACITY};
use crate::synth::SoundFontSynth;

/// Stack batch size used for Zero-Copy lock-free ring buffer pushes.
const DIRECT_PUSH_BATCH: usize = 256;

/// Lock-free, allocation-free MIDI producer & atomic telemetry handle for in-process hosts.
#[derive(Clone)]
pub struct DirectMidiSender {
    shared: Arc<RtEngineShared>,
}

impl DirectMidiSender {
    /// Push a single packed 32-bit KDMAPI short MIDI message (`status | (d1 << 8) | (d2 << 16)`).
    #[inline]
    pub fn push_short(&self, dw_msg: u32) -> bool {
        self.shared.telemetry.record_event_in();
        if !self.shared.running.load(Ordering::Acquire) {
            return false;
        }
        if self.shared.short_ring.try_push(MidiEvent::short_msg(dw_msg)) {
            true
        } else {
            self.shared.telemetry.record_event_dropped();
            false
        }
    }

    /// Push a slice of packed 32-bit short MIDI messages in lock-free batches with zero heap allocation.
    #[inline]
    pub fn push_short_batch(&self, msgs: &[u32]) -> usize {
        if msgs.is_empty() {
            return 0;
        }
        let running = self.shared.running.load(Ordering::Acquire);
        let mut pushed_total = 0usize;
        let mut stack_buf = [MidiEvent::EMPTY; DIRECT_PUSH_BATCH];

        for chunk in msgs.chunks(DIRECT_PUSH_BATCH) {
            for (dst, &dw_msg) in stack_buf.iter_mut().zip(chunk.iter()) {
                *dst = MidiEvent::short_msg(dw_msg);
            }
            for _ in 0..chunk.len() {
                self.shared.telemetry.record_event_in();
            }
            if !running {
                continue;
            }
            let pushed = self.shared.short_ring.try_push_batch(&stack_buf[..chunk.len()]);
            pushed_total += pushed;
            let dropped = chunk.len() - pushed;
            for _ in 0..dropped {
                self.shared.telemetry.record_event_dropped();
            }
        }
        pushed_total
    }

    /// Drain an iterator of packed 32-bit short MIDI messages into the lock-free ring buffer
    /// using a fixed stack staging buffer (zero heap allocations, zero locks).
    #[inline]
    pub fn push_short_iter(&self, iter: impl Iterator<Item = u32>) -> usize {
        let running = self.shared.running.load(Ordering::Acquire);
        let mut stack_buf = [MidiEvent::EMPTY; DIRECT_PUSH_BATCH];
        let mut count = 0usize;
        let mut pushed_total = 0usize;

        for dw_msg in iter {
            self.shared.telemetry.record_event_in();
            if !running {
                continue;
            }
            stack_buf[count] = MidiEvent::short_msg(dw_msg);
            count += 1;
            if count == DIRECT_PUSH_BATCH {
                let pushed = self.shared.short_ring.try_push_batch(&stack_buf[..count]);
                pushed_total += pushed;
                for _ in 0..(count - pushed) {
                    self.shared.telemetry.record_event_dropped();
                }
                count = 0;
            }
        }

        if count > 0 && running {
            let pushed = self.shared.short_ring.try_push_batch(&stack_buf[..count]);
            pushed_total += pushed;
            for _ in 0..(count - pushed) {
                self.shared.telemetry.record_event_dropped();
            }
        }

        pushed_total
    }

    /// Request an allocation-free, lock-free stream reset (clears all voices & controllers).
    #[inline]
    pub fn reset(&self) {
        self.shared.telemetry.record_event_in();
        if !self.shared.short_ring.try_push(MidiEvent::reset()) {
            self.shared.reset_pending.store(true, Ordering::Release);
            self.shared.telemetry.record_event_dropped();
        }
    }

    /// Lock-free active voice count snapshot.
    #[inline]
    pub fn voice_count(&self) -> u64 {
        self.shared.voice_count.load(Ordering::Relaxed)
    }

    /// Lock-free audio callback CPU load percentage snapshot.
    #[inline]
    pub fn rendering_time_pct(&self) -> f32 {
        f32::from_bits(self.shared.rendering_time_bits.load(Ordering::Relaxed))
    }

    /// Lock-free telemetry counter snapshot (`events_in`, `events_out`, `events_dropped`, `xrun_count`).
    #[inline]
    pub fn telemetry_snapshot(&self) -> TelemetrySnapshot {
        self.shared.telemetry.snapshot()
    }
}

/// Cold-path control state owned by [`DirectKdmapiEngine`].
/// Wrapped in a cold-path `Mutex` so [`DirectKdmapiEngine`] is `Send + Sync` without `unsafe`,
/// while hot-path [`DirectMidiSender`] never touches this lock.
struct DirectControlState {
    driver: Option<AudioDriver>,
    backend: Option<Box<dyn SynthBackend>>,
    config: SynthConfig,
    loaded_soundfonts: Vec<PathBuf>,
}

/// Owned in-process real-time `kdmapiv3` engine instance (no global singleton, clean RAII shutdown).
pub struct DirectKdmapiEngine {
    shared: Arc<RtEngineShared>,
    control: Mutex<DirectControlState>,
}

impl DirectKdmapiEngine {
    /// Start an owned real-time `kdmapiv3` stream with explicit manual configuration and
    /// in-memory SoundFont paths (zero temp `.csflist` files on disk).
    pub fn start(
        mut config: SynthConfig,
        ring_buffer_capacity: usize,
        soundfonts: &[PathBuf],
    ) -> Result<Self, BackendError> {
        let cap = ring_buffer_capacity
            .clamp(MIN_RING_CAPACITY, MAX_RING_CAPACITY)
            .checked_next_power_of_two()
            .unwrap_or(DEFAULT_SHORT_RING_CAPACITY);
        let shared = Arc::new(RtEngineShared::new(cap));

        let mut backend: Box<dyn SynthBackend> = Box::new(SoundFontSynth::new());
        backend.init(config)?;

        let sf_paths: Vec<PathBuf> = soundfonts
            .iter()
            .filter(|p| !p.as_os_str().is_empty())
            .cloned()
            .collect();
        if !sf_paths.is_empty() {
            let _ = backend.load_soundfonts(&sf_paths);
        }

        shared.stop_requested.store(false, Ordering::Release);
        shared.reset_pending.store(false, Ordering::Release);
        shared.running.store(true, Ordering::Release);

        let hooks = RtEngineHooks::new(Arc::clone(&shared));
        let (driver, negotiated_sr) = match AudioDriver::start(backend, config, hooks) {
            Ok(res) => res,
            Err(_) => {
                shared.running.store(false, Ordering::Release);
                return Err(BackendError::InitFailed);
            }
        };

        config.sample_rate = negotiated_sr;

        Ok(Self {
            shared,
            control: Mutex::new(DirectControlState {
                driver: Some(driver),
                backend: None,
                config,
                loaded_soundfonts: sf_paths,
            }),
        })
    }

    /// Obtain a clonable lock-free [`DirectMidiSender`] handle bound to this engine.
    pub fn sender(&self) -> DirectMidiSender {
        DirectMidiSender {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Cold-path: replace loaded SoundFonts directly from in-memory paths (zero temp files).
    pub fn load_soundfonts(&mut self, paths: &[PathBuf]) -> Result<(), BackendError> {
        let ctrl = self.control.get_mut().unwrap_or_else(|p| p.into_inner());
        let sf_paths: Vec<PathBuf> = paths
            .iter()
            .filter(|p| !p.as_os_str().is_empty())
            .cloned()
            .collect();

        let was_running = ctrl.driver.is_some();
        if was_running {
            self.shared.stop_requested.store(true, Ordering::Release);
            if let Some(driver) = ctrl.driver.take() {
                ctrl.backend = driver.stop();
            }
        }

        let load_res = if let Some(ref mut backend) = ctrl.backend {
            if sf_paths.is_empty() {
                backend.reset();
                Ok(())
            } else {
                backend.load_soundfonts(&sf_paths)
            }
        } else {
            Err(BackendError::NotInitialized)
        };

        if load_res.is_ok() {
            ctrl.loaded_soundfonts = sf_paths;
        }

        if was_running {
            if let Some(backend) = ctrl.backend.take() {
                self.shared.stop_requested.store(false, Ordering::Release);
                let hooks = RtEngineHooks::new(Arc::clone(&self.shared));
                match AudioDriver::start(backend, ctrl.config, hooks) {
                    Ok((driver, sr)) => {
                        ctrl.config.sample_rate = sr;
                        ctrl.driver = Some(driver);
                    }
                    Err(returned) => {
                        ctrl.backend = Some(returned);
                        self.shared.running.store(false, Ordering::Release);
                        return Err(BackendError::InitFailed);
                    }
                }
            }
        }

        load_res
    }

    /// Negotiated audio hardware sample rate.
    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.control
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .config
            .sample_rate
    }

    /// Active [`SynthConfig`] snapshot.
    #[inline]
    pub fn config(&self) -> SynthConfig {
        self.control
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .config
    }

    /// Cold-path clean shutdown.
    pub fn stop(&mut self) {
        let ctrl = self.control.get_mut().unwrap_or_else(|p| p.into_inner());
        self.shared.running.store(false, Ordering::Release);
        self.shared.stop_requested.store(true, Ordering::Release);
        if let Some(driver) = ctrl.driver.take() {
            if let Some(mut backend) = driver.stop() {
                backend.shutdown();
                ctrl.backend = Some(backend);
            }
        }
        self.shared.short_ring.clear();
        self.shared.sysex_ring.clear();
        self.shared.voice_count.store(0, Ordering::Relaxed);
        self.shared
            .rendering_time_bits
            .store(0.0f32.to_bits(), Ordering::Relaxed);
    }
}

impl Drop for DirectKdmapiEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::NO_AUDIO_ENV;

    #[test]
    fn direct_in_process_engine_batch_push_and_clean_shutdown() {
        std::env::set_var(NO_AUDIO_ENV, "1");
        let cfg = SynthConfig {
            max_voices: 256,
            buffer_frames: 256,
            render_threads: 2,
            ..SynthConfig::default()
        };
        let engine = DirectKdmapiEngine::start(cfg, 4096, &[]).expect("start direct engine");
        let sender = engine.sender();

        let batch: [u32; 64] = core::array::from_fn(|i| 0x0064_0090 | (((i % 88 + 21) as u32) << 8));
        assert_eq!(sender.push_short_batch(&batch), 64);
        assert_eq!(sender.push_short_iter(batch.iter().copied()), 64);
        sender.reset();

        for _ in 0..10_000 {
            if sender.telemetry_snapshot().events_out >= 129 {
                break;
            }
            std::thread::yield_now();
        }
        let snap = sender.telemetry_snapshot();
        assert_eq!(snap.events_in, 129);
        assert_eq!(snap.events_out, 129);
        assert_eq!(snap.events_dropped, 0);
        drop(engine);
    }
}
