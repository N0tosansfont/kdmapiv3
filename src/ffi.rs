//! KDMAPI FFI entry points (`#[no_mangle] pub extern "system"`).
//!
//! # Real-Time & Safety Invariants
//! - Hot-path functions (`SendDirectData`, `SendDirectDataNoBuf`, `SendDirectLongData`,
//!   `SendDirectLongDataNoBuf`, `ResetKDMAPIStream`, `SendCustomEvent`, `GetVoiceCount`,
//!   `GetRenderingTime`) NEVER allocate on the heap, NEVER take blocking locks, NEVER
//!   perform config parsing/logging/syscalls, and NEVER panic.
//! - External configuration (`kdmapiv3.toml`) and SoundFonts are loaded ONLY during
//!   `InitializeKDMAPIStream` (or cold-path `LoadCustomSoundFontsList`), never inside
//!   the audio render callback.
//! - Control-path functions are wrapped in `std::panic::catch_unwind` so Rust panics
//!   can never unwind across FFI.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use crate::audio::{AudioDriver, RtEngineHooks, RtEngineShared};
use crate::backend::path_util::{
    decode_c_or_utf16_path_ptr, resolve_soundfont_input, resolve_soundfont_path,
    MAX_UTF16_PATH_UNITS,
};
use crate::backend::{SynthBackend, SynthConfig, MAX_VOICE_LIMIT, MIN_VOICE_LIMIT};
use crate::config::{load_or_create_config, KdmapiConfig};
use crate::ringbuf::{
    MidiEvent, SysExPacket, TelemetrySnapshot, DEFAULT_SHORT_RING_CAPACITY, MAX_SYSEX_LEN,
};
use crate::synth::SoundFontSynth;
use crate::telemetry::{DiagnosticSample, TelemetryWorker};

/// Reported KDMAPI version (`3.0.0.0`).
pub const KDMAPI_VER_MAJOR: u32 = 3;
pub const KDMAPI_VER_MINOR: u32 = 0;
pub const KDMAPI_VER_BUILD: u32 = 0;
pub const KDMAPI_VER_REVISION: u32 = 0;

/// WinMM / KDMAPI status codes.
pub const MMSYSERR_NOERROR: u32 = 0;
pub const MMSYSERR_ERROR: u32 = 1;
pub const MMSYSERR_INVALPARAM: u32 = 11;

/// `MIDIHDR::dwFlags` bitmasks.
pub const MHDR_DONE: u32 = 0x0000_0001;
pub const MHDR_PREPARED: u32 = 0x0000_0002;
pub const MHDR_INQUEUE: u32 = 0x0000_0004;

/// KDMAPI `DriverSettings` identifiers.
pub const KDMAPI_SET: u32 = 0x0;
pub const KDMAPI_GET: u32 = 0x1;
pub const KDMAPI_SOUNDFONT: u32 = 0xF0000;
pub const KDMAPI_MAXVOICES: u32 = 0x10026;
pub const KDMAPI_AUDIOFREQ: u32 = 0x10018;

/// C-compatible Win32/KDMAPI `MIDIHDR` layout (64-bit ABI).
#[repr(C)]
pub struct MidiHdr {
    pub lp_data: *mut u8,
    pub dw_buffer_length: u32,
    pub dw_bytes_recorded: u32,
    pub dw_user: usize,
    pub dw_flags: u32,
    pub lp_next: *mut MidiHdr,
    pub reserved: usize,
    pub dw_offset: u32,
    pub dw_reserved: [usize; 8],
}

/// Control-plane state guarded by a cold-path-only mutex.
/// NEVER accessed from hot-path FFI calls or the audio render callback.
struct ControlState {
    driver: Option<AudioDriver>,
    telemetry_worker: Option<TelemetryWorker>,
    backend: Option<Box<dyn SynthBackend>>,
    config: SynthConfig,
    loaded_config: KdmapiConfig,
    /// Ordered SoundFont paths loaded from `kdmapiv3.toml` or `LoadCustomSoundFontsList`.
    loaded_soundfonts: Vec<PathBuf>,
    /// Tracks whether `LoadCustomSoundFontsList` explicitly overrode config soundfonts.
    custom_soundfonts_override: bool,
}

/// Global pre-allocated runtime engine.
///
/// Stored inside a `OnceLock` so `RUNTIME.get()` is a single wait-free `Acquire`
/// atomic load on the hot path with zero locks, zero heap allocations, and zero
/// use-after-free hazard across stream restarts.
struct RuntimeEngine {
    shared: Arc<RtEngineShared>,
    control: Mutex<ControlState>,
}

static RUNTIME: OnceLock<RuntimeEngine> = OnceLock::new();

impl RuntimeEngine {
    fn with_config(cfg: KdmapiConfig) -> Self {
        let synth_cfg = cfg.to_synth_config();
        let ring_cap = cfg.ring_buffer_capacity;
        Self {
            shared: Arc::new(RtEngineShared::new(ring_cap)),
            control: Mutex::new(ControlState {
                driver: None,
                telemetry_worker: None,
                backend: Some(Box::new(SoundFontSynth::new())),
                config: synth_cfg,
                loaded_config: cfg,
                loaded_soundfonts: Vec::new(),
                custom_soundfonts_override: false,
            }),
        }
    }

    fn new_default() -> Self {
        let mut cfg = KdmapiConfig::default();
        cfg.ring_buffer_capacity = DEFAULT_SHORT_RING_CAPACITY;
        Self::with_config(cfg)
    }

    #[inline]
    fn rt_hooks(&self) -> RtEngineHooks {
        RtEngineHooks::new(Arc::clone(&self.shared))
    }

    /// Cold path: apply loaded `kdmapiv3.toml`, load SoundFonts, start real-time `cpal`
    /// audio output (with headless fallback), and optionally spawn the telemetry logger.
    fn start(&self, cfg: KdmapiConfig) -> bool {
        let mut ctrl = match self.control.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        if ctrl.driver.is_some() {
            self.shared.running.store(true, Ordering::Release);
            return true;
        }

        let mut synth_cfg = cfg.to_synth_config();
        synth_cfg.sample_rate = ctrl.config.sample_rate;
        ctrl.config = synth_cfg;

        if !ctrl.custom_soundfonts_override && !cfg.soundfonts.is_empty() {
            let mut resolved = Vec::with_capacity(cfg.soundfonts.len());
            for entry in &cfg.soundfonts {
                if !entry.trim().is_empty() {
                    resolved.push(resolve_soundfont_path(entry));
                }
            }
            ctrl.loaded_soundfonts = resolved;
        }

        let telemetry_enabled = cfg.telemetry_enabled;
        let log_path = cfg.log_path.clone();
        ctrl.loaded_config = cfg;

        let mut backend = ctrl
            .backend
            .take()
            .unwrap_or_else(|| Box::new(SoundFontSynth::new()));

        if backend.init(ctrl.config).is_err() {
            ctrl.backend = Some(backend);
            return false;
        }

        if !ctrl.loaded_soundfonts.is_empty() {
            let _ = backend.load_soundfonts(&ctrl.loaded_soundfonts);
        }

        self.shared.short_ring.clear();
        self.shared.sysex_ring.clear();
        self.shared.stop_requested.store(false, Ordering::Release);
        self.shared.reset_pending.store(false, Ordering::Release);
        self.shared.voice_count.store(0, Ordering::Relaxed);
        self.shared
            .rendering_time_bits
            .store(0.0f32.to_bits(), Ordering::Relaxed);
        self.shared.running.store(true, Ordering::Release);

        match AudioDriver::start(backend, ctrl.config, self.rt_hooks()) {
            Ok((driver, negotiated_sr)) => {
                // If hardware negotiated a different sample rate and SoundFonts were loaded,
                // reload SF2 samples at the exact hardware rate before playback continues.
                if negotiated_sr != ctrl.config.sample_rate && !ctrl.loaded_soundfonts.is_empty() {
                    ctrl.config.sample_rate = negotiated_sr;
                    if let Some(mut reclaimed) = driver.stop() {
                        let _ = reclaimed.init(ctrl.config);
                        let _ = reclaimed.load_soundfonts(&ctrl.loaded_soundfonts);
                        match AudioDriver::start(reclaimed, ctrl.config, self.rt_hooks()) {
                            Ok((restarted, _)) => {
                                ctrl.driver = Some(restarted);
                            }
                            Err(b) => {
                                ctrl.backend = Some(b);
                                self.shared.running.store(false, Ordering::Release);
                                return false;
                            }
                        }
                    } else {
                        self.shared.running.store(false, Ordering::Release);
                        return false;
                    }
                } else {
                    ctrl.config.sample_rate = negotiated_sr;
                    ctrl.driver = Some(driver);
                }

                if telemetry_enabled && ctrl.telemetry_worker.is_none() {
                    let shared = Arc::clone(&self.shared);
                    ctrl.telemetry_worker = TelemetryWorker::spawn(log_path, move || {
                        DiagnosticSample {
                            counters: shared.telemetry.snapshot(),
                            active_voices: shared.voice_count.load(Ordering::Relaxed),
                            rendering_time_pct: f32::from_bits(
                                shared.rendering_time_bits.load(Ordering::Relaxed),
                            ),
                        }
                    });
                }
                true
            }
            Err(returned_backend) => {
                ctrl.backend = Some(returned_backend);
                self.shared.running.store(false, Ordering::Release);
                false
            }
        }
    }

    /// Cold path: stop telemetry and real-time audio driver cleanly and reclaim backend ownership.
    fn stop(&self) -> bool {
        let mut ctrl = match self.control.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        self.shared.running.store(false, Ordering::Release);
        self.shared.stop_requested.store(true, Ordering::Release);

        if let Some(telemetry_worker) = ctrl.telemetry_worker.take() {
            telemetry_worker.stop();
        }

        if let Some(driver) = ctrl.driver.take() {
            match driver.stop() {
                Some(mut backend) => {
                    backend.shutdown();
                    ctrl.backend = Some(backend);
                }
                None => {
                    ctrl.backend = Some(Box::new(SoundFontSynth::new()));
                    return false;
                }
            }
        }

        self.shared.short_ring.clear();
        self.shared.sysex_ring.clear();
        self.shared.voice_count.store(0, Ordering::Relaxed);
        self.shared
            .rendering_time_bits
            .store(0.0f32.to_bits(), Ordering::Relaxed);
        true
    }

    /// Cold path: load resolved SoundFont path(s) into the synth without blocking hot-path ingestion.
    fn load_soundfonts(&self, paths: Vec<PathBuf>) -> bool {
        if paths.is_empty() {
            return false;
        }

        let mut ctrl = match self.control.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        let was_running = ctrl.driver.is_some();
        if was_running && ctrl.loaded_soundfonts == paths {
            ctrl.custom_soundfonts_override = true;
            return true;
        }
        if was_running {
            self.shared.stop_requested.store(true, Ordering::Release);
            if let Some(driver) = ctrl.driver.take() {
                if let Some(backend) = driver.stop() {
                    ctrl.backend = Some(backend);
                }
            }
        }

        let mut ok = false;
        if let Some(ref mut backend) = ctrl.backend {
            if backend.load_soundfonts(&paths).is_ok() {
                ctrl.loaded_soundfonts = paths;
                ctrl.custom_soundfonts_override = true;
                ok = true;
            }
        }

        if was_running {
            if let Some(backend) = ctrl.backend.take() {
                self.shared.stop_requested.store(false, Ordering::Release);
                match AudioDriver::start(backend, ctrl.config, self.rt_hooks()) {
                    Ok((driver, sr)) => {
                        ctrl.config.sample_rate = sr;
                        ctrl.driver = Some(driver);
                    }
                    Err(b) => {
                        ctrl.backend = Some(b);
                    }
                }
            }
        }

        ok
    }
}

#[inline]
fn get_runtime_hot() -> Option<&'static RuntimeEngine> {
    RUNTIME.get()
}

fn get_or_init_runtime_default() -> &'static RuntimeEngine {
    RUNTIME.get_or_init(RuntimeEngine::new_default)
}

/// Rust-side non-blocking telemetry snapshot helper for tests, benchmarks, and monitors.
pub fn telemetry_snapshot() -> TelemetrySnapshot {
    match RUNTIME.get() {
        Some(engine) => engine.shared.telemetry.snapshot(),
        None => TelemetrySnapshot::default(),
    }
}

/// Reset atomic telemetry counters (primarily for tests/benchmarks).
pub fn reset_telemetry() {
    if let Some(engine) = RUNTIME.get() {
        engine.shared.telemetry.reset();
    }
}

/// Snapshot of the last resolved SoundFont path loaded through `LoadCustomSoundFontsList`
/// or `kdmapiv3.toml` (cold path helper for verification/tests).
pub fn last_loaded_soundfont_path() -> Option<PathBuf> {
    loaded_soundfont_paths().into_iter().next()
}

/// Snapshot of all ordered SoundFont paths currently loaded (cold path helper for tests).
pub fn loaded_soundfont_paths() -> Vec<PathBuf> {
    let Some(engine) = RUNTIME.get() else {
        return Vec::new();
    };
    let guard = match engine.control.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    guard.loaded_soundfonts.clone()
}

/// Snapshot of the active [`KdmapiConfig`] loaded during `InitializeKDMAPIStream`.
pub fn active_config_snapshot() -> Option<KdmapiConfig> {
    let engine = RUNTIME.get()?;
    let guard = match engine.control.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    Some(guard.loaded_config.clone())
}

/// Approximate queued event count remaining in the short ring buffer.
pub fn pending_short_events() -> usize {
    match RUNTIME.get() {
        Some(engine) => engine.shared.short_ring.len_approx(),
        None => 0,
    }
}

/// Current allocated capacity of the short MIDI ring buffer.
pub fn short_ring_capacity() -> usize {
    match RUNTIME.get() {
        Some(engine) => engine.shared.short_ring.capacity(),
        None => 0,
    }
}

// ============================================================================
// Exported KDMAPI C ABI Entry Points
// ============================================================================

/// Query the KDMAPI version implemented by this library.
#[no_mangle]
pub extern "system" fn ReturnKDMAPIVer(
    major: *mut u32,
    minor: *mut u32,
    build: *mut u32,
    revision: *mut u32,
) -> i32 {
    let mut all_valid = true;

    // SAFETY: Each pointer is explicitly null-checked before writing.
    unsafe {
        if !major.is_null() {
            *major = KDMAPI_VER_MAJOR;
        } else {
            all_valid = false;
        }

        if !minor.is_null() {
            *minor = KDMAPI_VER_MINOR;
        } else {
            all_valid = false;
        }

        if !build.is_null() {
            *build = KDMAPI_VER_BUILD;
        } else {
            all_valid = false;
        }

        if !revision.is_null() {
            *revision = KDMAPI_VER_REVISION;
        } else {
            all_valid = false;
        }
    }

    if all_valid {
        1
    } else {
        0
    }
}

/// Returns `1` (TRUE) indicating KDMAPI is available.
#[no_mangle]
pub extern "system" fn IsKDMAPIAvailable() -> i32 {
    1
}

/// Initialize the KDMAPI stream: loads (or auto-creates) `kdmapiv3.toml`,
/// pre-allocates fixed ring buffers and voice pool, loads configured SoundFonts,
/// starts real-time `cpal` audio output, and optionally starts the telemetry logger.
#[no_mangle]
pub extern "system" fn InitializeKDMAPIStream() -> i32 {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let cfg = load_or_create_config();
        let init_cfg = cfg.clone();
        let engine = RUNTIME.get_or_init(move || RuntimeEngine::with_config(init_cfg));
        if engine.start(cfg) {
            1
        } else {
            0
        }
    }));
    res.unwrap_or(0)
}

/// Terminate the KDMAPI stream and cleanly shut down telemetry, audio output, and synth.
#[no_mangle]
pub extern "system" fn TerminateKDMAPIStream() -> i32 {
    let res = catch_unwind(AssertUnwindSafe(|| match RUNTIME.get() {
        Some(engine) => {
            if engine.stop() {
                1
            } else {
                0
            }
        }
        None => 1,
    }));
    res.unwrap_or(0)
}

/// Reset all active notes/controllers on the stream without allocating or blocking.
#[no_mangle]
pub extern "system" fn ResetKDMAPIStream() {
    if let Some(engine) = get_runtime_hot() {
        engine.shared.telemetry.record_event_in();
        if !engine.shared.short_ring.try_push(MidiEvent::reset()) {
            engine.shared.reset_pending.store(true, Ordering::Release);
            engine.shared.telemetry.record_event_dropped();
        }
    }
}

/// Register optional WinMM-compatible callback handles (ABI-compatible lock-free no-op).
#[no_mangle]
pub extern "system" fn InitializeCallbackFeatures(
    _hmo: *mut c_void,
    _cb: usize,
    _inst: usize,
    _usr: usize,
    _cb_mode: u32,
) -> i32 {
    1
}

/// Invoke registered callback hook (panic-safe no-op).
#[no_mangle]
pub extern "system" fn RunCallbackFunction(_msg: u32, _p1: usize, _p2: usize) {}

/// Hot-path short MIDI message ingestion.
///
/// Strictly zero heap allocations, zero locks, zero syscalls, zero config I/O, and zero panics.
#[no_mangle]
pub extern "system" fn SendDirectData(dw_msg: u32) -> u32 {
    if let Some(engine) = get_runtime_hot() {
        engine.shared.telemetry.record_event_in();
        if engine.shared.running.load(Ordering::Acquire) {
            if !engine
                .shared
                .short_ring
                .try_push(MidiEvent::short_msg(dw_msg))
            {
                engine.shared.telemetry.record_event_dropped();
            }
        }
    }
    MMSYSERR_NOERROR
}

/// Hot-path unbuffered short MIDI message ingestion.
#[no_mangle]
pub extern "system" fn SendDirectDataNoBuf(dw_msg: u32) -> u32 {
    if let Some(engine) = get_runtime_hot() {
        engine.shared.telemetry.record_event_in();
        if engine.shared.running.load(Ordering::Acquire) {
            if !engine
                .shared
                .short_ring
                .try_push(MidiEvent::short_msg_no_buf(dw_msg))
            {
                engine.shared.telemetry.record_event_dropped();
            }
        }
    }
    MMSYSERR_NOERROR
}

/// Hot-path long / SysEx MIDI message ingestion with strict pointer & size validation.
#[no_mangle]
pub extern "system" fn SendDirectLongData(midi_hdr: *mut MidiHdr, midi_hdr_size: u32) -> u32 {
    if midi_hdr.is_null() || (midi_hdr_size as usize) < core::mem::size_of::<MidiHdr>() {
        return MMSYSERR_INVALPARAM;
    }

    // SAFETY: Non-null and `midi_hdr_size >= size_of::<MidiHdr>()` verified above.
    let hdr = unsafe { &mut *midi_hdr };

    let payload_len = if hdr.dw_bytes_recorded > 0 {
        hdr.dw_bytes_recorded.min(hdr.dw_buffer_length) as usize
    } else {
        hdr.dw_buffer_length as usize
    };

    if payload_len > 0 && hdr.lp_data.is_null() {
        return MMSYSERR_INVALPARAM;
    }

    if let Some(engine) = get_runtime_hot() {
        engine.shared.telemetry.record_event_in();
        if engine.shared.running.load(Ordering::Acquire) && payload_len > 0 {
            let copy_len = payload_len.min(MAX_SYSEX_LEN);
            // SAFETY: `hdr.lp_data` is non-null and caller guarantees `dw_buffer_length` readable bytes.
            let slice = unsafe { core::slice::from_raw_parts(hdr.lp_data as *const u8, copy_len) };
            let pkt = SysExPacket::from_slice_clamped(slice);
            if !engine.shared.sysex_ring.try_push(pkt) {
                engine.shared.telemetry.record_event_dropped();
            }
        }
    }

    hdr.dw_flags = (hdr.dw_flags & !MHDR_INQUEUE) | MHDR_DONE;
    MMSYSERR_NOERROR
}

/// Unbuffered long / SysEx MIDI message ingestion.
#[no_mangle]
pub extern "system" fn SendDirectLongDataNoBuf(midi_hdr: *mut MidiHdr, midi_hdr_size: u32) -> u32 {
    SendDirectLongData(midi_hdr, midi_hdr_size)
}

/// Mark a `MIDIHDR` as prepared after null and size validation.
#[no_mangle]
pub extern "system" fn PrepareLongData(midi_hdr: *mut MidiHdr, midi_hdr_size: u32) -> u32 {
    if midi_hdr.is_null() || (midi_hdr_size as usize) < core::mem::size_of::<MidiHdr>() {
        return MMSYSERR_INVALPARAM;
    }
    // SAFETY: Non-null and size checked above.
    unsafe {
        (*midi_hdr).dw_flags |= MHDR_PREPARED;
    }
    MMSYSERR_NOERROR
}

/// Unmark a `MIDIHDR` as prepared after null and size validation.
#[no_mangle]
pub extern "system" fn UnprepareLongData(midi_hdr: *mut MidiHdr, midi_hdr_size: u32) -> u32 {
    if midi_hdr.is_null() || (midi_hdr_size as usize) < core::mem::size_of::<MidiHdr>() {
        return MMSYSERR_INVALPARAM;
    }
    // SAFETY: Non-null and size checked above.
    unsafe {
        (*midi_hdr).dw_flags &= !MHDR_PREPARED;
    }
    MMSYSERR_NOERROR
}

/// Hot-path custom KDMAPI event ingestion (zero allocation, zero locks, zero panics).
#[no_mangle]
pub extern "system" fn SendCustomEvent(event_type: u32, chan: u32, param: u32) -> i32 {
    let Some(engine) = get_runtime_hot() else {
        return 0;
    };
    engine.shared.telemetry.record_event_in();
    if !engine.shared.running.load(Ordering::Acquire) {
        return 0;
    }
    if engine
        .shared
        .short_ring
        .try_push(MidiEvent::custom(event_type, chan, param))
    {
        1
    } else {
        engine.shared.telemetry.record_event_dropped();
        0
    }
}

/// Control-path KDMAPI driver settings getter/setter with pointer validation and panic safety.
#[no_mangle]
pub extern "system" fn DriverSettings(
    setting: u32,
    mode: u32,
    value: *mut c_void,
    cb_value: u32,
) -> i32 {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if cb_value > 0 && value.is_null() {
            return 0;
        }

        let engine = get_or_init_runtime_default();
        let mut ctrl = match engine.control.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        match setting {
            KDMAPI_MAXVOICES => {
                if (cb_value as usize) < core::mem::size_of::<u32>() || value.is_null() {
                    return 0;
                }
                let ptr = value as *mut u32;
                // SAFETY: Non-null and `cb_value >= 4` verified above.
                unsafe {
                    if mode == KDMAPI_GET {
                        *ptr = ctrl.config.max_voices;
                    } else if mode == KDMAPI_SET {
                        ctrl.config.max_voices = (*ptr).clamp(MIN_VOICE_LIMIT, MAX_VOICE_LIMIT);
                    }
                }
                1
            }
            KDMAPI_AUDIOFREQ => {
                if (cb_value as usize) < core::mem::size_of::<u32>() || value.is_null() {
                    return 0;
                }
                let ptr = value as *mut u32;
                // SAFETY: Non-null and `cb_value >= 4` verified above.
                unsafe {
                    if mode == KDMAPI_GET {
                        *ptr = ctrl.config.sample_rate;
                    } else if mode == KDMAPI_SET {
                        ctrl.config.sample_rate = (*ptr).clamp(8_000, 384_000);
                    }
                }
                1
            }
            KDMAPI_SOUNDFONT => {
                if value.is_null() {
                    0
                } else {
                    1
                }
            }
            _ => 1,
        }
    }));
    res.unwrap_or(0)
}

/// Load a custom SoundFont or legacy `.sflist`/`.csflist`/`.omlist` path passed as a
/// NUL-terminated UTF-16 (`PCWSTR`, Windows/Wine) or UTF-8 (`*const c_char`, Linux) string.
#[no_mangle]
pub extern "system" fn LoadCustomSoundFontsList(directory: *const u16) -> i32 {
    let res = catch_unwind(AssertUnwindSafe(|| {
        if directory.is_null() {
            return 0;
        }
        // SAFETY: Non-null verified above; bounded read stops at NUL or `MAX_UTF16_PATH_UNITS`.
        let Some(raw) = (unsafe { decode_c_or_utf16_path_ptr(directory, MAX_UTF16_PATH_UNITS) })
        else {
            return 0;
        };
        if raw.trim().is_empty() {
            return 0;
        }

        let resolved = resolve_soundfont_input(&raw);
        let engine = get_or_init_runtime_default();
        if engine.load_soundfonts(resolved) {
            1
        } else {
            0
        }
    }));
    res.unwrap_or(0)
}

/// Lock-free query for the synthesizer's current CPU rendering load percentage.
#[no_mangle]
pub extern "system" fn GetRenderingTime() -> f32 {
    match get_runtime_hot() {
        Some(engine) => {
            f32::from_bits(engine.shared.rendering_time_bits.load(Ordering::Relaxed))
        }
        None => 0.0,
    }
}

/// Lock-free query for the synthesizer's current active voice count.
#[no_mangle]
pub extern "system" fn GetVoiceCount() -> u64 {
    match get_runtime_hot() {
        Some(engine) => engine.shared.voice_count.load(Ordering::Relaxed),
        None => 0,
    }
}
