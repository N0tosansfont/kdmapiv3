#![allow(non_snake_case)]

//! `kdmapiv3` (`OmniMIDI`) — Drop-in, ultra-low-latency 64-bit Rust KDMAPI implementation.
//!
//! Architecture:
//! - [`ringbuf`]: Pre-allocated, lock-free bounded ring buffers and atomic telemetry counters.
//! - [`backend`]: Modular [`backend::SynthBackend`] trait and OS-isolated platform helpers.
//! - [`synth`]: Hard real-time, pre-allocated SoundFont (`.sf2`/`.sfz`) synth with Linear/Sinc8/Sinc16/Sinc32.
//! - [`audio`]: Cross-platform low-latency `cpal` audio output (ALSA/PipeWire & WASAPI).
//! - [`config`]: Cold-path `kdmapiv3.toml` loader & safe default auto-generator.
//! - [`telemetry`]: Optional, non-blocking low-priority background `kdmapiv3.log` writer.
//! - [`ffi`]: Panic-safe, null-checked `#[no_mangle] pub extern "system"` KDMAPI C ABI.

#[cfg(not(target_pointer_width = "64"))]
compile_error!(
    "we didn't build this for 32-bit, but if you're stubborn enough to compile it on x86 anyway, we won't stop you :) — kdmapiv3 requires a 64-bit target (target_pointer_width = \"64\")."
);

pub mod audio;
pub mod backend;
pub mod config;
pub mod direct;
pub mod ffi;
pub mod ringbuf;
pub mod synth;
pub mod telemetry;

pub use audio::{RtEngineHooks, RtEngineShared, NO_AUDIO_ENV};
pub use backend::{
    BackendError, InterpolationMode, SynthBackend, SynthConfig, SynthStats,
    DEFAULT_BUFFER_FRAMES, DEFAULT_RENDER_THREADS, DEFAULT_SAMPLE_RATE, DEFAULT_VOICE_LIMIT,
    MAX_BUFFER_FRAMES, MAX_RENDER_THREADS, MAX_VOICE_LIMIT, MIN_BUFFER_FRAMES, MIN_RENDER_THREADS,
    MIN_VOICE_LIMIT,
};
pub use config::{
    load_or_create_config, KdmapiConfig, CONFIG_ENV_OVERRIDE, CONFIG_FILENAME,
    DEFAULT_CONFIG_TEMPLATE, MAX_RING_CAPACITY, MIN_RING_CAPACITY, TELEMETRY_LOG_FILENAME,
};
pub use direct::{DirectKdmapiEngine, DirectMidiSender};
pub use ffi::{
    active_config_snapshot, last_loaded_soundfont_path, loaded_soundfont_paths,
    pending_short_events, reset_telemetry, short_ring_capacity, telemetry_snapshot, DriverSettings,
    GetRenderingTime, GetVoiceCount, InitializeCallbackFeatures, InitializeKDMAPIStream,
    IsKDMAPIAvailable, LoadCustomSoundFontsList, MidiHdr, PrepareLongData, ResetKDMAPIStream,
    ReturnKDMAPIVer, RunCallbackFunction, SendCustomEvent, SendDirectData, SendDirectDataNoBuf,
    SendDirectLongData, SendDirectLongDataNoBuf, TerminateKDMAPIStream, UnprepareLongData,
    KDMAPI_VER_BUILD, KDMAPI_VER_MAJOR, KDMAPI_VER_MINOR, KDMAPI_VER_REVISION, MHDR_DONE,
    MHDR_INQUEUE, MHDR_PREPARED, MMSYSERR_INVALPARAM, MMSYSERR_NOERROR,
};
pub use ringbuf::{
    EventKind, LockFreeRingBuffer, MidiEvent, SysExPacket, TelemetryCounters, TelemetrySnapshot,
    DEFAULT_SHORT_RING_CAPACITY, DEFAULT_SYSEX_RING_CAPACITY, MAX_SYSEX_LEN,
};
pub use synth::{SoundFontBank, SoundFontSynth};
