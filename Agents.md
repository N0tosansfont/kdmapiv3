# AGENTS.md — kdmapiv3 (Rust-New Build)

## 1. Project Mission
Build `kdmapiv3`, a drop-in, ultra-low-latency Rust (`cdylib`) replacement for KDMAPI (OmniMIDI) targeting high-density Black MIDI playback (10,000+ voices, 8pt/16pt Sinc interpolation, millions of events/sec) across Linux native and Wine/Windows environments.

## 2. Hard Real-Time Audio Rules (CRITICAL — ZERO TOLERANCE)
- **No Heap Allocations on Hot Path:** `SendDirectData`, `SendDirectDataNoBuf`, and audio render/consumer loops MUST NOT allocate (`Vec::push`, `Box::new`, `String`, `format!`, etc.). Pre-allocate fixed-size buffers during `InitializeKDMAPIStream`.
- **No Blocking Locks on Hot Path:** NEVER use `std::sync::Mutex`, `RwLock`, or blocking I/O inside MIDI ingestion or audio render paths. Use lock-free ring buffers (`rtrb` or `ringbuf`) and `Atomic*` operations (`Acquire`/`Release`/`Relaxed` as appropriate).
- **No Logging/Syscalls on Hot Path:** NEVER call `println!`, `eprintln!`, file I/O, or heavy syscalls inside hot paths. Expose lock-free atomic telemetry counters (`events_in`, `events_dropped`, `xrun_count`) for a separate non-blocking reader.
- **Graceful Overflow:** On ring-buffer saturation during extreme note bursts, drop cleanly at the bounded ring edge without panicking or stalling the caller thread.

## 3. Black MIDI Real-Time Integrity Standards (CRITICAL — ALWAYS ENFORCE)
- **Zero Note Skipping / Zero Queue-Level Filtering (노트 스킵 절대 금지):** NEVER skip, stride, sample, decimate, or discard `Note-On` events in the MIDI ingestion or drain path (`SendDirectData*`, `drain_midi_events`, ring-buffer consumer) before they reach `SynthBackend::process_short_event()`. Every dequeued MIDI event (Note-On, Note-Off, CC, Pitch Bend, Program Change, SysEx, Reset) MUST be delivered 100% faithfully in FIFO order to the synth backend.
- **No Arbitrary Velocity / Key Pre-Filtering (임의 벨로시티 컷 / 노트 무시 금지):** NEVER add artificial velocity cutoffs (e.g., ignoring low velocities `1..N`), key-range stripping, or heuristic note-dropping hacks to inflate apparent NPS or reduce CPU load. Only standard MIDI (`vel == 0` = Note-Off) and SF2 2.01 acoustic silence floor (`<= -100 dB` / `1e-5`, e.g., muted `CC7=0`/`CC11=0`) are allowed inside the synth.
- **Legitimate Polyphony & Pipeline Optimizations Only (정공법 최적화만 허용):** Bounding active voices MUST happen strictly through authentic, user-configured `voice_limit` polyphony management and $O(1)$ voice stealing inside the synthesizer (`SoundFontSynth`), paired with genuine engine optimizations (AVX2/FMA SIMD, lock-free batched ring popping, cache-dense SoA voice layouts, zero-alloc hot paths, and user-explicit DSP effect toggles in `kdmapiv3.toml`).
- **Backpressure Without Skipping:** If a real-time audio callback hits its drain time budget (`drain_deadline`), it may yield to DSP rendering strictly by leaving unpopped events intact in the ring buffer for the next callback—NEVER by discarding or skipping Note-Ons inside the consumer loop.

## 4. FFI, ABI & Panic Safety
- Every exported KDMAPI symbol MUST be `#[no_mangle] pub extern "system" fn ...`.
- **Never Unwind Across FFI:** Guard FFI boundaries or keep hot paths strictly panic-free (`no unwrap()`, `no expect()`, no out-of-bounds indexing). Unwinding across FFI/Wine crashes the host player immediately.
- Validate all raw pointers (`ReturnKDMAPIVer`, `LoadCustomSoundFontsList`, `SendDirectLongData`) for null before reading/writing.
- Pass only C primitives and raw pointers across FFI—never Rust `String`, `Vec`, `Option`, or `Result`.

## 5. Linux & Wine Cross-Platform Compatibility (STRICT)
- Must compile cleanly for both targets from Linux:
  1. Linux native: `cargo build --release --target x86_64-unknown-linux-gnu` -> `libOmniMIDI.so`
  2. Windows / Wine PE: `cargo build --release --target x86_64-pc-windows-gnu` -> `OmniMIDI.dll`
- **OS Isolation:** Keep core MIDI parsing, lock-free ring buffers, voice/queue logic, and DSP mixing in 100% OS-agnostic modules. Gate all OS-specific crates (`windows-sys`, `winapi`, `alsa`, `pipewire`, `nix`, `libc`) strictly behind `#[cfg(target_os = "windows")]` or `#[cfg(target_os = "linux")]`.
- **Clean Exports:** Provide a `.def` file or linker export control for Windows/Wine DLLs so exported symbols have clean undecorated names, and restrict exported symbols on Linux `.so` to avoid collisions with host players (e.g., Wasabi/BASS).
- **UTF-16 & Path Normalization:** Convert `*const u16` (`PCWSTR`) safely via `String::from_utf16_lossy` up to null terminator. Normalize `\` to `/` and support case-insensitive fallback when resolving `.sf2`/`.sfz` paths under Linux/Wine filesystems.
- **Avoid `wineserver` & RT Pitfalls:** Under Wine, avoid per-event Windows kernel synchronization objects (`CreateEvent`/`SetEvent` per MIDI message) that trigger `wineserver` IPC overhead and XRUN spikes; rely on atomic ring-buffer signaling and adaptive backoff. On Linux, request RT scheduling (`SCHED_FIFO` / PipeWire RT) gracefully with safe fallback when unprivileged.

## 6. Architecture & Verification Standard
- Keep synthesizer backends modular behind a trait (`SynthBackend`) so XSynth, BASSMIDI, or custom DSP engines can be plugged in cleanly.
- Every structural change to the ring buffer or hot path must include a benchmark or stress test simulating high-rate MIDI event bursts and checking zero allocations and symbol exports.