# kdmapiv3

Ultra-low-latency, hard-real-time Rust (`cdylib`) drop-in replacement for **KDMAPI / OmniMIDI**, built for high-density Black MIDI playback across native Linux and Wine/Proton.

- **Zero allocations, zero locks, zero I/O** on `SendDirectData`, `SendDirectDataNoBuf`, `SendDirectLongData`, and the audio/MIDI consumer loop.
- **Dual-target drop-in binaries:** `libOmniMIDI.so` (Linux) and `OmniMIDI.dll` (Windows / Wine), completely decoupled from GUI crates.
- **Standalone GUI Configurator:** `kdmapiv3-config` (`kdmapiv3-config.exe`) built with `eframe`/`egui` for editing `kdmapiv3.toml` atomically, managing `.sf2`/`.sfz` SoundFonts with drag-and-drop, and viewing live `kdmapiv3.log` telemetry.

---

## Quick Start

### 1. Build & Package Both Targets (Core Library + GUI Configurator)

```bash
./scripts/build_all.sh
```

This builds `libOmniMIDI.so`, `OmniMIDI.dll`, `kdmapiv3-config`, and `kdmapiv3-config.exe`, verifies all 18 exported KDMAPI symbols (`nm` / `objdump`), and outputs a ready-to-copy `dist/` bundle:

```text
dist/
├── kdmapiv3.toml
├── linux/
│   ├── libOmniMIDI.so
│   ├── kdmapiv3-config
│   └── kdmapiv3.toml
└── windows/
    ├── OmniMIDI.dll
    ├── kdmapiv3-config.exe
    └── kdmapiv3.toml
```

### 2. Native Linux (Wasabi) Drop-In Placement

1. Copy `dist/linux/libOmniMIDI.so` (and optionally `dist/linux/kdmapiv3-config`) into your Wasabi directory or library search path.
2. Place `kdmapiv3.toml` either **next to `libOmniMIDI.so`** or at **`~/.config/kdmapiv3/kdmapiv3.toml`** (auto-created on first `InitializeKDMAPIStream` if absent), or run `./kdmapiv3-config` to tune settings and drag-and-drop SoundFonts.
3. Launch Wasabi and select **KDMAPI / OmniMIDI** as the MIDI output.

### 3. Wine / Proton (Piano From Above / Windows Wasabi) Drop-In Placement

1. Copy `dist/windows/OmniMIDI.dll`, `dist/windows/kdmapiv3-config.exe`, and `dist/windows/kdmapiv3.toml` into the same folder as the Windows player executable (`PianoFromAbove.exe` or `wasabi.exe`), or place `OmniMIDI.dll` in `drive_c/windows/system32/`.
2. Launch the player with native DLL override enabled so Wine loads `OmniMIDI.dll` directly:

```bash
WINEDLLOVERRIDES="OmniMIDI=n" wine PianoFromAbove.exe
```

For Steam / Proton launch options:

```text
WINEDLLOVERRIDES="OmniMIDI=n" %command%
```

---

## Configuration (`kdmapiv3.toml`)

`kdmapiv3.toml` is loaded **only once during `InitializeKDMAPIStream`** (never on the hot path) and saved atomically (temp file + rename) by `kdmapiv3-config`:

```toml
voice_limit = 10000              # 1 - 1,000,000 (default: 10000)
interpolation = "linear"         # "linear" | "sinc8" | "sinc16" | "sinc32" (default: "linear")
buffer_frames = 512              # 32 - 8192 frames (default: 512)
render_threads = 4               # 1 - 256 DSP render threads (default: 4)
ring_buffer_capacity = 1048576   # Power of two, 1024 - 16777216 (default: 1,048,576)
soundfonts = [
    '/home/user/SoundFonts/KeppyGrand.sf2',
]
telemetry_enabled = false        # When true, writes 1s summaries to kdmapiv3.log

[effects]
enable_lowpass_filter = true
enable_vibrato_lfo = true
disable_all_effects = false
```

- **Legacy SoundFont compatibility:** Both `soundfonts` in `kdmapiv3.toml` and legacy `LoadCustomSoundFontsList` calls (`.sf2`, `.sfz`, `.sflist`, `.omlist`) work out of the box, including Windows backslashes, Wine drive prefixes, and case-insensitive filename matching on Linux filesystems.
- **Optional non-blocking diagnostics:** Setting `telemetry_enabled = true` spawns a separate low-priority background thread that reads atomic counters (`events_in`, `events_out`, `events_dropped`, `xrun_count`) once per second and appends summaries to `kdmapiv3.log`. `kdmapiv3-config` tails this file at 2Hz without blocking rendering.

---

## Note from the Author

Hi! I'm just an ordinary person living in a small city in South Korea.  
I deeply love and respect everyone in the Black MIDI community—original developers, MIDI creators, and players alike—and I hold no ill will toward anyone. This project was put together out of pure passion for exploring real-time playback limits, working right from the CLI :)

---

안녕하세요! 대한민국에 살고 있는 한 소도시 사람입니다.  
검은악보(Black MIDI) 관련자분들 모두를 사랑하고 존경하며, 어떠한 악감정도 없습니다. 순수한 애정으로 CLI 환경에서 작업했습니다 :)
