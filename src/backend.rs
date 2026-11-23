//! Modular synthesizer backend trait, shared engine configuration bounds,
//! and strictly isolated OS-specific helpers for Linux and Windows/Wine.

use std::path::{Path, PathBuf};

/// Single Source of Truth for engine bounds and the single Default Recommended Profile.
pub const MIN_VOICE_LIMIT: u32 = 1;
pub const MAX_VOICE_LIMIT: u32 = 1_000_000;
pub const DEFAULT_VOICE_LIMIT: u32 = 10_000;

pub const MIN_BUFFER_FRAMES: u32 = 32;
pub const MAX_BUFFER_FRAMES: u32 = 8_192;
pub const DEFAULT_BUFFER_FRAMES: u32 = 512;

pub const MIN_RENDER_THREADS: u32 = 1;
pub const MAX_RENDER_THREADS: u32 = 256;
pub const DEFAULT_RENDER_THREADS: u32 = 4;

pub const DEFAULT_SAMPLE_RATE: u32 = 48_000;

/// Resampling interpolation modes selectable via `kdmapiv3.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    Linear,
    Sinc8,
    Sinc16,
    Sinc32,
}

impl InterpolationMode {
    /// Parse `"linear"`, `"sinc8"`, `"sinc16"`, or `"sinc32"` (case-insensitive), defaulting to `Linear`.
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "sinc8" | "8pt" | "8" => Self::Sinc8,
            "sinc16" | "16pt" | "16" => Self::Sinc16,
            "sinc32" | "32pt" | "32" => Self::Sinc32,
            _ => Self::Linear,
        }
    }

    /// Number of filter taps associated with this interpolation mode.
    #[inline]
    pub const fn taps(self) -> u8 {
        match self {
            Self::Linear => 2,
            Self::Sinc8 => 8,
            Self::Sinc16 => 16,
            Self::Sinc32 => 32,
        }
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Sinc8 => "sinc8",
            Self::Sinc16 => "sinc16",
            Self::Sinc32 => "sinc32",
        }
    }
}

impl Default for InterpolationMode {
    fn default() -> Self {
        Self::Linear
    }
}

/// Synthesizer initialization parameters (single Default Recommended Profile + full manual control).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynthConfig {
    pub sample_rate: u32,
    pub max_voices: u32,
    pub buffer_frames: u32,
    /// Total DSP render thread count (`1` = single-threaded audio callback, `N >= 2` = audio thread + `N-1` workers).
    pub render_threads: u32,
    pub interpolation: InterpolationMode,
    /// Enable SF2 resonant low-pass filter & default velocity-to-filter modulator.
    pub enable_lowpass_filter: bool,
    /// Enable Vibrato LFO (CC1 Mod Wheel & SF2 `vibLfoToPitch`).
    pub enable_vibrato_lfo: bool,
    /// Master override for Low-Spec / Ultra-High Voice Black MIDI mode (forces all DSP effects off).
    pub disable_all_effects: bool,
}

impl SynthConfig {
    #[inline]
    pub const fn lowpass_filter_active(&self) -> bool {
        !self.disable_all_effects && self.enable_lowpass_filter
    }

    #[inline]
    pub const fn vibrato_lfo_active(&self) -> bool {
        !self.disable_all_effects && self.enable_vibrato_lfo
    }
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            max_voices: DEFAULT_VOICE_LIMIT,
            buffer_frames: DEFAULT_BUFFER_FRAMES,
            render_threads: DEFAULT_RENDER_THREADS,
            interpolation: InterpolationMode::Linear,
            enable_lowpass_filter: true,
            enable_vibrato_lfo: true,
            disable_all_effects: false,
        }
    }
}

/// Lock-free / O(1) snapshot of synthesizer status exposed to KDMAPI telemetry queries.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SynthStats {
    pub active_voices: u64,
    pub rendering_time_pct: f32,
    pub xrun_detected: bool,
}

/// Error codes returned by control-path backend operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendError {
    InvalidPath,
    NotInitialized,
    InitFailed,
    Unsupported,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::InvalidPath => write!(f, "invalid SoundFont path"),
            BackendError::NotInitialized => write!(f, "synthesizer backend not initialized"),
            BackendError::InitFailed => write!(f, "audio/synthesizer initialization failed"),
            BackendError::Unsupported => write!(f, "unsupported operation"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Modular synthesizer backend trait.
///
/// Future backends (XSynth, BASSMIDI, custom SIMD Sinc DSP) implement this trait.
/// Methods marked as hot path MUST NEVER allocate, take blocking locks, log, or panic.
pub trait SynthBackend: Send + 'static {
    /// Human-readable backend identifier.
    fn name(&self) -> &'static str;

    /// Control path: allocate resources and initialize the backend.
    fn init(&mut self, config: SynthConfig) -> Result<(), BackendError>;

    /// Control path: release audio/synth resources cleanly.
    fn shutdown(&mut self);

    /// Hot path: reset all active voices and channel controllers without allocating.
    fn reset(&mut self);

    /// Hot path: ingest a packed 32-bit KDMAPI short MIDI message (`status | (d1 << 8) | (d2 << 16)`).
    fn process_short_event(&mut self, dw_msg: u32);

    /// Hot path: ingest a raw SysEx / long MIDI message slice.
    fn process_sysex(&mut self, data: &[u8]);

    /// Hot path: ingest a KDMAPI custom event.
    fn process_custom_event(&mut self, event_type: u32, chan: u32, param: u32);

    /// Hot path: render interleaved stereo `f32` samples into a pre-allocated buffer.
    fn render_audio(&mut self, output: &mut [f32]);

    /// Control path: load a single SoundFont or SoundFont list path.
    fn load_soundfont_list(&mut self, path: &Path) -> Result<(), BackendError>;

    /// Control path: load an ordered list of resolved `.sf2`/`.sfz` SoundFont paths.
    fn load_soundfonts(&mut self, paths: &[PathBuf]) -> Result<(), BackendError> {
        if let Some(first) = paths.first() {
            self.load_soundfont_list(first)?;
        }
        Ok(())
    }

    /// O(1), allocation-free backend statistics snapshot.
    fn stats(&self) -> SynthStats;
}

/// Cross-platform UTF-16 decoding, Windows/Wine path normalization, legacy `.sflist`/`.omlist`
/// parsing, and case-insensitive `.sf2`/`.sfz` resolution helpers.
///
/// Cold control-path only (called by `InitializeKDMAPIStream` and `LoadCustomSoundFontsList`).
pub mod path_util {
    use std::path::{Path, PathBuf};

    /// Maximum UTF-16 code units scanned for a NUL terminator to prevent unbounded reads.
    pub const MAX_UTF16_PATH_UNITS: usize = 32_768;

    /// Safely decode a NUL-terminated UTF-16 (`PCWSTR` / `*const u16`) pointer using
    /// `String::from_utf16_lossy`, bounded by `max_units`.
    ///
    /// Returns `None` if `ptr` is null.
    ///
    /// # Safety
    /// If `ptr` is non-null, it must point to valid readable `u16` memory up to either
    /// the first `0u16` terminator or `max_units` elements.
    pub unsafe fn decode_utf16_ptr(ptr: *const u16, max_units: usize) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        let limit = max_units.min(MAX_UTF16_PATH_UNITS);
        let bytes = ptr as *const u8;
        let mut units = Vec::new();
        for i in 0..limit {
            let lo = *bytes.add(i * 2);
            let hi = *bytes.add(i * 2 + 1);
            let unit = u16::from_le_bytes([lo, hi]);
            if unit == 0 {
                break;
            }
            units.push(unit);
        }
        Some(String::from_utf16_lossy(&units))
    }

    /// Decode a KDMAPI path pointer supporting both Windows/Wine UTF-16LE (`PCWSTR`)
    /// and native Linux KDMAPI UTF-8 (`*const c_char`, used by `kdmapi-rs`/Wasabi and
    /// OmniMIDI v2 Linux).
    ///
    /// # Safety
    /// If `ptr` is non-null, it must point to a valid NUL-terminated UTF-16LE string
    /// (on Windows, or legacy callers on Linux) or a valid NUL-terminated UTF-8 C string
    /// (on Linux/Unix).
    pub unsafe fn decode_c_or_utf16_path_ptr(ptr: *const u16, max_units: usize) -> Option<String> {
        if ptr.is_null() {
            return None;
        }

        #[cfg(target_os = "windows")]
        {
            decode_utf16_ptr(ptr, max_units)
        }

        #[cfg(not(target_os = "windows"))]
        {
            let bytes = ptr as *const u8;
            let b0 = *bytes;
            if b0 == 0 {
                // Empty UTF-8 C string `""` (1 byte) or empty UTF-16LE `[0u16]`.
                return Some(String::new());
            }

            // `b0 != 0` guarantees at least 2 readable bytes in both UTF-8 and UTF-16LE.
            let b1 = *bytes.add(1);
            if b1 == 0 {
                // ASCII-starting UTF-16LE string (e.g. `Z:\...`, `/tmp/...`).
                return decode_utf16_ptr(ptr, max_units);
            }

            let limit = max_units.min(MAX_UTF16_PATH_UNITS);
            let mut len = 0usize;
            while len < limit {
                if *bytes.add(len) == 0 {
                    break;
                }
                len += 1;
            }
            let slice = std::slice::from_raw_parts(bytes, len);
            if let Ok(utf8) = std::str::from_utf8(slice) {
                return Some(utf8.to_owned());
            }

            // Fallback for non-ASCII-starting UTF-16LE strings.
            decode_utf16_ptr(ptr, max_units)
        }
    }

    /// Normalize Windows backslashes (`\`) to POSIX forward slashes (`/`).
    pub fn normalize_slashes(raw: &str) -> String {
        raw.replace('\\', "/")
    }

    /// Normalize a single SoundFont path and apply OS-specific path resolution (including
    /// Wine drive-prefix translation and case-insensitive `.sf2`/`.sfz` fallback on Linux).
    pub fn resolve_soundfont_path(raw: &str) -> PathBuf {
        let trimmed = raw.trim().trim_matches('"').trim_matches('\'');
        let normalized = normalize_slashes(trimmed);
        super::platform::resolve_platform_path(&normalized)
    }

    /// Resolve an input passed to `LoadCustomSoundFontsList`:
    /// - Supports direct `.sf2`/`.sfz` paths
    /// - Supports semicolon-separated inline path lists
    /// - Supports legacy OmniMIDI `.sflist` / `.omlist` / `.txt` list files (one path per line,
    ///   optional `sf.path = ...` syntax, `#` or `//` comments, relative or Windows/Wine paths)
    pub fn resolve_soundfont_input(raw: &str) -> Vec<PathBuf> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        // Check if caller passed multiple semicolon-separated paths inline.
        if trimmed.contains(';') {
            let mut out = Vec::new();
            for part in trimmed.split(';') {
                let p = part.trim();
                if !p.is_empty() {
                    out.push(resolve_soundfont_path(p));
                }
            }
            return out;
        }

        let resolved = resolve_soundfont_path(trimmed);
        if is_legacy_list_file(&resolved) && resolved.is_file() {
            if let Ok(contents) = std::fs::read_to_string(&resolved) {
                let base_dir = resolved.parent();
                let parsed = parse_legacy_soundfont_list(&contents, base_dir);
                if !parsed.is_empty() {
                    return parsed;
                }
            }
        }

        vec![resolved]
    }

    /// Parse legacy OmniMIDI `.sflist`/`.omlist` text content into resolved SoundFont paths.
    pub fn parse_legacy_soundfont_list(contents: &str, base_dir: Option<&Path>) -> Vec<PathBuf> {
        let mut results = Vec::new();
        for raw_line in contents.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            // Support OmniMIDI `sf.path = ...` or `sf.start` / `sf.end` block lines.
            if line.eq_ignore_ascii_case("sf.start") || line.eq_ignore_ascii_case("sf.end") {
                continue;
            }
            let entry = if let Some(rest) = strip_prefix_ignore_ascii_case(line, "sf.path") {
                let rest = rest.trim_start();
                if let Some(after_eq) = rest.strip_prefix('=') {
                    after_eq.trim()
                } else {
                    continue;
                }
            } else if line.contains('=') && line.starts_with("sf.") {
                // Skip non-path OmniMIDI list attributes (sf.enabled, sf.bank, etc.).
                continue;
            } else {
                line
            };

            if entry.is_empty() {
                continue;
            }

            let candidate = resolve_soundfont_path(entry);
            if candidate.is_relative() {
                if let Some(dir) = base_dir {
                    let joined = dir.join(&candidate);
                    results.push(resolve_case_insensitive(&joined));
                    continue;
                }
            }
            results.push(candidate);
        }
        results
    }

    fn strip_prefix_ignore_ascii_case<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
        if s.len() < prefix.len() {
            return None;
        }
        let head = s.get(..prefix.len())?;
        if head.eq_ignore_ascii_case(prefix) {
            s.get(prefix.len()..)
        } else {
            None
        }
    }

    fn is_legacy_list_file(path: &Path) -> bool {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => matches!(
                ext.to_ascii_lowercase().as_str(),
                "sflist" | "csflist" | "omlist" | "txt"
            ),
            None => false,
        }
    }

    /// Case-insensitive component-by-component lookup helper used for `.sf2`/`.sfz`
    /// resolution on case-sensitive filesystems (Linux/Wine).
    pub fn resolve_case_insensitive(path: &Path) -> PathBuf {
        if path.exists() {
            return path.to_path_buf();
        }

        let mut current = PathBuf::new();
        for comp in path.components() {
            use std::path::Component;
            match comp {
                Component::Prefix(p) => current.push(p.as_os_str()),
                Component::RootDir => current.push(comp.as_os_str()),
                Component::CurDir => current.push("."),
                Component::ParentDir => current.push(".."),
                Component::Normal(part) => {
                    let candidate = current.join(part);
                    if candidate.exists() {
                        current = candidate;
                        continue;
                    }

                    let target_lower = part.to_string_lossy().to_lowercase();
                    let search_dir = if current.as_os_str().is_empty() {
                        Path::new(".")
                    } else {
                        current.as_path()
                    };

                    let mut matched = None;
                    if let Ok(entries) = std::fs::read_dir(search_dir) {
                        for entry in entries.flatten() {
                            if entry.file_name().to_string_lossy().to_lowercase() == target_lower {
                                matched = Some(entry.path());
                                break;
                            }
                        }
                    }

                    match matched {
                        Some(found) => current = found,
                        None => current = candidate,
                    }
                }
            }
        }
        current
    }
}

/// Result of attempting to elevate the audio/MIDI consumer thread to real-time priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtPriorityResult {
    Elevated,
    FallbackUnprivileged,
}

/// Strictly isolated Linux-specific audio/RT and Wine path helpers.
#[cfg(target_os = "linux")]
pub mod platform {
    use super::RtPriorityResult;
    use std::path::{Path, PathBuf};

    /// Request `SCHED_FIFO` real-time scheduling for the calling thread on Linux,
    /// falling back safely without error when running unprivileged.
    pub fn configure_realtime_thread() -> RtPriorityResult {
        // SAFETY: `sched_get_priority_min`/`max` and `sched_setscheduler` on `pid == 0`
        // (current thread) are standard POSIX/Linux libc calls with a valid `sched_param`.
        unsafe {
            let min_prio = libc::sched_get_priority_min(libc::SCHED_FIFO);
            let max_prio = libc::sched_get_priority_max(libc::SCHED_FIFO);
            if min_prio < 0 || max_prio < min_prio {
                return RtPriorityResult::FallbackUnprivileged;
            }
            let target_prio = (min_prio + 10).min(max_prio);
            let param = libc::sched_param {
                sched_priority: target_prio,
            };
            if libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) == 0 {
                RtPriorityResult::Elevated
            } else {
                RtPriorityResult::FallbackUnprivileged
            }
        }
    }

    /// Translate Windows/Wine drive prefixes (`Z:/...`, `C:/...`) to Linux paths
    /// and apply case-insensitive filesystem fallback for `.sf2`/`.sfz` SoundFonts.
    pub fn resolve_platform_path(normalized: &str) -> PathBuf {
        let bytes = normalized.as_bytes();
        if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'/' {
            let drive_lower = bytes[0].to_ascii_lowercase() as char;
            let rest = &normalized[3..];

            if drive_lower == 'z' {
                // Wine maps Z:\ directly to Linux filesystem root `/`.
                let root_path = Path::new("/").join(rest);
                return super::path_util::resolve_case_insensitive(&root_path);
            }

            // Map other Wine drives (e.g. C:\...) via $WINEPREFIX/dosdevices/<drive>:
            // or $HOME/.wine/dosdevices/<drive>: if available.
            if let Some(wine_prefix) = wine_prefix_dir() {
                let dosdev = wine_prefix
                    .join("dosdevices")
                    .join(format!("{}:", drive_lower))
                    .join(rest);
                let resolved = super::path_util::resolve_case_insensitive(&dosdev);
                if resolved.exists() {
                    return resolved;
                }
                if drive_lower == 'c' {
                    let drive_c = wine_prefix.join("drive_c").join(rest);
                    let resolved_c = super::path_util::resolve_case_insensitive(&drive_c);
                    if resolved_c.exists() {
                        return resolved_c;
                    }
                }
            }
        }

        super::path_util::resolve_case_insensitive(Path::new(normalized))
    }

    fn wine_prefix_dir() -> Option<PathBuf> {
        if let Some(prefix) = std::env::var_os("WINEPREFIX").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(prefix));
        }
        let home = std::env::var_os("HOME").filter(|v| !v.is_empty())?;
        Some(PathBuf::from(home).join(".wine"))
    }
}

/// Strictly isolated Windows/Wine-specific audio/RT and path helpers.
#[cfg(target_os = "windows")]
pub mod platform {
    use super::RtPriorityResult;
    use std::path::{Path, PathBuf};
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
    };

    /// Request time-critical Win32 thread priority for the consumer thread.
    ///
    /// Intentionally avoids per-event `CreateEvent`/`SetEvent` kernel objects so
    /// Wine deployments do not incur `wineserver` IPC latency or XRUN spikes.
    pub fn configure_realtime_thread() -> RtPriorityResult {
        // SAFETY: `GetCurrentThread()` returns a valid pseudo-handle for the calling thread.
        unsafe {
            let handle = GetCurrentThread();
            if SetThreadPriority(handle, THREAD_PRIORITY_TIME_CRITICAL) != 0 {
                RtPriorityResult::Elevated
            } else {
                RtPriorityResult::FallbackUnprivileged
            }
        }
    }

    /// Resolve SoundFont paths on Windows/Wine, preserving drive prefixes while
    /// supporting normalized slashes and case-insensitive fallback under Wine mounts.
    pub fn resolve_platform_path(normalized: &str) -> PathBuf {
        super::path_util::resolve_case_insensitive(Path::new(normalized))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_backslash_and_legacy_sflist_parsing_work() {
        let wide: Vec<u16> = "Z:\\tmp\\Piano\\MyFont.SF2\0".encode_utf16().collect();
        let decoded = unsafe { path_util::decode_utf16_ptr(wide.as_ptr(), 1024) };
        assert_eq!(decoded.as_deref(), Some("Z:\\tmp\\Piano\\MyFont.SF2"));

        let normalized = path_util::normalize_slashes(decoded.as_deref().unwrap_or_default());
        assert_eq!(normalized, "Z:/tmp/Piano/MyFont.SF2");

        // Null pointer must return None safely.
        let null_res = unsafe { path_util::decode_utf16_ptr(std::ptr::null(), 1024) };
        assert!(null_res.is_none());

        let omlist = "\
# Legacy OmniMIDI SoundFont list
sf.start
sf.path = Z:\\tmp\\First.sf2
sf.enabled = 1
sf.end
// Plain line entry
/tmp/Second.sfz
";
        let entries = path_util::parse_legacy_soundfont_list(omlist, None);
        assert_eq!(entries.len(), 2);
        assert!(entries[0].to_string_lossy().ends_with("First.sf2"));
        assert!(entries[1].to_string_lossy().ends_with("Second.sfz"));
    }
}
