//! External `kdmapiv3.toml` configuration loader and safe auto-generator.
//!
//! Strictly cold-path only: called exclusively during `InitializeKDMAPIStream`,
//! never on `SendDirectData`, `SendDirectDataNoBuf`, or the audio/consumer loop.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::backend::{
    InterpolationMode, SynthConfig, DEFAULT_BUFFER_FRAMES, DEFAULT_RENDER_THREADS,
    DEFAULT_SAMPLE_RATE, DEFAULT_VOICE_LIMIT, MAX_BUFFER_FRAMES, MAX_RENDER_THREADS,
    MAX_VOICE_LIMIT, MIN_BUFFER_FRAMES, MIN_RENDER_THREADS, MIN_VOICE_LIMIT,
};
use crate::ringbuf::DEFAULT_SHORT_RING_CAPACITY;

/// Default filename searched next to `libOmniMIDI.so` / `OmniMIDI.dll` or in user config dir.
pub const CONFIG_FILENAME: &str = "kdmapiv3.toml";

/// Default telemetry log filename created next to `kdmapiv3.toml` when `telemetry_enabled = true`.
pub const TELEMETRY_LOG_FILENAME: &str = "kdmapiv3.log";

/// Optional environment variable to override `kdmapiv3.toml` location explicitly.
pub const CONFIG_ENV_OVERRIDE: &str = "KDMAPIV3_CONFIG";

pub const MIN_RING_CAPACITY: usize = 1_024;
pub const MAX_RING_CAPACITY: usize = 16_777_216;

/// Well-commented default `kdmapiv3.toml` template (single Default Recommended Profile).
pub const DEFAULT_CONFIG_TEMPLATE: &str = r#"# kdmapiv3 configuration file
# Loaded once during InitializeKDMAPIStream (never touched on the audio/MIDI hot path).
#
# Placement priority:
#   1. $KDMAPIV3_CONFIG (optional environment variable override)
#   2. Next to libOmniMIDI.so / OmniMIDI.dll (portable drop-in next to Wasabi or PFA)
#   3. Linux:   ~/.config/kdmapiv3/kdmapiv3.toml
#      Windows: %APPDATA%\kdmapiv3\kdmapiv3.toml

# Maximum simultaneous voices for Black MIDI playback (1 - 1,000,000; default: 10000).
voice_limit = 10000

# Resampling interpolation quality: "linear", "sinc8", "sinc16", or "sinc32" (default: "linear").
interpolation = "linear"

# Audio render buffer size in frames (32 - 8192; default: 512).
buffer_frames = 512

# Total DSP render threads (1 = single-threaded, 2 - 256 = multi-threaded; default: 4).
render_threads = 4

# Pre-allocated lock-free MIDI ring buffer slots (power of two, 1024 - 16777216; default: 1048576).
ring_buffer_capacity = 1048576

# Ordered list of SoundFont (.sf2 / .sfz) paths.
# Windows backslashes, Wine Z:\ paths, and mismatched casing are normalized automatically.
# Example:
# soundfonts = [
#     '/home/user/SoundFonts/KeppyGrand.sf2',
#     'C:\SoundFonts\ExtraPercussion.sf2',
# ]
soundfonts = []

# Optional non-blocking diagnostics: when true, a separate low-priority background thread
# writes a 1-second atomic counter summary to kdmapiv3.log without touching the hot path.
telemetry_enabled = false

# Low-Spec / Black MIDI DSP Effect Toggles (zero hot-path overhead when disabled).
[effects]
enable_lowpass_filter = true
enable_vibrato_lfo = true
disable_all_effects = false
"#;

/// `[effects]` TOML table for optional DSP feature toggles.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct RawEffectsConfig {
    enable_lowpass_filter: bool,
    enable_vibrato_lfo: bool,
    disable_all_effects: bool,
}

impl Default for RawEffectsConfig {
    fn default() -> Self {
        Self {
            enable_lowpass_filter: true,
            enable_vibrato_lfo: true,
            disable_all_effects: false,
        }
    }
}

/// Raw TOML-deserializable config struct with safe defaults for every field.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct RawConfig {
    voice_limit: u32,
    interpolation: String,
    buffer_frames: u32,
    render_threads: u32,
    ring_buffer_capacity: usize,
    soundfonts: Vec<String>,
    telemetry_enabled: bool,
    effects: RawEffectsConfig,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            voice_limit: DEFAULT_VOICE_LIMIT,
            interpolation: InterpolationMode::default().as_str().to_string(),
            buffer_frames: DEFAULT_BUFFER_FRAMES,
            render_threads: DEFAULT_RENDER_THREADS,
            ring_buffer_capacity: DEFAULT_SHORT_RING_CAPACITY,
            soundfonts: Vec::new(),
            telemetry_enabled: false,
            effects: RawEffectsConfig::default(),
        }
    }
}

/// Validated runtime configuration loaded from `kdmapiv3.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdmapiConfig {
    pub voice_limit: u32,
    pub interpolation: InterpolationMode,
    pub buffer_frames: u32,
    pub render_threads: u32,
    pub ring_buffer_capacity: usize,
    pub soundfonts: Vec<String>,
    pub telemetry_enabled: bool,
    pub enable_lowpass_filter: bool,
    pub enable_vibrato_lfo: bool,
    pub disable_all_effects: bool,
    /// Location of the loaded or auto-created `kdmapiv3.toml` (if available).
    pub config_path: Option<PathBuf>,
    /// Target path for `kdmapiv3.log` when `telemetry_enabled = true`.
    pub log_path: PathBuf,
}

impl Default for KdmapiConfig {
    fn default() -> Self {
        let default_dir = platform::user_config_dir().unwrap_or_else(std::env::temp_dir);
        Self {
            voice_limit: DEFAULT_VOICE_LIMIT,
            interpolation: InterpolationMode::default(),
            buffer_frames: DEFAULT_BUFFER_FRAMES,
            render_threads: DEFAULT_RENDER_THREADS,
            ring_buffer_capacity: DEFAULT_SHORT_RING_CAPACITY,
            soundfonts: Vec::new(),
            telemetry_enabled: false,
            enable_lowpass_filter: true,
            enable_vibrato_lfo: true,
            disable_all_effects: false,
            config_path: None,
            log_path: default_dir.join(TELEMETRY_LOG_FILENAME),
        }
    }
}

impl KdmapiConfig {
    /// Convert validated config fields into backend [`SynthConfig`].
    pub fn to_synth_config(&self) -> SynthConfig {
        SynthConfig {
            sample_rate: DEFAULT_SAMPLE_RATE,
            max_voices: self.voice_limit,
            buffer_frames: self.buffer_frames,
            render_threads: self.render_threads,
            interpolation: self.interpolation,
            enable_lowpass_filter: self.enable_lowpass_filter,
            enable_vibrato_lfo: self.enable_vibrato_lfo,
            disable_all_effects: self.disable_all_effects,
        }
    }

    /// Parse TOML contents into a validated [`KdmapiConfig`] with safe clamping.
    pub fn parse_toml_str(contents: &str, config_path: Option<PathBuf>) -> Result<Self, toml::de::Error> {
        let raw: RawConfig = toml::from_str(contents)?;
        Ok(Self::from_raw(raw, config_path))
    }

    fn from_raw(raw: RawConfig, config_path: Option<PathBuf>) -> Self {
        let voice_limit = raw.voice_limit.clamp(MIN_VOICE_LIMIT, MAX_VOICE_LIMIT);
        let interpolation = InterpolationMode::parse(&raw.interpolation);
        let buffer_frames = raw.buffer_frames.clamp(MIN_BUFFER_FRAMES, MAX_BUFFER_FRAMES);
        let render_threads = raw
            .render_threads
            .clamp(MIN_RENDER_THREADS, MAX_RENDER_THREADS);
        let ring_buffer_capacity = raw
            .ring_buffer_capacity
            .clamp(MIN_RING_CAPACITY, MAX_RING_CAPACITY)
            .checked_next_power_of_two()
            .unwrap_or(DEFAULT_SHORT_RING_CAPACITY);

        let log_dir = config_path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .or_else(platform::user_config_dir)
            .unwrap_or_else(std::env::temp_dir);

        Self {
            voice_limit,
            interpolation,
            buffer_frames,
            render_threads,
            ring_buffer_capacity,
            soundfonts: raw.soundfonts,
            telemetry_enabled: raw.telemetry_enabled,
            enable_lowpass_filter: raw.effects.enable_lowpass_filter,
            enable_vibrato_lfo: raw.effects.enable_vibrato_lfo,
            disable_all_effects: raw.effects.disable_all_effects,
            config_path,
            log_path: log_dir.join(TELEMETRY_LOG_FILENAME),
        }
    }
}

/// Load `kdmapiv3.toml` from standard locations, or safely auto-create a default file.
///
/// Never panics and never fails initialization if the filesystem is read-only or
/// the user's TOML has a syntax error (gracefully falls back to safe defaults).
pub fn load_or_create_config() -> KdmapiConfig {
    if let Some(override_path) = std::env::var_os(CONFIG_ENV_OVERRIDE).filter(|v| !v.is_empty()) {
        return load_or_create_at_path(PathBuf::from(override_path));
    }

    let candidates = candidate_config_paths();

    // 1. Prefer any already-existing kdmapiv3.toml in priority order.
    for path in &candidates {
        if path.is_file() {
            return read_config_file_or_default(path);
        }
    }

    // 2. Otherwise auto-create a well-commented default config in the first writable candidate.
    for path in &candidates {
        if try_write_default_config(path) {
            return KdmapiConfig::from_raw(RawConfig::default(), Some(path.clone()));
        }
    }

    // 3. Fallback if all candidate directories are read-only.
    KdmapiConfig::default()
}

/// Load from a specific path, auto-creating `DEFAULT_CONFIG_TEMPLATE` if missing.
pub fn load_or_create_at_path(path: PathBuf) -> KdmapiConfig {
    if path.is_file() {
        return read_config_file_or_default(&path);
    }
    if try_write_default_config(&path) {
        KdmapiConfig::from_raw(RawConfig::default(), Some(path))
    } else {
        KdmapiConfig::from_raw(RawConfig::default(), Some(path))
    }
}

fn read_config_file_or_default(path: &Path) -> KdmapiConfig {
    match fs::read_to_string(path) {
        Ok(contents) => match KdmapiConfig::parse_toml_str(&contents, Some(path.to_path_buf())) {
            Ok(cfg) => cfg,
            Err(_) => KdmapiConfig::from_raw(RawConfig::default(), Some(path.to_path_buf())),
        },
        Err(_) => KdmapiConfig::from_raw(RawConfig::default(), Some(path.to_path_buf())),
    }
}

fn try_write_default_config(path: &Path) -> bool {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    fs::write(path, DEFAULT_CONFIG_TEMPLATE).is_ok()
}

fn candidate_config_paths() -> Vec<PathBuf> {
    let mut paths = Vec::with_capacity(2);
    if let Some(lib_dir) = platform::current_module_dir() {
        paths.push(lib_dir.join(CONFIG_FILENAME));
    }
    if let Some(user_dir) = platform::user_config_dir() {
        let user_cfg = user_dir.join(CONFIG_FILENAME);
        if !paths.iter().any(|p| p == &user_cfg) {
            paths.push(user_cfg);
        }
    }
    paths
}

/// Strictly isolated Linux config-directory and shared-object location helpers.
#[cfg(target_os = "linux")]
mod platform {
    use std::ffi::{c_void, CStr};
    use std::path::{Path, PathBuf};

    fn dladdr_anchor() {}

    /// Locate the directory containing `libOmniMIDI.so` using `libc::dladdr`.
    pub fn current_module_dir() -> Option<PathBuf> {
        let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::dladdr(dladdr_anchor as *const c_void, &mut info) };
        if rc == 0 || info.dli_fname.is_null() {
            return None;
        }
        // SAFETY: `dladdr` returned non-zero with a valid NUL-terminated path pointer.
        let cstr = unsafe { CStr::from_ptr(info.dli_fname) };
        let path_str = cstr.to_str().ok()?;
        let parent = Path::new(path_str).parent()?;
        if parent.as_os_str().is_empty() {
            None
        } else {
            Some(parent.to_path_buf())
        }
    }

    /// Resolve `$XDG_CONFIG_HOME/kdmapiv3` or `~/.config/kdmapiv3`.
    pub fn user_config_dir() -> Option<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(xdg).join("kdmapiv3"));
        }
        let home = std::env::var_os("HOME").filter(|v| !v.is_empty())?;
        Some(PathBuf::from(home).join(".config").join("kdmapiv3"))
    }
}

/// Strictly isolated Windows/Wine config-directory and DLL location helpers.
#[cfg(target_os = "windows")]
mod platform {
    use std::ffi::c_void;
    use std::path::{Path, PathBuf};
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };

    fn module_anchor() {}

    /// Locate the directory containing `OmniMIDI.dll` via `GetModuleHandleExW` + `GetModuleFileNameW`.
    pub fn current_module_dir() -> Option<PathBuf> {
        let mut hmod: HMODULE = std::ptr::null_mut();
        let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        let ok = unsafe {
            GetModuleHandleExW(flags, module_anchor as *const c_void as *const u16, &mut hmod)
        };
        if ok == 0 || hmod.is_null() {
            return None;
        }

        let mut buf = [0u16; 1024];
        let len = unsafe { GetModuleFileNameW(hmod, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        if len == 0 || len >= buf.len() {
            return None;
        }

        let path_str = String::from_utf16_lossy(&buf[..len]);
        let parent = Path::new(&path_str).parent()?;
        if parent.as_os_str().is_empty() {
            None
        } else {
            Some(parent.to_path_buf())
        }
    }

    /// Resolve `%APPDATA%\kdmapiv3` on Windows/Wine.
    pub fn user_config_dir() -> Option<PathBuf> {
        let appdata = std::env::var_os("APPDATA").filter(|v| !v.is_empty())?;
        Some(PathBuf::from(appdata).join("kdmapiv3"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_parses_cleanly() {
        let cfg = KdmapiConfig::parse_toml_str(DEFAULT_CONFIG_TEMPLATE, None)
            .expect("default template must be valid TOML");
        assert_eq!(cfg.voice_limit, DEFAULT_VOICE_LIMIT);
        assert_eq!(cfg.interpolation, InterpolationMode::Linear);
        assert_eq!(cfg.buffer_frames, DEFAULT_BUFFER_FRAMES);
        assert_eq!(cfg.render_threads, DEFAULT_RENDER_THREADS);
        assert_eq!(cfg.ring_buffer_capacity, 1_048_576);
        assert!(cfg.soundfonts.is_empty());
        assert!(!cfg.telemetry_enabled);
        assert!(cfg.enable_lowpass_filter);
        assert!(cfg.enable_vibrato_lfo);
        assert!(!cfg.disable_all_effects);
    }

    #[test]
    fn custom_toml_parses_and_clamps_power_of_two() {
        let sample = r#"
voice_limit = 250000
interpolation = "sinc16"
buffer_frames = 192
render_threads = 24
ring_buffer_capacity = 100000
soundfonts = [
    'C:\SoundFonts\Grand.sf2',
    '/home/user/Strings.sfz',
]
telemetry_enabled = true

[effects]
enable_lowpass_filter = false
enable_vibrato_lfo = true
disable_all_effects = true
"#;
        let cfg = KdmapiConfig::parse_toml_str(sample, None).expect("parse custom config");
        assert_eq!(cfg.voice_limit, 250_000);
        assert_eq!(cfg.interpolation, InterpolationMode::Sinc16);
        assert_eq!(cfg.buffer_frames, 192);
        assert_eq!(cfg.render_threads, 24);
        assert_eq!(cfg.ring_buffer_capacity, 131_072);
        assert_eq!(cfg.soundfonts.len(), 2);
        assert!(cfg.telemetry_enabled);
        assert!(!cfg.enable_lowpass_filter);
        assert!(cfg.enable_vibrato_lfo);
        assert!(cfg.disable_all_effects);
        let sc = cfg.to_synth_config();
        assert_eq!(sc.render_threads, 24);
        assert_eq!(sc.max_voices, 250_000);
        assert!(!sc.lowpass_filter_active());
        assert!(!sc.vibrato_lfo_active());
    }

    #[test]
    fn auto_creates_default_config_when_missing() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "kdmapiv3_cfg_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp_dir);
        let cfg_path = tmp_dir.join(CONFIG_FILENAME);

        let cfg = load_or_create_at_path(cfg_path.clone());
        assert!(cfg_path.is_file(), "kdmapiv3.toml should be auto-created");
        assert_eq!(cfg.voice_limit, DEFAULT_VOICE_LIMIT);
        assert_eq!(cfg.interpolation, InterpolationMode::Linear);
        assert_eq!(cfg.buffer_frames, DEFAULT_BUFFER_FRAMES);
        assert_eq!(cfg.render_threads, DEFAULT_RENDER_THREADS);
        assert_eq!(cfg.ring_buffer_capacity, 1_048_576);

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
