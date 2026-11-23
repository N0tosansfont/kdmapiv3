//! Configuration model, path discovery, SoundFont path normalization, and
//! atomic write-then-rename persistence for `kdmapiv3.toml`.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

pub const CONFIG_FILENAME: &str = "kdmapiv3.toml";
pub const TELEMETRY_LOG_FILENAME: &str = "kdmapiv3.log";
pub const CONFIG_ENV_OVERRIDE: &str = "KDMAPIV3_CONFIG";

// Synchronized with core `kdmapiv3::backend` Single Source of Truth bounds & Default Profile.
pub const VOICE_LIMIT_MIN: u32 = 1;
pub const VOICE_LIMIT_MAX: u32 = 1_000_000;
pub const DEFAULT_VOICE_LIMIT: u32 = 10_000;

pub const BUFFER_FRAMES_MIN: u32 = 32;
pub const BUFFER_FRAMES_MAX: u32 = 8_192;
pub const DEFAULT_BUFFER_FRAMES: u32 = 512;

pub const RENDER_THREADS_MIN: u32 = 1;
pub const RENDER_THREADS_MAX: u32 = 256;
pub const DEFAULT_RENDER_THREADS: u32 = 4;

pub const RING_CAPACITY_MIN: usize = 1_024;
pub const RING_CAPACITY_MAX: usize = 16_777_216;
pub const DEFAULT_RING_CAPACITY: usize = 1_048_576;

pub const BUFFER_FRAME_PRESETS: [u32; 8] = [32, 64, 128, 256, 512, 1024, 2048, 4096];
pub const RING_CAPACITY_PRESETS: [usize; 5] = [65_536, 262_144, 524_288, 1_048_576, 2_097_152];

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationChoice {
    Linear,
    Sinc8,
    Sinc16,
    Sinc32,
}

impl InterpolationChoice {
    pub const ALL: [Self; 4] = [Self::Linear, Self::Sinc8, Self::Sinc16, Self::Sinc32];

    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "sinc8" | "8pt" | "8" => Self::Sinc8,
            "sinc16" | "16pt" | "16" => Self::Sinc16,
            "sinc32" | "32pt" | "32" => Self::Sinc32,
            _ => Self::Linear,
        }
    }

    pub const fn toml_value(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Sinc8 => "sinc8",
            Self::Sinc16 => "sinc16",
            Self::Sinc32 => "sinc32",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear (Fastest)",
            Self::Sinc8 => "Sinc 8pt (Balanced)",
            Self::Sinc16 => "Sinc 16pt (Highest Quality)",
            Self::Sinc32 => "Sinc 32pt (Extreme)",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoundFontItem {
    pub path: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

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

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct RawTomlConfig {
    voice_limit: u32,
    interpolation: String,
    buffer_frames: u32,
    render_threads: u32,
    ring_buffer_capacity: usize,
    soundfonts: Vec<String>,
    soundfont_entries: Vec<SoundFontItem>,
    telemetry_enabled: bool,
    effects: RawEffectsConfig,
}

impl Default for RawTomlConfig {
    fn default() -> Self {
        Self {
            voice_limit: DEFAULT_VOICE_LIMIT,
            interpolation: "linear".to_string(),
            buffer_frames: DEFAULT_BUFFER_FRAMES,
            render_threads: DEFAULT_RENDER_THREADS,
            ring_buffer_capacity: DEFAULT_RING_CAPACITY,
            soundfonts: Vec::new(),
            soundfont_entries: Vec::new(),
            telemetry_enabled: false,
            effects: RawEffectsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorConfig {
    pub voice_limit: u32,
    pub interpolation: InterpolationChoice,
    pub buffer_frames: u32,
    pub render_threads: u32,
    pub ring_buffer_capacity: usize,
    pub soundfonts: Vec<SoundFontItem>,
    pub telemetry_enabled: bool,
    pub enable_lowpass_filter: bool,
    pub enable_vibrato_lfo: bool,
    pub disable_all_effects: bool,
}

impl Default for EditorConfig {
    fn default() -> Self {
        Self {
            voice_limit: DEFAULT_VOICE_LIMIT,
            interpolation: InterpolationChoice::Linear,
            buffer_frames: DEFAULT_BUFFER_FRAMES,
            render_threads: DEFAULT_RENDER_THREADS,
            ring_buffer_capacity: DEFAULT_RING_CAPACITY,
            soundfonts: Vec::new(),
            telemetry_enabled: false,
            enable_lowpass_filter: true,
            enable_vibrato_lfo: true,
            disable_all_effects: false,
        }
    }
}

impl EditorConfig {
    /// Parse `kdmapiv3.toml` content into [`EditorConfig`], reconciling both
    /// `soundfonts` (active ordered paths) and optional `soundfont_entries`
    /// (preserves disabled SoundFont positions across GUI sessions).
    pub fn from_toml_str(contents: &str) -> Result<Self, String> {
        let raw: RawTomlConfig =
            toml::from_str(contents).map_err(|e| format!("TOML parse error: {}", e))?;

        let voice_limit = raw.voice_limit.clamp(VOICE_LIMIT_MIN, VOICE_LIMIT_MAX);
        let interpolation = InterpolationChoice::parse(&raw.interpolation);
        let buffer_frames = raw.buffer_frames.clamp(BUFFER_FRAMES_MIN, BUFFER_FRAMES_MAX);
        let render_threads = raw
            .render_threads
            .clamp(RENDER_THREADS_MIN, RENDER_THREADS_MAX);
        let ring_buffer_capacity = raw
            .ring_buffer_capacity
            .clamp(RING_CAPACITY_MIN, RING_CAPACITY_MAX)
            .checked_next_power_of_two()
            .unwrap_or(DEFAULT_RING_CAPACITY);

        let active_from_list: Vec<String> = raw
            .soundfonts
            .into_iter()
            .map(|p| normalize_path_separators(&p))
            .filter(|p| !p.is_empty())
            .collect();

        let soundfonts = if !raw.soundfont_entries.is_empty() {
            let entries: Vec<SoundFontItem> = raw
                .soundfont_entries
                .into_iter()
                .map(|item| SoundFontItem {
                    path: normalize_path_separators(&item.path),
                    enabled: item.enabled,
                })
                .filter(|item| !item.path.is_empty())
                .collect();

            let enabled_from_entries: Vec<&str> = entries
                .iter()
                .filter(|e| e.enabled)
                .map(|e| e.path.as_str())
                .collect();
            let active_refs: Vec<&str> = active_from_list.iter().map(String::as_str).collect();

            // If user manually edited `soundfonts = [...]`, prefer their manual edits.
            if enabled_from_entries == active_refs {
                entries
            } else {
                active_from_list
                    .into_iter()
                    .map(|path| SoundFontItem {
                        path,
                        enabled: true,
                    })
                    .collect()
            }
        } else {
            active_from_list
                .into_iter()
                .map(|path| SoundFontItem {
                    path,
                    enabled: true,
                })
                .collect()
        };

        Ok(Self {
            voice_limit,
            interpolation,
            buffer_frames,
            render_threads,
            ring_buffer_capacity,
            soundfonts,
            telemetry_enabled: raw.telemetry_enabled,
            enable_lowpass_filter: raw.effects.enable_lowpass_filter,
            enable_vibrato_lfo: raw.effects.enable_vibrato_lfo,
            disable_all_effects: raw.effects.disable_all_effects,
        })
    }

    /// Serialize cleanly to a well-commented `kdmapiv3.toml` document.
    ///
    /// Always writes active enabled paths to `soundfonts = [...]` (100% compatible with
    /// the core `OmniMIDI` engine) and persists `[[soundfont_entries]]` when any disabled
    /// entry exists so toggle state and priority order are preserved on reload.
    pub fn to_toml_string(&self) -> String {
        let mut out = String::with_capacity(1024);
        out.push_str("# kdmapiv3 configuration file\n");
        out.push_str("# Managed by kdmapiv3-config (loaded by OmniMIDI only during InitializeKDMAPIStream).\n\n");

        out.push_str(&format!(
            "voice_limit = {}\n",
            self.voice_limit.clamp(VOICE_LIMIT_MIN, VOICE_LIMIT_MAX)
        ));
        out.push_str(&format!(
            "interpolation = \"{}\"\n",
            self.interpolation.toml_value()
        ));
        out.push_str(&format!(
            "buffer_frames = {}\n",
            self.buffer_frames
                .clamp(BUFFER_FRAMES_MIN, BUFFER_FRAMES_MAX)
        ));
        out.push_str(&format!(
            "render_threads = {}\n",
            self.render_threads
                .clamp(RENDER_THREADS_MIN, RENDER_THREADS_MAX)
        ));
        out.push_str(&format!(
            "ring_buffer_capacity = {}\n",
            self.ring_buffer_capacity
                .clamp(RING_CAPACITY_MIN, RING_CAPACITY_MAX)
                .checked_next_power_of_two()
                .unwrap_or(DEFAULT_RING_CAPACITY)
        ));
        out.push_str(&format!(
            "telemetry_enabled = {}\n\n",
            self.telemetry_enabled
        ));

        let enabled_paths: Vec<String> = self
            .soundfonts
            .iter()
            .filter(|sf| sf.enabled)
            .map(|sf| normalize_path_separators(&sf.path))
            .filter(|p| !p.is_empty())
            .collect();

        if enabled_paths.is_empty() {
            out.push_str("soundfonts = []\n");
        } else {
            out.push_str("soundfonts = [\n");
            for path in &enabled_paths {
                out.push_str(&format!("    {},\n", quote_toml_string(path)));
            }
            out.push_str("]\n");
        }

        let has_disabled = self.soundfonts.iter().any(|sf| !sf.enabled);
        if has_disabled {
            out.push_str("\n# Preserves disabled SoundFont entries & priority order for kdmapiv3-config.\n");
            for sf in &self.soundfonts {
                let norm = normalize_path_separators(&sf.path);
                if norm.is_empty() {
                    continue;
                }
                out.push_str("[[soundfont_entries]]\n");
                out.push_str(&format!("path = {}\n", quote_toml_string(&norm)));
                out.push_str(&format!("enabled = {}\n", sf.enabled));
            }
        }

        out.push_str("\n# Low-Spec / Black MIDI DSP Effect Toggles (zero hot-path overhead when disabled).\n");
        out.push_str("[effects]\n");
        out.push_str(&format!(
            "enable_lowpass_filter = {}\n",
            self.enable_lowpass_filter
        ));
        out.push_str(&format!(
            "enable_vibrato_lfo = {}\n",
            self.enable_vibrato_lfo
        ));
        out.push_str(&format!(
            "disable_all_effects = {}\n",
            self.disable_all_effects
        ));

        out
    }
}

/// Normalize Windows backslashes (`\`) to clean forward slashes (`/`), trimming quotes/whitespace.
pub fn normalize_path_separators(raw: &str) -> String {
    raw.trim()
        .trim_matches('"')
        .trim_matches('\'')
        .replace('\\', "/")
}

/// Quote a string safely as a TOML basic string.
fn quote_toml_string(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() + 2);
    escaped.push('"');
    for ch in s.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

/// Discover the primary `kdmapiv3.toml` path using the same priority as the core engine:
/// 1. `$KDMAPIV3_CONFIG`
/// 2. Existing `kdmapiv3.toml` next to `kdmapiv3-config` executable
/// 3. Existing user config dir (`~/.config/kdmapiv3/kdmapiv3.toml` or `%APPDATA%\kdmapiv3\kdmapiv3.toml`)
/// 4. Portable path next to executable if writable, else user config dir
pub fn discover_config_path() -> PathBuf {
    if let Some(env_path) = std::env::var_os(CONFIG_ENV_OVERRIDE).filter(|v| !v.is_empty()) {
        return PathBuf::from(env_path);
    }

    let exe_cfg = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(CONFIG_FILENAME)));
    let user_cfg = platform::user_config_dir().map(|d| d.join(CONFIG_FILENAME));

    if let Some(ref p) = exe_cfg {
        if p.is_file() {
            return p.clone();
        }
    }
    if let Some(ref p) = user_cfg {
        if p.is_file() {
            return p.clone();
        }
    }

    // Prefer portable config next to the binary when its directory is writable.
    if let Some(ref p) = exe_cfg {
        if p.parent().map(is_dir_writable).unwrap_or(false) {
            return p.clone();
        }
    }

    user_cfg
        .or(exe_cfg)
        .unwrap_or_else(|| PathBuf::from(CONFIG_FILENAME))
}

/// Resolve the corresponding `kdmapiv3.log` path next to `config_path`.
pub fn log_path_for_config(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|dir| dir.join(TELEMETRY_LOG_FILENAME))
        .unwrap_or_else(|| PathBuf::from(TELEMETRY_LOG_FILENAME))
}

fn is_dir_writable(dir: &Path) -> bool {
    let probe = dir.join(format!(".kdmapiv3_probe_{}", std::process::id()));
    match File::create(&probe) {
        Ok(_) => {
            let _ = fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Load `EditorConfig` from disk; returns defaults if the file does not exist yet.
pub fn load_config_file(path: &Path) -> Result<(EditorConfig, bool), String> {
    if !path.is_file() {
        return Ok((EditorConfig::default(), false));
    }
    let contents = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    let cfg = EditorConfig::from_toml_str(&contents)?;
    Ok((cfg, true))
}

/// Atomically persist `kdmapiv3.toml` using write-to-temp-file + fsync + atomic rename
/// so the audio engine can never observe a partially written file.
pub fn save_config_atomic(path: &Path, cfg: &EditorConfig) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    fs::create_dir_all(parent)
        .map_err(|e| format!("Failed to create config directory {}: {}", parent.display(), e))?;

    let seq = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(CONFIG_FILENAME);
    let tmp_path = parent.join(format!(".{}.tmp.{}.{}", file_name, std::process::id(), seq));

    let payload = cfg.to_toml_string();

    {
        let mut tmp_file = File::create(&tmp_path)
            .map_err(|e| format!("Failed to create temp config {}: {}", tmp_path.display(), e))?;
        if let Err(e) = tmp_file.write_all(payload.as_bytes()) {
            let _ = fs::remove_file(&tmp_path);
            return Err(format!("Failed to write temp config: {}", e));
        }
        if let Err(e) = tmp_file.sync_all() {
            let _ = fs::remove_file(&tmp_path);
            return Err(format!("Failed to flush temp config: {}", e));
        }
    }

    if let Err(e) = platform::atomic_replace(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(e);
    }

    Ok(())
}

/// Check if a dropped or selected path has a `.sf2` or `.sfz` extension (case-insensitive).
pub fn is_supported_soundfont(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "sf2" | "sfz"),
        None => false,
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::fs;
    use std::path::{Path, PathBuf};

    pub fn user_config_dir() -> Option<PathBuf> {
        if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
            return Some(PathBuf::from(xdg).join("kdmapiv3"));
        }
        let home = std::env::var_os("HOME").filter(|v| !v.is_empty())?;
        Some(PathBuf::from(home).join(".config").join("kdmapiv3"))
    }

    pub fn atomic_replace(tmp_path: &Path, dst_path: &Path) -> Result<(), String> {
        fs::rename(tmp_path, dst_path).map_err(|e| {
            format!(
                "Failed to atomically replace {} with {}: {}",
                dst_path.display(),
                tmp_path.display(),
                e
            )
        })
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    pub fn user_config_dir() -> Option<PathBuf> {
        let appdata = std::env::var_os("APPDATA").filter(|v| !v.is_empty())?;
        Some(PathBuf::from(appdata).join("kdmapiv3"))
    }

    pub fn atomic_replace(tmp_path: &Path, dst_path: &Path) -> Result<(), String> {
        let src_wide: Vec<u16> = tmp_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let dst_wide: Vec<u16> = dst_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        // SAFETY: Both pointers are NUL-terminated UTF-16 buffers valid for the call duration.
        let ok = unsafe {
            MoveFileExW(
                src_wide.as_ptr(),
                dst_wide.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        };
        if ok != 0 {
            Ok(())
        } else {
            Err(format!(
                "MoveFileExW failed replacing {}: {}",
                dst_path.display(),
                std::io::Error::last_os_error()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_save_and_roundtrip_preserve_disabled_soundfonts() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "kdmapiv3_gui_cfg_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp_dir);
        let cfg_path = tmp_dir.join(CONFIG_FILENAME);

        let cfg = EditorConfig {
            voice_limit: 250_000,
            interpolation: InterpolationChoice::Sinc32,
            buffer_frames: 192,
            render_threads: 24,
            ring_buffer_capacity: 4_194_304,
            soundfonts: vec![
                SoundFontItem {
                    path: "C:\\SoundFonts\\Grand.sf2".to_string(),
                    enabled: true,
                },
                SoundFontItem {
                    path: "Z:\\home\\user\\Backup.sfz".to_string(),
                    enabled: false,
                },
            ],
            telemetry_enabled: true,
            enable_lowpass_filter: false,
            enable_vibrato_lfo: true,
            disable_all_effects: true,
        };

        save_config_atomic(&cfg_path, &cfg).expect("atomic save should succeed");
        let raw = fs::read_to_string(&cfg_path).expect("read saved toml");
        assert!(raw.contains("\"C:/SoundFonts/Grand.sf2\""));
        assert!(!raw.contains("C:\\SoundFonts"));
        assert!(raw.contains("render_threads = 24"));
        assert!(raw.contains("[effects]"));
        assert!(raw.contains("disable_all_effects = true"));

        let (reloaded, existed) = load_config_file(&cfg_path).expect("reload saved toml");
        assert!(existed);
        assert_eq!(reloaded.voice_limit, 250_000);
        assert_eq!(reloaded.interpolation, InterpolationChoice::Sinc32);
        assert_eq!(reloaded.buffer_frames, 192);
        assert_eq!(reloaded.render_threads, 24);
        assert_eq!(reloaded.ring_buffer_capacity, 4_194_304);
        assert!(reloaded.telemetry_enabled);
        assert!(!reloaded.enable_lowpass_filter);
        assert!(reloaded.enable_vibrato_lfo);
        assert!(reloaded.disable_all_effects);

        // Verify custom non-preset buffer_frames and ring_buffer_capacity are NEVER snapped.
        let cfg_custom =
            EditorConfig::from_toml_str("buffer_frames = 384\nring_buffer_capacity = 65536\n")
                .expect("parse custom values");
        assert_eq!(cfg_custom.buffer_frames, 384);
        assert_eq!(cfg_custom.ring_buffer_capacity, 65_536);
        assert_eq!(reloaded.soundfonts.len(), 2);
        assert_eq!(reloaded.soundfonts[0].path, "C:/SoundFonts/Grand.sf2");
        assert!(reloaded.soundfonts[0].enabled);
        assert_eq!(reloaded.soundfonts[1].path, "Z:/home/user/Backup.sfz");
        assert!(!reloaded.soundfonts[1].enabled);

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
