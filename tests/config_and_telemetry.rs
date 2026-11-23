//! Integration tests for `kdmapiv3.toml` auto-creation/loading, legacy SoundFont
//! list compatibility, and optional non-blocking background `kdmapiv3.log` telemetry.

use std::fs;
use std::thread;
use std::time::Duration;

use OmniMIDI::{
    active_config_snapshot, loaded_soundfont_paths, reset_telemetry,
    short_ring_capacity, telemetry_snapshot, InitializeKDMAPIStream, InterpolationMode,
    LoadCustomSoundFontsList, SendDirectData, TerminateKDMAPIStream, CONFIG_ENV_OVERRIDE,
    MMSYSERR_NOERROR, TELEMETRY_LOG_FILENAME,
};

#[test]
fn config_loading_soundfonts_and_nonblocking_telemetry_log() {
    let tmp_dir = std::env::temp_dir().join(format!(
        "kdmapiv3_cfg_telem_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&tmp_dir);
    let sf_dir = tmp_dir.join("SoundFonts");
    fs::create_dir_all(&sf_dir).expect("create temp sf dir");

    let sf1 = sf_dir.join("ConcertGrand.sf2");
    let sf2 = sf_dir.join("ExtraDrums.sfz");
    fs::write(&sf1, b"RIFF0000sfbk").expect("write sf1");
    fs::write(&sf2, b"<region> sample=kick.wav\n").expect("write sf2");

    let cfg_path = tmp_dir.join("kdmapiv3.toml");
    let log_path = tmp_dir.join(TELEMETRY_LOG_FILENAME);

    let win_sf1 = {
        let s = sf1.to_string_lossy().replace('/', "\\");
        #[cfg(target_os = "linux")]
        {
            format!("Z:{}", s.to_uppercase())
        }
        #[cfg(not(target_os = "linux"))]
        {
            s.to_uppercase()
        }
    };

    let custom_toml = format!(
        r#"
voice_limit = 16384
interpolation = "sinc16"
buffer_frames = 128
render_threads = 6
ring_buffer_capacity = 65536
soundfonts = [
    '{sf1}',
    '{sf2}',
]
telemetry_enabled = true
"#,
        sf1 = win_sf1,
        sf2 = sf2.display(),
    );
    fs::write(&cfg_path, custom_toml).expect("write custom kdmapiv3.toml");

    // Point kdmapiv3 at our temporary config file and headless audio mode for deterministic test.
    std::env::set_var(CONFIG_ENV_OVERRIDE, &cfg_path);
    std::env::set_var(OmniMIDI::NO_AUDIO_ENV, "1");

    assert_eq!(InitializeKDMAPIStream(), 1);
    reset_telemetry();

    let active_cfg = active_config_snapshot().expect("active config loaded");
    assert_eq!(active_cfg.voice_limit, 16_384);
    assert_eq!(active_cfg.interpolation, InterpolationMode::Sinc16);
    assert_eq!(active_cfg.buffer_frames, 128);
    assert_eq!(active_cfg.render_threads, 6);
    assert_eq!(active_cfg.ring_buffer_capacity, 65_536);
    assert!(active_cfg.telemetry_enabled);
    assert_eq!(short_ring_capacity(), 65_536);

    // SoundFonts from kdmapiv3.toml must be normalized & resolved case-insensitively in order.
    let loaded = loaded_soundfont_paths();
    assert_eq!(loaded, vec![sf1.clone(), sf2.clone()]);

    // Send a burst of MIDI events so events_in and events_out advance.
    for i in 0..5_000u32 {
        let msg = 0x0064_0090 | ((i & 0x7F) << 8);
        assert_eq!(SendDirectData(msg), MMSYSERR_NOERROR);
    }

    for _ in 0..10_000 {
        if telemetry_snapshot().events_out >= 5_000 {
            break;
        }
        thread::yield_now();
    }

    let snap = telemetry_snapshot();
    assert_eq!(snap.events_in, 5_000);
    assert_eq!(snap.events_out, 5_000);

    // Wait slightly over 1 second so the background telemetry worker flushes at least one tick.
    thread::sleep(Duration::from_millis(1_150));

    // Also test legacy .sflist loading via LoadCustomSoundFontsList(*const u16).
    let sflist_path = tmp_dir.join("legacy.omlist");
    let sflist_body = format!(
        "# Legacy OmniMIDI list\nsf.start\nsf.path = {}\nsf.enabled = 1\nsf.end\n{}\n",
        win_sf1,
        sf2.display()
    );
    fs::write(&sflist_path, sflist_body).expect("write legacy.omlist");
    let wide_list: Vec<u16> = format!("{}\0", sflist_path.display())
        .encode_utf16()
        .collect();
    assert_eq!(LoadCustomSoundFontsList(wide_list.as_ptr()), 1);
    assert_eq!(loaded_soundfont_paths(), vec![sf1.clone(), sf2.clone()]);

    assert_eq!(TerminateKDMAPIStream(), 1);

    let log_contents = fs::read_to_string(&log_path).expect("kdmapiv3.log should exist");
    assert!(log_contents.contains("events_in="));
    assert!(log_contents.contains("events_out="));
    assert!(log_contents.contains("events_dropped="));
    assert!(log_contents.contains("xrun_count="));

    std::env::remove_var(CONFIG_ENV_OVERRIDE);
    let _ = fs::remove_dir_all(&tmp_dir);
}
