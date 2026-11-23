use std::ffi::c_void;
use std::fs;
use std::mem::size_of;
use std::ptr;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use OmniMIDI::{
    last_loaded_soundfont_path, pending_short_events, reset_telemetry, telemetry_snapshot,
    DriverSettings, GetRenderingTime, GetVoiceCount, InitializeCallbackFeatures,
    InitializeKDMAPIStream, IsKDMAPIAvailable, LoadCustomSoundFontsList, MidiHdr, PrepareLongData,
    ResetKDMAPIStream, ReturnKDMAPIVer, RunCallbackFunction, SendCustomEvent, SendDirectData,
    SendDirectDataNoBuf, SendDirectLongData, SendDirectLongDataNoBuf, TerminateKDMAPIStream,
    UnprepareLongData, KDMAPI_VER_BUILD, KDMAPI_VER_MAJOR, KDMAPI_VER_MINOR, KDMAPI_VER_REVISION,
    MHDR_DONE, MHDR_INQUEUE, MHDR_PREPARED, MMSYSERR_INVALPARAM, MMSYSERR_NOERROR,
};

// Serialize tests that touch the process-global KDMAPI stream lifecycle.
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn version_availability_and_null_pointer_safety() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    assert_eq!(IsKDMAPIAvailable(), 1);

    let (mut maj, mut min, mut bld, mut rev) = (0u32, 0u32, 0u32, 0u32);
    assert_eq!(ReturnKDMAPIVer(&mut maj, &mut min, &mut bld, &mut rev), 1);
    assert_eq!(
        (maj, min, bld, rev),
        (
            KDMAPI_VER_MAJOR,
            KDMAPI_VER_MINOR,
            KDMAPI_VER_BUILD,
            KDMAPI_VER_REVISION
        )
    );

    // Null pointer checks must not crash and must return 0.
    assert_eq!(
        ReturnKDMAPIVer(ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
        0
    );
    let mut partial_maj = 99u32;
    assert_eq!(
        ReturnKDMAPIVer(&mut partial_maj, ptr::null_mut(), ptr::null_mut(), ptr::null_mut()),
        0
    );
    assert_eq!(partial_maj, KDMAPI_VER_MAJOR);
}

#[test]
fn long_data_and_driver_settings_validate_pointers() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var(OmniMIDI::NO_AUDIO_ENV, "1");

    assert_eq!(InitializeKDMAPIStream(), 1);

    // Null MIDIHDR and undersized header must return MMSYSERR_INVALPARAM.
    assert_eq!(
        SendDirectLongData(ptr::null_mut(), size_of::<MidiHdr>() as u32),
        MMSYSERR_INVALPARAM
    );
    assert_eq!(
        SendDirectLongDataNoBuf(ptr::null_mut(), size_of::<MidiHdr>() as u32),
        MMSYSERR_INVALPARAM
    );
    assert_eq!(
        PrepareLongData(ptr::null_mut(), size_of::<MidiHdr>() as u32),
        MMSYSERR_INVALPARAM
    );
    assert_eq!(
        UnprepareLongData(ptr::null_mut(), size_of::<MidiHdr>() as u32),
        MMSYSERR_INVALPARAM
    );

    // Valid GM System On SysEx payload.
    let mut sysex = [0xF0u8, 0x7E, 0x7F, 0x09, 0x01, 0xF7];
    let mut hdr = MidiHdr {
        lp_data: sysex.as_mut_ptr(),
        dw_buffer_length: sysex.len() as u32,
        dw_bytes_recorded: sysex.len() as u32,
        dw_user: 0,
        dw_flags: MHDR_INQUEUE,
        lp_next: ptr::null_mut(),
        reserved: 0,
        dw_offset: 0,
        dw_reserved: [0; 8],
    };

    // Too-small header size rejected safely.
    assert_eq!(SendDirectLongData(&mut hdr, 4), MMSYSERR_INVALPARAM);

    assert_eq!(
        PrepareLongData(&mut hdr, size_of::<MidiHdr>() as u32),
        MMSYSERR_NOERROR
    );
    assert_ne!(hdr.dw_flags & MHDR_PREPARED, 0);

    assert_eq!(
        SendDirectLongData(&mut hdr, size_of::<MidiHdr>() as u32),
        MMSYSERR_NOERROR
    );
    assert_ne!(hdr.dw_flags & MHDR_DONE, 0);
    assert_eq!(hdr.dw_flags & MHDR_INQUEUE, 0);

    assert_eq!(
        UnprepareLongData(&mut hdr, size_of::<MidiHdr>() as u32),
        MMSYSERR_NOERROR
    );
    assert_eq!(hdr.dw_flags & MHDR_PREPARED, 0);

    // Non-zero length with null lp_data rejected safely.
    hdr.lp_data = ptr::null_mut();
    assert_eq!(
        SendDirectLongData(&mut hdr, size_of::<MidiHdr>() as u32),
        MMSYSERR_INVALPARAM
    );

    // DriverSettings null check + round-trip.
    assert_eq!(
        DriverSettings(OmniMIDI::ffi::KDMAPI_MAXVOICES, OmniMIDI::ffi::KDMAPI_GET, ptr::null_mut(), 4),
        0
    );
    let mut voices_cfg: u32 = 16_384;
    assert_eq!(
        DriverSettings(
            OmniMIDI::ffi::KDMAPI_MAXVOICES,
            OmniMIDI::ffi::KDMAPI_SET,
            (&mut voices_cfg as *mut u32).cast::<c_void>(),
            4
        ),
        1
    );
    let mut read_back: u32 = 0;
    assert_eq!(
        DriverSettings(
            OmniMIDI::ffi::KDMAPI_MAXVOICES,
            OmniMIDI::ffi::KDMAPI_GET,
            (&mut read_back as *mut u32).cast::<c_void>(),
            4
        ),
        1
    );
    assert_eq!(read_back, 16_384);

    assert_eq!(
        InitializeCallbackFeatures(ptr::null_mut(), 0, 0, 0, 0),
        1
    );
    RunCallbackFunction(0, 0, 0);
    assert_eq!(TerminateKDMAPIStream(), 1);
}

#[test]
fn utf16_soundfont_normalization_and_case_insensitive_lookup() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    // Null and empty UTF-16 pointers rejected safely.
    assert_eq!(LoadCustomSoundFontsList(ptr::null()), 0);
    let empty_utf16: [u16; 1] = [0];
    assert_eq!(LoadCustomSoundFontsList(empty_utf16.as_ptr()), 0);

    // Create a temporary SoundFont file with mixed casing and query it via
    // Windows backslashes + mismatched case (plus Wine Z:\ prefix on Linux).
    let tmp_base = std::env::temp_dir().join(format!("kdmapiv3_sf_test_{}", std::process::id()));
    let sf_dir = tmp_base.join("SoundFonts");
    let sf_file = sf_dir.join("BlackMidiGrand.sf2");
    fs::create_dir_all(&sf_dir).expect("create temp sf dir");
    fs::write(&sf_file, b"RIFF0000sfbk").expect("write dummy sf2");

    let raw_win_style = {
        let base_str = tmp_base.to_string_lossy().replace('/', "\\");
        #[cfg(target_os = "linux")]
        {
            format!("Z:{}\\soundfonts\\blackmidigrand.SF2\0", base_str)
        }
        #[cfg(not(target_os = "linux"))]
        {
            format!("{}\\soundfonts\\blackmidigrand.SF2\0", base_str)
        }
    };

    let wide: Vec<u16> = raw_win_style.encode_utf16().collect();
    assert_eq!(LoadCustomSoundFontsList(wide.as_ptr()), 1);

    let resolved = last_loaded_soundfont_path().expect("soundfont path recorded");
    assert_eq!(resolved, sf_file);

    let _ = fs::remove_dir_all(&tmp_base);
}

#[test]
fn multithreaded_high_rate_burst_is_lockfree_and_panicfree() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var(OmniMIDI::NO_AUDIO_ENV, "1");

    assert_eq!(InitializeKDMAPIStream(), 1);
    reset_telemetry();

    let threads = 4usize;
    let events_per_thread = 100_000usize;

    let handles: Vec<_> = (0..threads)
        .map(|tid| {
            thread::spawn(move || {
                for i in 0..events_per_thread {
                    let note = ((i + tid) & 0x7F) as u32;
                    let ch = (tid & 0x0F) as u32;
                    let vel = if (i & 1) == 0 { 100u32 } else { 0u32 };
                    let msg = (vel << 16) | (note << 8) | (0x90 | ch);
                    if (i & 3) == 0 {
                        assert_eq!(SendDirectDataNoBuf(msg), MMSYSERR_NOERROR);
                    } else {
                        assert_eq!(SendDirectData(msg), MMSYSERR_NOERROR);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("producer thread must not panic");
    }

    assert_eq!(SendCustomEvent(1, 0, 42), 1);
    ResetKDMAPIStream();

    // Wait briefly for consumer to drain.
    for _ in 0..200 {
        if pending_short_events() == 0 {
            break;
        }
        thread::sleep(Duration::from_millis(1));
    }

    let snap = telemetry_snapshot();
    assert!(snap.events_in >= (threads * events_per_thread) as u64);
    assert!(GetRenderingTime() >= 0.0);
    let _voices = GetVoiceCount();

    assert_eq!(TerminateKDMAPIStream(), 1);
    // Double terminate must be safe and idempotent.
    assert_eq!(TerminateKDMAPIStream(), 1);
}
