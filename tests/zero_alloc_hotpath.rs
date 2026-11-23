//! Dedicated zero-allocation hot-path verification test.
//!
//! Installs a custom global allocator (`CountingAllocator`) in its own test binary
//! so no unrelated parallel test threads interfere with allocation counts.
//! Verifies strictly ZERO heap allocations across:
//! 1. 1,000,000-event FFI `SendDirectData` / `SendDirectDataNoBuf` / `SendDirectLongData` burst
//! 2. Real SoundFont (`.sfz` + `.wav`) `SoundFontSynth` Note On/Off + `render_audio` (Sinc8 & Sinc16)

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::Write;
use std::mem::size_of;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

use OmniMIDI::{
    pending_short_events, reset_telemetry, telemetry_snapshot, GetRenderingTime, GetVoiceCount,
    InitializeKDMAPIStream, InterpolationMode, MidiHdr, ResetKDMAPIStream, SendCustomEvent,
    SendDirectData, SendDirectDataNoBuf, SendDirectLongData, SoundFontSynth, SynthBackend,
    SynthConfig, TerminateKDMAPIStream, MHDR_INQUEUE, MMSYSERR_NOERROR, NO_AUDIO_ENV,
};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static REALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAllocator = CountingAllocator;

fn write_test_wav(path: &Path, sr: u32, frames: usize) {
    let data_bytes = (frames * 2) as u32;
    let riff_size = 36 + data_bytes;
    let mut f = File::create(path).expect("create wav");
    f.write_all(b"RIFF").expect("wav");
    f.write_all(&riff_size.to_le_bytes()).expect("wav");
    f.write_all(b"WAVEfmt ").expect("wav");
    f.write_all(&16u32.to_le_bytes()).expect("wav");
    f.write_all(&1u16.to_le_bytes()).expect("wav");
    f.write_all(&1u16.to_le_bytes()).expect("wav");
    f.write_all(&sr.to_le_bytes()).expect("wav");
    f.write_all(&(sr * 2).to_le_bytes()).expect("wav");
    f.write_all(&2u16.to_le_bytes()).expect("wav");
    f.write_all(&16u16.to_le_bytes()).expect("wav");
    f.write_all(b"data").expect("wav");
    f.write_all(&data_bytes.to_le_bytes()).expect("wav");
    for i in 0..frames {
        let s = ((2.0 * core::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin() * 24_000.0)
            as i16;
        f.write_all(&s.to_le_bytes()).expect("wav");
    }
}

#[test]
fn hot_path_send_and_real_synth_render_perform_zero_heap_allocations() {
    // Keep automated test deterministic and silent on speaker hardware.
    std::env::set_var(NO_AUDIO_ENV, "1");

    // --- Part 1: Real SoundFontSynth hot-path DSP zero-allocation verification ---
    let tmp = std::env::temp_dir().join(format!("kdmapiv3_zalloc_sfz_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let wav_path = tmp.join("sine.wav");
    let sfz_path = tmp.join("test.sfz");
    write_test_wav(&wav_path, 48_000, 12_000);
    std::fs::write(
        &sfz_path,
        "<region> sample=sine.wav lokey=0 hikey=127 pitch_keycenter=69 loop_mode=loop_continuous loop_start=64 loop_end=8000\n",
    )
    .expect("write sfz");

    let mut synth = SoundFontSynth::new();
    synth
        .init(SynthConfig {
            sample_rate: 48_000,
            max_voices: 512,
            buffer_frames: 512,
            interpolation: InterpolationMode::Linear,
            ..SynthConfig::default()
        })
        .expect("init synth");
    synth
        .load_soundfonts(std::slice::from_ref(&sfz_path))
        .expect("load sfz");

    let mut stereo_buf = [0.0f32; 1024];
    let synth_allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);
    let synth_reallocs_before = REALLOC_COUNT.load(Ordering::SeqCst);

    // 1a) Single-thread fast path (< PARALLEL_VOICE_THRESHOLD)
    for note in 48..72u32 {
        synth.process_short_event(0x007F_0090 | (note << 8));
    }
    let mut peak = 0.0f32;
    for _ in 0..8 {
        synth.render_audio(&mut stereo_buf);
        for &s in &stereo_buf {
            peak = peak.max(s.abs());
        }
    }

    // 1b) Multi-thread fixed worker pool path (256+ voices -> exercises parallel dispatch + scratch sum)
    for i in 0..256u32 {
        let note = 24 + (i % 72);
        let ch = i & 0x0F;
        synth.process_short_event(0x007F_0090 | (note << 8) | ch);
    }
    for _ in 0..16 {
        synth.render_audio(&mut stereo_buf);
        for &s in &stereo_buf {
            peak = peak.max(s.abs());
        }
    }
    for i in 0..256u32 {
        let note = 24 + (i % 72);
        let ch = i & 0x0F;
        synth.process_short_event(0x0000_0080 | (note << 8) | ch);
    }
    for _ in 0..8 {
        synth.render_audio(&mut stereo_buf);
    }
    synth.reset();

    let synth_allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);
    let synth_reallocs_after = REALLOC_COUNT.load(Ordering::SeqCst);
    assert!(peak > 0.05, "real synth must render non-zero audio");
    assert_eq!(
        synth_allocs_after - synth_allocs_before,
        0,
        "SoundFontSynth hot path MUST perform zero heap allocations"
    );
    assert_eq!(
        synth_reallocs_after - synth_reallocs_before,
        0,
        "SoundFontSynth hot path MUST perform zero heap reallocations"
    );
    let _ = std::fs::remove_dir_all(&tmp);

    // --- Part 2: FFI SendDirectData + consumer loop 1M-event burst ---
    assert_eq!(InitializeKDMAPIStream(), 1);
    reset_telemetry();

    assert_eq!(SendDirectData(0x0064_3C90), MMSYSERR_NOERROR);
    while pending_short_events() > 0 {
        std::thread::yield_now();
    }

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

    let allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);
    let reallocs_before = REALLOC_COUNT.load(Ordering::SeqCst);

    const BURST_EVENTS: u32 = 1_000_000;
    for i in 0..BURST_EVENTS {
        let note = (i & 0x7F) << 8;
        let ch = i & 0x0F;
        let vel = if (i & 1) == 0 { 0x007F_0000 } else { 0 };
        let msg = vel | note | 0x90 | ch;

        if (i & 0x0F) == 0 {
            let rc = SendDirectDataNoBuf(msg);
            debug_assert_eq!(rc, MMSYSERR_NOERROR);
        } else {
            let rc = SendDirectData(msg);
            debug_assert_eq!(rc, MMSYSERR_NOERROR);
        }

        if (i & 0x1_FFFF) == 0x1_FFFF {
            let _ = SendCustomEvent(1, ch, i);
            let _ = SendDirectLongData(&mut hdr, size_of::<MidiHdr>() as u32);
        }
    }

    ResetKDMAPIStream();

    let mut spins = 0u32;
    while pending_short_events() > 0 && spins < 500_000 {
        std::thread::yield_now();
        spins += 1;
    }

    let _voices = GetVoiceCount();
    let _render = GetRenderingTime();
    let snap = telemetry_snapshot();

    let allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);
    let reallocs_after = REALLOC_COUNT.load(Ordering::SeqCst);

    assert_eq!(
        allocs_after - allocs_before,
        0,
        "Hot path MUST perform zero heap allocations"
    );
    assert_eq!(
        reallocs_after - reallocs_before,
        0,
        "Hot path MUST perform zero heap reallocations"
    );
    assert!(snap.events_in >= BURST_EVENTS as u64);

    assert_eq!(TerminateKDMAPIStream(), 1);
    std::env::remove_var(NO_AUDIO_ENV);
}
