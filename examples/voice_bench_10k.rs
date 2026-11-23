//! Release benchmark simulating 10,000 active voices at 48 kHz, `buffer_frames = 512`,
//! comparing Single-Threaded Scalar vs Multi-Threaded AVX2/FMA across `Linear`, `Sinc8`,
//! `Sinc16`, and `Sinc32` interpolation while enforcing strictly ZERO hot-path allocations.
//!
//! Run with:
//!   cargo run --release --example voice_bench_10k

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use OmniMIDI::{InterpolationMode, SoundFontSynth, SynthBackend, SynthConfig};

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

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
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAllocator = CountingAllocator;

const SAMPLE_RATE: u32 = 48_000;
const BUFFER_FRAMES: usize = 512;
const TARGET_VOICES: usize = 10_000;
const WARMUP_BLOCKS: usize = 8;
const MEASURE_BLOCKS: usize = 64;

struct BenchResult {
    avg_ms: f64,
    p95_ms: f64,
    render_pct: f64,
    active_voices: u64,
    hot_allocs: usize,
    peak: f32,
}

fn write_bench_sfz_fixture(dir: &Path) -> PathBuf {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).expect("create bench dir");
    let wav_path = dir.join("piano_tone.wav");
    let sfz_path = dir.join("piano_bench.sfz");

    let sr = 44_100u32;
    let frames = 44_100usize;
    let data_bytes = (frames * 2) as u32;
    let riff_size = 36 + data_bytes;
    let mut f = File::create(&wav_path).expect("create wav");
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
        let t = i as f32 / sr as f32;
        let s = (2.0 * core::f32::consts::PI * 261.63 * t).sin() * 0.7
            + (2.0 * core::f32::consts::PI * 523.25 * t).sin() * 0.3;
        let s16 = (s * 24_000.0) as i16;
        f.write_all(&s16.to_le_bytes()).expect("wav");
    }

    std::fs::write(
        &sfz_path,
        "<region> sample=piano_tone.wav lokey=0 hikey=127 pitch_keycenter=60 loop_mode=loop_continuous loop_start=256 loop_end=40000 ampeg_attack=0.001 ampeg_sustain=100 ampeg_release=0.2\n",
    )
    .expect("write sfz");
    sfz_path
}

fn run_case(
    sf_path: &Path,
    interp: InterpolationMode,
    parallel: bool,
    force_scalar: bool,
    disable_all_effects: bool,
    measure_blocks: usize,
) -> BenchResult {
    let mut synth = SoundFontSynth::new();
    synth
        .init(SynthConfig {
            sample_rate: SAMPLE_RATE,
            max_voices: TARGET_VOICES as u32,
            buffer_frames: BUFFER_FRAMES as u32,
            render_threads: if parallel {
                std::thread::available_parallelism()
                    .map(|n| n.get() as u32)
                    .unwrap_or(4)
            } else {
                1
            },
            interpolation: interp,
            enable_lowpass_filter: true,
            enable_vibrato_lfo: true,
            disable_all_effects,
        })
        .expect("init synth");
    synth
        .load_soundfonts(std::slice::from_ref(&sf_path.to_path_buf()))
        .expect("load soundfont");
    synth.set_parallel_enabled(parallel);
    synth.set_force_scalar(force_scalar);

    // Spawn 10,000 active voices distributed across 16 MIDI channels & 88 piano keys.
    for i in 0..TARGET_VOICES {
        let ch = (i & 0x0F) as u32;
        let note = (21 + (i % 88)) as u32;
        let vel = (64 + (i % 63)) as u32;
        synth.process_short_event((vel << 16) | (note << 8) | (0x90 | ch));
    }

    let active_voices = synth.stats().active_voices;
    let mut stereo_buf = vec![0.0f32; BUFFER_FRAMES * 2];

    for _ in 0..WARMUP_BLOCKS {
        synth.render_audio(&mut stereo_buf);
    }

    let mut block_ms = Vec::with_capacity(measure_blocks);
    let mut peak = 0.0f32;
    let allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);

    for _ in 0..measure_blocks {
        let t0 = Instant::now();
        synth.render_audio(&mut stereo_buf);
        let dt_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        block_ms.push(dt_ms);
        for &s in &stereo_buf {
            peak = peak.max(s.abs());
        }
    }

    let allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);
    let hot_allocs = allocs_after.saturating_sub(allocs_before);

    block_ms.sort_by(|a, b| a.total_cmp(b));
    let avg_ms = block_ms.iter().sum::<f64>() / (block_ms.len() as f64);
    let p95_idx = ((block_ms.len() as f64) * 0.95) as usize;
    let p95_ms = block_ms[p95_idx.min(block_ms.len() - 1)];
    let budget_ms = (BUFFER_FRAMES as f64 / SAMPLE_RATE as f64) * 1_000.0;
    let render_pct = (avg_ms / budget_ms) * 100.0;

    BenchResult {
        avg_ms,
        p95_ms,
        render_pct,
        active_voices,
        hot_allocs,
        peak,
    }
}

fn main() {
    let tmp_dir = std::env::temp_dir().join(format!("kdmapiv3_bench10k_{}", std::process::id()));
    let sfz_path = write_bench_sfz_fixture(&tmp_dir);

    let mut probe = SoundFontSynth::new();
    probe.init(SynthConfig::default()).expect("probe init");
    let budget_ms = (BUFFER_FRAMES as f64 / SAMPLE_RATE as f64) * 1_000.0;

    println!("=== kdmapiv3 10,000-Voice Real-Time DSP Benchmark (SF2 2.01 + Sinc32) ===");
    println!(
        "Audio Config : {} Hz | buffer_frames = {} (deadline = {:.3} ms)",
        SAMPLE_RATE, BUFFER_FRAMES, budget_ms
    );
    println!(
        "Runtime CPU  : {} render threads | AVX2+FMA = {}",
        probe.render_thread_count(),
        probe.simd_avx2_enabled()
    );
    println!();

    let base_sinc8 = run_case(&sfz_path, InterpolationMode::Sinc8, false, true, false, 12);
    let base_sinc32 = run_case(&sfz_path, InterpolationMode::Sinc32, false, true, false, 8);
    let fast_linear = run_case(
        &sfz_path,
        InterpolationMode::Linear,
        true,
        false,
        false,
        MEASURE_BLOCKS,
    );
    let fast_sinc8 = run_case(
        &sfz_path,
        InterpolationMode::Sinc8,
        true,
        false,
        false,
        MEASURE_BLOCKS,
    );
    let fast_sinc16 = run_case(
        &sfz_path,
        InterpolationMode::Sinc16,
        true,
        false,
        false,
        MEASURE_BLOCKS,
    );
    let fast_sinc32 = run_case(
        &sfz_path,
        InterpolationMode::Sinc32,
        true,
        false,
        false,
        MEASURE_BLOCKS,
    );

    println!(
        "{:<28} {:>8} {:>10} {:>10} {:>11} {:>8}",
        "Mode", "Voices", "Avg (ms)", "P95 (ms)", "render_pct", "Allocs"
    );
    println!("{}", "-".repeat(80));
    for (label, r) in [
        ("Sinc8  [1T Scalar baseline]", &base_sinc8),
        ("Sinc32 [1T Scalar baseline]", &base_sinc32),
        ("Linear [MT Branchless]", &fast_linear),
        ("Sinc8  [MT AVX2/FMA]", &fast_sinc8),
        ("Sinc16 [MT AVX2/FMA]", &fast_sinc16),
        ("Sinc32 [MT AVX2/FMA]", &fast_sinc32),
    ] {
        println!(
            "{:<28} {:>8} {:>10.3} {:>10.3} {:>10.2}% {:>8}",
            label, r.active_voices, r.avg_ms, r.p95_ms, r.render_pct, r.hot_allocs
        );
        assert_eq!(r.hot_allocs, 0, "{} performed heap allocations!", label);
        assert!(r.peak > 0.05, "{} produced silence!", label);
    }

    println!();
    println!(
        "Sinc8  Speedup (MT AVX2/FMA vs 1T Scalar): {:.2}x ({:.2}% -> {:.2}%)",
        base_sinc8.avg_ms / fast_sinc8.avg_ms,
        base_sinc8.render_pct,
        fast_sinc8.render_pct
    );
    println!(
        "Sinc32 Speedup (MT AVX2/FMA vs 1T Scalar): {:.2}x ({:.2}% -> {:.2}%)",
        base_sinc32.avg_ms / fast_sinc32.avg_ms,
        base_sinc32.render_pct,
        fast_sinc32.render_pct
    );

    // Optional real-world SoundFont check via CLI arg 1 or `KDMAPIV3_BENCH_SF2` env var.
    let real_sf2_opt = std::env::args_os()
        .nth(1)
        .or_else(|| std::env::var_os("KDMAPIV3_BENCH_SF2"))
        .map(PathBuf::from);
    if let Some(real_sf2) = real_sf2_opt.filter(|p| p.is_file()) {
        let sf_name = real_sf2
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("custom.sf2");
        println!();
        println!("--- Real SoundFont Check ({}) ---", sf_name);
        for (label, mode, no_fx) in [
            ("Custom SF2 Linear [MT + Effects] ", InterpolationMode::Linear, false),
            ("Custom SF2 Sinc8  [MT + Effects] ", InterpolationMode::Sinc8, false),
            ("Custom SF2 Sinc16 [MT + Effects] ", InterpolationMode::Sinc16, false),
            ("Custom SF2 Sinc32 [MT + Effects] ", InterpolationMode::Sinc32, false),
            ("Custom SF2 Sinc8  [Low-Spec Off] ", InterpolationMode::Sinc8, true),
            ("Custom SF2 Sinc32 [Low-Spec Off] ", InterpolationMode::Sinc32, true),
        ] {
            let r = run_case(&real_sf2, mode, true, false, no_fx, MEASURE_BLOCKS);
            println!(
                "{}: voices={} | avg={:.3} ms | p95={:.3} ms | render_pct={:.2}% | allocs={}",
                label, r.active_voices, r.avg_ms, r.p95_ms, r.render_pct, r.hot_allocs
            );
            assert_eq!(r.hot_allocs, 0);
            assert!(r.peak > 0.01, "{} produced silence!", label);
        }
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
