//! Hard real-time, pre-allocated SoundFont (`.sf2` 2.01 / `.sfz`) synthesis engine.
//!
//! # Real-Time Guarantees
//! - All SoundFont parsing, sample decoding + 32-sample boundary padding, 32-byte aligned
//!   Sinc LUT generation, 128-entry SF2 concave LUTs, voice-pool pre-allocation, and fixed
//!   worker-thread pool setup happen strictly on the cold control path (`init` / `load_soundfonts`).
//! - `process_short_event`, `process_sysex`, `process_custom_event`, `reset`, and
//!   `render_audio` perform **ZERO heap allocations**, **ZERO locks**, **ZERO syscalls**,
//!   and **ZERO panics**.
//! - Supports `Linear`, `Sinc8`, `Sinc16`, and `Sinc32` (32-tap Blackman-Harris windowed sinc)
//!   with portable runtime-dispatched explicit AVX2/FMA SIMD kernels and clean scalar fallbacks.
//! - Implements SF2 2.01 acoustic fidelity (concave velocity/CC7/CC11 curves, `initialAttenuation`,
//!   exponential-in-dB Decay/Release, `keynumToVolEnvHold`/`Decay`, zero-overhead bypassable 2-pole
//!   resonant LPF + default velocity-to-filter modulator, linked stereo pairs + constant-power pan,
//!   `sampleModes` 0/1/3, `exclusiveClass` chokes, CC64 Sustain + CC66 Sostenuto, RPN 0 pitch bend
//!   sensitivity + `scaleTuning`, and zero-overhead bypassable Vibrato LFO).

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{fence, AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle, Thread};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, _mm256_add_ps, _mm256_castps128_ps256, _mm256_castps256_ps128,
    _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_hadd_ps, _mm256_insertf128_ps, _mm256_load_ps,
    _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
    _mm_movehl_ps, _mm_mul_ps, _mm_set1_ps, _mm_set_ps, _mm_shuffle_ps, _mm_storeu_ps,
    _mm_unpackhi_ps, _mm_unpacklo_ps,
};

use soundfont::raw::{GeneratorType, SampleChunk, SampleLink};
use soundfont::{SoundFont2, Zone};
use symphonia::core::audio::SampleBuffer as SymphoniaSampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use xsynth_soundfonts::LoopMode;

use crate::backend::{
    BackendError, InterpolationMode, SynthBackend, SynthConfig, SynthStats, MAX_BUFFER_FRAMES,
    MAX_RENDER_THREADS, MAX_VOICE_LIMIT, MIN_BUFFER_FRAMES, MIN_RENDER_THREADS, MIN_VOICE_LIMIT,
};

const SINC_PHASES: usize = 256;
/// Pre-padding on both sides of every decoded sample buffer so Sinc32 (`-15..=+16`)
/// never reads out of bounds in branchless fast-path inner loops.
const SAMPLE_PAD: usize = 32;
/// Audible culling threshold (~ -100 dB / -96 dB). Voices below this are reclaimed immediately.
const MIN_ENV_AMP: f32 = 1.0e-5;
/// Natural log of `1e-5` (-100 dB full-scale attenuation per SF2 2.01 Decay/Release spec).
const LN_ENV_ATTEN: f32 = -11.512_925;
/// Fixed O(1) round-robin candidate probes inside active voices on voice steal.
const STEAL_RR_PROBES: usize = 8;
const VOICE_WORK_CHUNK: usize = 32;
/// Short hot-spin iterations before cooperative yielding to avoid per-callback `futex` thrashing.
const WORKER_HOT_SPIN_ITERS: u32 = 4_096;
/// Cooperative yield iterations before workers park into OS sleep during genuine silence.
const WORKER_YIELD_ITERS: u32 = 64;
/// Bounded RT audio thread spin iterations before yielding when waiting on an in-flight worker.
const RT_WAIT_SPIN_ITERS: u32 = 256;
/// Sample-rate-independent click-free choke (exclusiveClass) and CC120 All Sound Off decay times.
const CHOKE_RELEASE_SECS: f32 = 0.002;
const KILL_RELEASE_SECS: f32 = 0.0025;
const NO_VOICE_LINK: u32 = u32::MAX;
const NO_SAMPLE: u32 = u32::MAX;
const FP_SCALE: f64 = 4_294_967_296.0; // 2^32

/// Compute per-sample exponential decay multiplier reaching `MIN_ENV_AMP` (-100 dB) in `duration_secs`
/// at the given `sample_rate`.
#[inline]
fn exp_decay_mul_for_secs(duration_secs: f32, sample_rate: u32) -> f32 {
    let samples = (duration_secs * (sample_rate.max(8_000) as f32)).max(1.0);
    (LN_ENV_ATTEN / samples).exp().clamp(0.5, 0.999_999)
}

/// Precompute the 128-entry SF2 2.01 concave attenuation gain LUT (`(val / 127)^2`).
///
/// Matches `-20 * log10((val / 127)^2)` dB attenuation converted to linear amplitude
/// (`10^(-dB / 20) = (val / 127)^2`), with `lut[0] == 0.0` and `lut[127] == 1.0`.
pub fn build_sf2_concave_lut() -> [f32; 128] {
    let mut lut = [0.0f32; 128];
    lut[0] = 0.0;
    for v in 1..127 {
        let norm = (v as f64) / 127.0;
        lut[v] = (norm * norm) as f32;
    }
    lut[127] = 1.0;
    lut
}

#[inline]
fn timecents_to_secs(tc: f32) -> f32 {
    if tc <= -11_999.0 {
        0.0
    } else {
        2.0f32.powf(tc.clamp(-12_000.0, 8_000.0) / 1_200.0)
    }
}

#[inline]
fn secs_to_timecents(secs: f32) -> f32 {
    if secs <= 1.0e-5 {
        -12_000.0
    } else {
        1_200.0 * secs.log2()
    }
}

/// 32-byte aligned 256-phase 8-tap Blackman-Harris Sinc lookup table (8 KiB).
#[repr(align(32))]
struct AlignedSinc8 {
    rows: [[f32; 8]; SINC_PHASES],
}

/// 32-byte aligned 256-phase 16-tap Blackman-Harris Sinc lookup table (16 KiB).
#[repr(align(32))]
struct AlignedSinc16 {
    rows: [[f32; 16]; SINC_PHASES],
}

/// 32-byte aligned 256-phase 32-tap Blackman-Harris Sinc lookup table (32 KiB).
#[repr(align(32))]
struct AlignedSinc32 {
    rows: [[f32; 32]; SINC_PHASES],
}

/// Pre-computed 32-byte aligned windowed Sinc lookup tables (computed once during construction).
struct SincTables {
    sinc8: Box<AlignedSinc8>,
    sinc16: Box<AlignedSinc16>,
    sinc32: Box<AlignedSinc32>,
}

impl SincTables {
    fn build() -> Self {
        let mut sinc8 = Box::new(AlignedSinc8 {
            rows: [[0.0f32; 8]; SINC_PHASES],
        });
        let mut sinc16 = Box::new(AlignedSinc16 {
            rows: [[0.0f32; 16]; SINC_PHASES],
        });
        let mut sinc32 = Box::new(AlignedSinc32 {
            rows: [[0.0f32; 32]; SINC_PHASES],
        });

        for phase in 0..SINC_PHASES {
            let frac = phase as f64 / SINC_PHASES as f64;
            fill_sinc_kernel(&mut sinc8.rows[phase], frac);
            fill_sinc_kernel(&mut sinc16.rows[phase], frac);
            fill_sinc_kernel(&mut sinc32.rows[phase], frac);
        }

        Self {
            sinc8,
            sinc16,
            sinc32,
        }
    }
}

/// Compute a normalized Blackman-Harris windowed sinc kernel for fractional offset `frac`.
fn fill_sinc_kernel<const TAPS: usize>(out: &mut [f32; TAPS], frac: f64) {
    let half = (TAPS as f64) * 0.5;
    let center = half - 1.0;
    let mut sum = 0.0f64;
    let mut tmp = [0.0f64; TAPS];

    for (i, slot) in tmp.iter_mut().enumerate() {
        let x = (i as f64 - center) - frac;
        let sinc = if x.abs() < 1.0e-9 {
            1.0
        } else {
            let pix = PI * x;
            pix.sin() / pix
        };

        // Blackman-Harris window over [-half, +half]
        let u = ((x + half) / (TAPS as f64)).clamp(0.0, 1.0);
        let w = 0.35875
            - 0.48829 * (2.0 * PI * u).cos()
            + 0.14128 * (4.0 * PI * u).cos()
            - 0.01168 * (6.0 * PI * u).cos();

        let val = sinc * w;
        *slot = val;
        sum += val;
    }

    let inv_sum = if sum.abs() > 1.0e-12 { 1.0 / sum } else { 1.0 };
    for i in 0..TAPS {
        out[i] = (tmp[i] * inv_sum) as f32;
    }
}

/// Build a pre-padded sample buffer with `SAMPLE_PAD` (32) samples before index 0 and after
/// `raw.len() - 1` so branchless Sinc8/16/32 kernels never read out of bounds.
fn build_padded_samples(raw: &[f32]) -> Box<[f32]> {
    let len = raw.len();
    let mut padded = vec![0.0f32; SAMPLE_PAD + len + SAMPLE_PAD];
    if len > 0 {
        let first = raw[0];
        let last = raw[len - 1];
        padded[..SAMPLE_PAD].fill(first);
        padded[SAMPLE_PAD..SAMPLE_PAD + len].copy_from_slice(raw);
        padded[SAMPLE_PAD + len..].fill(last);
    }
    padded.into_boxed_slice()
}

/// Pre-decoded contiguous mono audio sample buffer owned by [`SoundFontBank`].
///
/// Logical sample `0..sample_len` starts at offset `SAMPLE_PAD` inside `padded`.
/// Stereo pairs reference two `SampleAsset` entries (`sample_idx` and `right_sample_idx`).
struct SampleAsset {
    padded: Box<[f32]>,
    sample_len: u32,
    sample_rate: u32,
}

impl SampleAsset {
    fn from_raw(raw: &[f32], sample_rate: u32) -> Self {
        let sample_len = raw.len().min(u32::MAX as usize) as u32;
        Self {
            padded: build_padded_samples(&raw[..sample_len as usize]),
            sample_len,
            sample_rate: sample_rate.max(1),
        }
    }

    /// Pointer to logical sample index 0 (valid for `-SAMPLE_PAD .. sample_len + SAMPLE_PAD`).
    #[inline]
    fn data_ptr(&self) -> *const f32 {
        // SAFETY: `padded.len() == SAMPLE_PAD + sample_len + SAMPLE_PAD >= SAMPLE_PAD`.
        unsafe { self.padded.as_ptr().add(SAMPLE_PAD) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum RegionLoopMode {
    NoLoop = 0,
    OneShot = 1,
    LoopContinuous = 2,
    LoopSustain = 3,
}

impl From<LoopMode> for RegionLoopMode {
    fn from(mode: LoopMode) -> Self {
        match mode {
            LoopMode::NoLoop => Self::NoLoop,
            LoopMode::OneShot => Self::OneShot,
            LoopMode::LoopContinuous => Self::LoopContinuous,
            LoopMode::LoopSustain => Self::LoopSustain,
        }
    }
}

/// Flat, `Copy` SoundFont key/velocity zone descriptor with full SF2 2.01 generators.
#[derive(Clone, Copy)]
struct SynthRegion {
    sample_idx: u32,
    /// Optional linked right-channel sample index (`NO_SAMPLE` if mono).
    right_sample_idx: u32,
    key_low: u8,
    key_high: u8,
    vel_low: u8,
    vel_high: u8,
    root_key: u8,
    exclusive_class: u8,
    loop_mode: RegionLoopMode,
    loop_start: u32,
    loop_end: u32,
    offset: u32,
    /// Linear gain derived from SF2 `initialAttenuation` (gen 48) or SFZ volume.
    gain: f32,
    /// Pan in `-1.0..=+1.0` derived from SF2 `pan` (gen 17, `-500..+500`) or SFZ pan.
    pan: f32,
    /// Combined coarse + fine + sample pitchadj in cents.
    tune_cents: f32,
    /// SF2 `scaleTuning` (gen 56, cents per key; default 100.0).
    scale_tuning: f32,
    /// Volume envelope Delay / Attack / Hold / Decay / Release in timecents.
    env_delay_tc: f32,
    env_attack_tc: f32,
    env_hold_tc: f32,
    env_decay_tc: f32,
    env_release_tc: f32,
    /// Key-number scaling for Hold & Decay in timecents per key from key 60 (gens 39 & 40).
    key_to_hold_tc: f32,
    key_to_decay_tc: f32,
    /// Linear sustain amplitude (`0.0..=1.0`, from SF2 `sustainVolEnv` cB).
    env_sustain: f32,
    /// SF2 `initialFilterFc` (gen 8, cents, `1500..=13500`; `13500` = wide open).
    filter_fc_cents: f32,
    /// SF2 `initialFilterQ` (gen 9, centibels, `0..=960`).
    filter_q_cb: f32,
    /// Default velocity-to-filter cutoff modulation (`-2400.0` cents for SF2 2.01; `0.0` for SFZ).
    vel_to_filter_cents: f32,
    /// Vibrato LFO delay (seconds), frequency (Hz), and base pitch depth (cents, gen 6).
    vib_delay_secs: f32,
    vib_freq_hz: f32,
    vib_to_pitch_cents: f32,
}

/// Sorted preset index entry mapping `(bank, preset)` -> contiguous range in `regions`.
#[derive(Clone, Copy)]
struct PresetEntry {
    bank: u16,
    preset: u16,
    region_start: u32,
    region_count: u32,
    key_bucket_base: u32,
}

#[derive(Clone, Copy)]
struct RegionPrecomp {
    delay_samples: u32,
    attack_inc: f32,
    base_hold_samples: u32,
    base_decay_mul: f32,
    release_mul: f32,
    has_key_to_hold: bool,
    has_key_to_decay: bool,
    tune_sr_mul: f64,
    filter_inv_2q: f32,
}

const FILTER_TRIG_LUT_STEPS: usize = 512;
const FILTER_TRIG_LUT_LEN: usize = FILTER_TRIG_LUT_STEPS + 1;

fn build_semitone_ratio_table() -> Box<[f64; 255]> {
    let mut arr = Box::new([1.0f64; 255]);
    for i in 0..255 {
        arr[i] = 2.0f64.powf((i as i32 - 127) as f64 / 12.0);
    }
    arr
}

fn build_filter_trig_lut(sr: u32) -> Box<[(f32, f32); FILTER_TRIG_LUT_LEN]> {
    let sr_f64 = sr.max(8_000) as f64;
    let mut arr = Box::new([(1.0f32, 0.0f32); FILTER_TRIG_LUT_LEN]);
    for i in 0..FILTER_TRIG_LUT_LEN {
        let cents = 1_500.0 + (i as f64) * (12_000.0 / FILTER_TRIG_LUT_STEPS as f64);
        let fc = (8.175_798_915_6 * 2.0f64.powf(cents / 1_200.0)).clamp(20.0, 0.45 * sr_f64);
        let w0 = 2.0 * core::f64::consts::PI * fc / sr_f64;
        arr[i] = (w0.cos() as f32, w0.sin() as f32);
    }
    arr
}

fn compute_filter_max_active_cents(sr: u32) -> f32 {
    let sr_f64 = sr.max(8_000) as f64;
    (1_200.0 * ((0.45 * sr_f64) / 8.175_798_915_6).log2()).min(13_499.0) as f32
}

/// Immutable, pre-loaded SoundFont bank ready for lock-free, allocation-free rendering.
pub struct SoundFontBank {
    samples: Box<[SampleAsset]>,
    regions: Box<[SynthRegion]>,
    presets: Box<[PresetEntry]>,
    key_region_ranges: Box<[(u32, u16)]>,
    key_region_indices: Box<[u32]>,
    region_precomp: Box<[RegionPrecomp]>,
    semitone_ratios: Box<[f64; 255]>,
    filter_trig_lut: Box<[(f32, f32); FILTER_TRIG_LUT_LEN]>,
    filter_max_active_cents: f32,
    precomp_sr: u32,
}

impl SoundFontBank {
    pub fn empty() -> Self {
        Self {
            samples: Box::new([]),
            regions: Box::new([]),
            presets: Box::new([]),
            key_region_ranges: Box::new([]),
            key_region_indices: Box::new([]),
            region_precomp: Box::new([]),
            semitone_ratios: build_semitone_ratio_table(),
            filter_trig_lut: build_filter_trig_lut(48_000),
            filter_max_active_cents: compute_filter_max_active_cents(48_000),
            precomp_sr: 48_000,
        }
    }

    fn from_parts(
        samples: Box<[SampleAsset]>,
        regions: Box<[SynthRegion]>,
        mut presets: Box<[PresetEntry]>,
        output_sample_rate: u32,
    ) -> Self {
        let mut key_region_ranges: Vec<(u32, u16)> = Vec::with_capacity(presets.len() * 128);
        let mut key_region_indices: Vec<u32> = Vec::new();
        for p in presets.iter_mut() {
            let start = p.region_start as usize;
            let end = start
                .saturating_add(p.region_count as usize)
                .min(regions.len());
            p.key_bucket_base = key_region_ranges.len() as u32;
            let mut per_key: [Vec<u32>; 128] = core::array::from_fn(|_| Vec::new());
            for idx in start..end {
                let r = &regions[idx];
                let lo = (r.key_low as usize).min(127);
                let hi = (r.key_high as usize).min(127);
                if lo <= hi {
                    for k in lo..=hi {
                        per_key[k].push(idx as u32);
                    }
                }
            }
            for bucket in &per_key {
                let b_start = key_region_indices.len() as u32;
                let b_len = bucket.len().min(u16::MAX as usize) as u16;
                key_region_indices.extend_from_slice(&bucket[..b_len as usize]);
                key_region_ranges.push((b_start, b_len));
            }
        }

        let precomp_sr = output_sample_rate.max(8_000);
        let sr_f64 = precomp_sr as f64;
        let sr_f32 = precomp_sr as f32;
        let mut region_precomp = Vec::with_capacity(regions.len());
        for r in regions.iter() {
            let delay_samples = (timecents_to_secs(r.env_delay_tc) * sr_f32) as u32;
            let attack_samples = (timecents_to_secs(r.env_attack_tc).max(0.001) * sr_f32).max(1.0);
            let base_hold_samples = (timecents_to_secs(r.env_hold_tc) * sr_f32) as u32;
            let base_decay_samples =
                (timecents_to_secs(r.env_decay_tc).max(0.001) * sr_f32).max(1.0);
            let release_samples =
                (timecents_to_secs(r.env_release_tc).max(0.005) * sr_f32).max(16.0);
            let sample_sr = samples
                .get(r.sample_idx as usize)
                .map(|a| a.sample_rate.max(1) as f64)
                .unwrap_or(sr_f64);
            let q_db = (r.filter_q_cb / 10.0).clamp(0.0, 96.0) as f64;
            let q_lin = 10.0f64.powf(q_db / 20.0) * core::f64::consts::FRAC_1_SQRT_2;
            region_precomp.push(RegionPrecomp {
                delay_samples,
                attack_inc: 1.0 / attack_samples,
                base_hold_samples,
                base_decay_mul: (LN_ENV_ATTEN / base_decay_samples)
                    .exp()
                    .clamp(0.5, 0.999_999),
                release_mul: (LN_ENV_ATTEN / release_samples)
                    .exp()
                    .clamp(0.5, 0.999_999),
                has_key_to_hold: r.key_to_hold_tc != 0.0,
                has_key_to_decay: r.key_to_decay_tc != 0.0,
                tune_sr_mul: 2.0f64.powf(r.tune_cents as f64 / 1200.0) * (sample_sr / sr_f64),
                filter_inv_2q: (0.5 / q_lin) as f32,
            });
        }

        Self {
            samples,
            regions,
            presets,
            key_region_ranges: key_region_ranges.into_boxed_slice(),
            key_region_indices: key_region_indices.into_boxed_slice(),
            region_precomp: region_precomp.into_boxed_slice(),
            semitone_ratios: build_semitone_ratio_table(),
            filter_trig_lut: build_filter_trig_lut(precomp_sr),
            filter_max_active_cents: compute_filter_max_active_cents(precomp_sr),
            precomp_sr,
        }
    }

    pub fn preset_count(&self) -> usize {
        self.presets.len()
    }

    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Cold-path loader: parses an ordered list of `.sf2` / `.sfz` paths.
    /// Earlier entries take priority for the same `(bank, preset)` pair.
    pub fn load_ordered(paths: &[PathBuf], output_sample_rate: u32) -> Self {
        let mut samples: Vec<SampleAsset> = Vec::new();
        let mut regions: Vec<SynthRegion> = Vec::new();
        let mut preset_map: HashMap<(u16, u16), Vec<SynthRegion>> = HashMap::new();
        let mut sfz_path_dedup: HashMap<PathBuf, (u32, u32)> = HashMap::new();

        for path in paths {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();

            match ext.as_str() {
                "sf2" => {
                    load_sf2_spec_into(path, &mut samples, &mut preset_map);
                }
                "sfz" => {
                    load_sfz_into(
                        path,
                        output_sample_rate,
                        &mut samples,
                        &mut preset_map,
                        &mut sfz_path_dedup,
                    );
                }
                _ => {}
            }
        }

        let mut keys: Vec<(u16, u16)> = preset_map.keys().copied().collect();
        keys.sort_unstable();

        let mut presets = Vec::with_capacity(keys.len());
        for (bank, preset) in keys {
            if let Some(mut regs) = preset_map.remove(&(bank, preset)) {
                if regs.is_empty() {
                    continue;
                }
                let start = regions.len() as u32;
                let count = regs.len() as u32;
                regions.append(&mut regs);
                presets.push(PresetEntry {
                    bank,
                    preset,
                    region_start: start,
                    region_count: count,
                    key_bucket_base: 0,
                });
            }
        }

        Self::from_parts(
            samples.into_boxed_slice(),
            regions.into_boxed_slice(),
            presets.into_boxed_slice(),
            output_sample_rate,
        )
    }

    /// Allocation-free O(log P) preset lookup with standard GM fallbacks.
    #[inline]
    fn find_preset_entry(&self, bank: u16, preset: u16) -> Option<PresetEntry> {
        if let Some(entry) = self.binary_search_preset(bank, preset) {
            return Some(entry);
        }
        if bank == 128 {
            if let Some(entry) = self.binary_search_preset(128, 0) {
                return Some(entry);
            }
        } else {
            if bank != 0 {
                if let Some(entry) = self.binary_search_preset(0, preset) {
                    return Some(entry);
                }
            }
            if let Some(entry) = self.binary_search_preset(0, 0) {
                return Some(entry);
            }
        }
        self.presets.first().copied()
    }

    #[inline]
    fn binary_search_preset(&self, bank: u16, preset: u16) -> Option<PresetEntry> {
        let target = (bank, preset);
        let idx = self
            .presets
            .binary_search_by_key(&target, |p| (p.bank, p.preset))
            .ok()?;
        self.presets.get(idx).copied()
    }
}

// ============================================================================
// Spec-Accurate SF2 2.01 Loader (Native Sample Rates + Full Generator Merging)
// ============================================================================

#[derive(Clone, Copy)]
struct RawSf2ZoneGens {
    index: Option<u16>,
    key_low: u8,
    key_high: u8,
    vel_low: u8,
    vel_high: u8,
    has_key_range: bool,
    has_vel_range: bool,
    start_offset: i32,
    end_offset: i32,
    loop_start_offset: i32,
    loop_end_offset: i32,
    loop_mode: Option<RegionLoopMode>,
    attenuation_cb: Option<i16>,
    pan: Option<i16>,
    coarse_tune: Option<i16>,
    fine_tune: Option<i16>,
    scale_tuning: Option<i16>,
    root_override: Option<i16>,
    exclusive_class: Option<u8>,
    env_delay_tc: Option<i16>,
    env_attack_tc: Option<i16>,
    env_hold_tc: Option<i16>,
    env_decay_tc: Option<i16>,
    env_sustain_cb: Option<i16>,
    env_release_tc: Option<i16>,
    key_to_hold_tc: Option<i16>,
    key_to_decay_tc: Option<i16>,
    filter_fc_cents: Option<i16>,
    filter_q_cb: Option<i16>,
    vib_delay_tc: Option<i16>,
    vib_freq_cents: Option<i16>,
    vib_to_pitch_cents: Option<i16>,
}

impl Default for RawSf2ZoneGens {
    fn default() -> Self {
        Self {
            index: None,
            key_low: 0,
            key_high: 127,
            vel_low: 0,
            vel_high: 127,
            has_key_range: false,
            has_vel_range: false,
            start_offset: 0,
            end_offset: 0,
            loop_start_offset: 0,
            loop_end_offset: 0,
            loop_mode: None,
            attenuation_cb: None,
            pan: None,
            coarse_tune: None,
            fine_tune: None,
            scale_tuning: None,
            root_override: None,
            exclusive_class: None,
            env_delay_tc: None,
            env_attack_tc: None,
            env_hold_tc: None,
            env_decay_tc: None,
            env_sustain_cb: None,
            env_release_tc: None,
            key_to_hold_tc: None,
            key_to_decay_tc: None,
            filter_fc_cents: None,
            filter_q_cb: None,
            vib_delay_tc: None,
            vib_freq_cents: None,
            vib_to_pitch_cents: None,
        }
    }
}

fn parse_sf2_zones(zones: &[Zone], is_preset: bool) -> Vec<RawSf2ZoneGens> {
    let mut result = Vec::new();
    let mut global = RawSf2ZoneGens::default();

    for (z_idx, zone) in zones.iter().enumerate() {
        let mut g = if z_idx == 0 {
            RawSf2ZoneGens::default()
        } else {
            global
        };
        g.index = None;

        for gen in &zone.gen_list {
            let Ok(ty) = gen.ty.into_result() else {
                continue;
            };
            match ty {
                GeneratorType::StartAddrsOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.start_offset += v as i32;
                    }
                }
                GeneratorType::StartAddrsCoarseOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.start_offset += (v as i32) * 32_768;
                    }
                }
                GeneratorType::EndAddrsOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.end_offset += v as i32;
                    }
                }
                GeneratorType::EndAddrsCoarseOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.end_offset += (v as i32) * 32_768;
                    }
                }
                GeneratorType::StartloopAddrsOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.loop_start_offset += v as i32;
                    }
                }
                GeneratorType::StartloopAddrsCoarseOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.loop_start_offset += (v as i32) * 32_768;
                    }
                }
                GeneratorType::EndloopAddrsOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.loop_end_offset += v as i32;
                    }
                }
                GeneratorType::EndloopAddrsCoarseOffset => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.loop_end_offset += (v as i32) * 32_768;
                    }
                }
                GeneratorType::KeyRange => {
                    if let Some(r) = gen.amount.as_range() {
                        g.key_low = r.low;
                        g.key_high = r.high;
                        g.has_key_range = true;
                    }
                }
                GeneratorType::VelRange => {
                    if let Some(r) = gen.amount.as_range() {
                        g.vel_low = r.low;
                        g.vel_high = r.high;
                        g.has_vel_range = true;
                    }
                }
                GeneratorType::InitialFilterFc => g.filter_fc_cents = gen.amount.as_i16().copied(),
                GeneratorType::InitialFilterQ => g.filter_q_cb = gen.amount.as_i16().copied(),
                GeneratorType::Pan => g.pan = gen.amount.as_i16().copied(),
                GeneratorType::InitialAttenuation => {
                    g.attenuation_cb = gen.amount.as_i16().copied();
                }
                GeneratorType::DelayVolEnv => g.env_delay_tc = gen.amount.as_i16().copied(),
                GeneratorType::AttackVolEnv => g.env_attack_tc = gen.amount.as_i16().copied(),
                GeneratorType::HoldVolEnv => g.env_hold_tc = gen.amount.as_i16().copied(),
                GeneratorType::DecayVolEnv => g.env_decay_tc = gen.amount.as_i16().copied(),
                GeneratorType::SustainVolEnv => g.env_sustain_cb = gen.amount.as_i16().copied(),
                GeneratorType::ReleaseVolEnv => g.env_release_tc = gen.amount.as_i16().copied(),
                GeneratorType::KeynumToVolEnvHold => {
                    g.key_to_hold_tc = gen.amount.as_i16().copied();
                }
                GeneratorType::KeynumToVolEnvDecay => {
                    g.key_to_decay_tc = gen.amount.as_i16().copied();
                }
                GeneratorType::CoarseTune => g.coarse_tune = gen.amount.as_i16().copied(),
                GeneratorType::FineTune => g.fine_tune = gen.amount.as_i16().copied(),
                GeneratorType::ScaleTuning => g.scale_tuning = gen.amount.as_i16().copied(),
                GeneratorType::OverridingRootKey => g.root_override = gen.amount.as_i16().copied(),
                GeneratorType::ExclusiveClass => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.exclusive_class = Some((v.clamp(0, 127)) as u8);
                    }
                }
                GeneratorType::SampleModes => {
                    if let Some(&v) = gen.amount.as_i16() {
                        g.loop_mode = Some(match v & 3 {
                            1 => RegionLoopMode::LoopContinuous,
                            3 => RegionLoopMode::LoopSustain,
                            _ => RegionLoopMode::NoLoop,
                        });
                    }
                }
                GeneratorType::DelayVibLFO => g.vib_delay_tc = gen.amount.as_i16().copied(),
                GeneratorType::FreqVibLFO => g.vib_freq_cents = gen.amount.as_i16().copied(),
                GeneratorType::VibLfoToPitch => g.vib_to_pitch_cents = gen.amount.as_i16().copied(),
                GeneratorType::Instrument if is_preset => {
                    g.index = gen.amount.as_u16().copied();
                }
                GeneratorType::SampleID if !is_preset => {
                    g.index = gen.amount.as_u16().copied();
                }
                _ => {}
            }
        }

        if z_idx == 0 && g.index.is_none() {
            global = g;
        } else if g.index.is_some() {
            result.push(g);
        }
    }

    result
}

fn read_sf2_chunk(file: &mut File, chunk: SampleChunk) -> Option<Vec<u8>> {
    let mut buf = vec![0u8; chunk.len as usize];
    file.seek(SeekFrom::Start(chunk.offset)).ok()?;
    file.read_exact(&mut buf).ok()?;
    Some(buf)
}

#[derive(Clone, Copy)]
struct Sf2PendingRegion {
    region: SynthRegion,
    shdr_idx: u16,
    link_idx: u16,
    sample_type: i8, // -1 = Left, +1 = Right, 0 = Mono
    pan_raw: i16,
}

fn load_sf2_spec_into(
    path: &Path,
    samples: &mut Vec<SampleAsset>,
    preset_map: &mut HashMap<(u16, u16), Vec<SynthRegion>>,
) {
    let Ok(mut file) = File::open(path) else {
        return;
    };
    let Ok(sf2) = SoundFont2::load(&mut file) else {
        return;
    };

    let Some(smpl_chunk) = sf2.sample_data.smpl else {
        return;
    };
    let Some(smpl_bytes) = read_sf2_chunk(&mut file, smpl_chunk) else {
        return;
    };

    // Decode raw 16-bit (or 24-bit via `sm24`) PCM once at native sample rate (zero rubato overhead).
    let sample_count = smpl_bytes.len() / 2;
    let mut pcm = Vec::with_capacity(sample_count);
    let sm24_bytes = sf2
        .sample_data
        .sm24
        .and_then(|c| read_sf2_chunk(&mut file, c))
        .filter(|extra| extra.len() >= sample_count);

    if let Some(extra) = sm24_bytes {
        const INV_I24: f32 = 1.0 / 2_147_483_648.0;
        for i in 0..sample_count {
            let lo = smpl_bytes[i * 2];
            let hi = smpl_bytes[i * 2 + 1];
            let lsb = extra[i];
            let val = i32::from_le_bytes([0, lsb, lo, hi]);
            pcm.push((val as f32) * INV_I24);
        }
    } else {
        const INV_I16: f32 = 1.0 / 32_768.0;
        for chunk in smpl_bytes.chunks_exact(2) {
            let val = i16::from_le_bytes([chunk[0], chunk[1]]);
            pcm.push((val as f32) * INV_I16);
        }
    }

    // Build native-rate pre-padded `SampleAsset` per SF2 sample header (deduped by shdr index).
    let mut shdr_to_asset: Vec<Option<u32>> = vec![None; sf2.sample_headers.len()];
    let pcm_len = pcm.len();

    let parsed_instruments: Vec<Vec<RawSf2ZoneGens>> = sf2
        .instruments
        .iter()
        .map(|inst| parse_sf2_zones(&inst.zones, false))
        .collect();

    for preset in &sf2.presets {
        let key = (preset.header.bank, preset.header.preset);
        if preset_map.contains_key(&key) {
            // Earlier SoundFont in `kdmapiv3.toml` takes priority.
            continue;
        }

        let p_zones = parse_sf2_zones(&preset.zones, true);
        let mut pending: Vec<Sf2PendingRegion> = Vec::new();

        for pz in &p_zones {
            let Some(inst_idx) = pz.index else {
                continue;
            };
            let Some(izones) = parsed_instruments.get(inst_idx as usize) else {
                continue;
            };

            for iz in izones {
                let Some(shdr_idx) = iz.index else {
                    continue;
                };
                let Some(shdr) = sf2.sample_headers.get(shdr_idx as usize) else {
                    continue;
                };

                // Key & Velocity ranges intersect between preset zone and instrument zone.
                let key_low = iz.key_low.max(if pz.has_key_range { pz.key_low } else { 0 });
                let key_high = iz
                    .key_high
                    .min(if pz.has_key_range { pz.key_high } else { 127 });
                let vel_low = iz.vel_low.max(if pz.has_vel_range { pz.vel_low } else { 0 });
                let vel_high = iz
                    .vel_high
                    .min(if pz.has_vel_range { pz.vel_high } else { 127 });
                if key_low > key_high || vel_low > vel_high {
                    continue;
                }

                let raw_start = shdr.start as usize;
                let raw_end = (shdr.end as usize).min(pcm_len);
                if raw_end <= raw_start + 1 {
                    continue;
                }

                let asset_idx = if let Some(idx) = shdr_to_asset[shdr_idx as usize] {
                    idx
                } else {
                    let idx = samples.len() as u32;
                    samples.push(SampleAsset::from_raw(
                        &pcm[raw_start..raw_end],
                        shdr.sample_rate.max(8_000),
                    ));
                    shdr_to_asset[shdr_idx as usize] = Some(idx);
                    idx
                };

                let base_len = (raw_end - raw_start) as i64;
                let eff_end = (base_len + iz.end_offset as i64).clamp(2, base_len) as u32;
                let offset = (iz.start_offset as i64).clamp(0, (eff_end - 1) as i64) as u32;

                let rel_loop_start = (shdr.loop_start as i64 - shdr.start as i64)
                    + iz.loop_start_offset as i64;
                let rel_loop_end =
                    (shdr.loop_end as i64 - shdr.start as i64) + iz.loop_end_offset as i64;
                let loop_start = rel_loop_start.clamp(0, (eff_end - 1) as i64) as u32;
                let loop_end = rel_loop_end.clamp(0, eff_end as i64) as u32;

                let mut loop_mode = iz.loop_mode.unwrap_or(RegionLoopMode::NoLoop);
                if matches!(
                    loop_mode,
                    RegionLoopMode::LoopContinuous | RegionLoopMode::LoopSustain
                ) && loop_end <= loop_start.saturating_add(1)
                {
                    loop_mode = RegionLoopMode::NoLoop;
                }

                // Per SF2 2.01 Section 7.3: preset generators add onto instrument generators.
                let atten_cb = (iz.attenuation_cb.unwrap_or(0) as i32
                    + pz.attenuation_cb.unwrap_or(0) as i32)
                    .clamp(0, 1440);
                let gain = 10.0f32.powf(-(atten_cb as f32) / 200.0);

                let pan_raw =
                    (iz.pan.unwrap_or(0) as i32 + pz.pan.unwrap_or(0) as i32).clamp(-500, 500)
                        as i16;
                let pan = (pan_raw as f32) / 500.0;

                let root_key = match iz.root_override {
                    Some(v) if (0..=127).contains(&v) => v as u8,
                    _ => {
                        if shdr.origpitch <= 127 {
                            shdr.origpitch
                        } else {
                            60
                        }
                    }
                };

                let coarse_tune =
                    iz.coarse_tune.unwrap_or(0) as i32 + pz.coarse_tune.unwrap_or(0) as i32;
                let fine_tune = iz.fine_tune.unwrap_or(0) as i32
                    + pz.fine_tune.unwrap_or(0) as i32
                    + shdr.pitchadj as i32;
                let tune_cents = (coarse_tune as f32) * 100.0 + (fine_tune as f32);

                let scale_tuning = (iz.scale_tuning.unwrap_or(100) as i32
                    + pz.scale_tuning.unwrap_or(0) as i32)
                    .clamp(0, 1200) as f32;

                let env_delay_tc = (iz.env_delay_tc.unwrap_or(-12_000) as f32)
                    + (pz.env_delay_tc.unwrap_or(0) as f32);
                let env_attack_tc = (iz.env_attack_tc.unwrap_or(-12_000) as f32)
                    + (pz.env_attack_tc.unwrap_or(0) as f32);
                let env_hold_tc = (iz.env_hold_tc.unwrap_or(-12_000) as f32)
                    + (pz.env_hold_tc.unwrap_or(0) as f32);
                let env_decay_tc = (iz.env_decay_tc.unwrap_or(-12_000) as f32)
                    + (pz.env_decay_tc.unwrap_or(0) as f32);
                let env_release_tc = (iz.env_release_tc.unwrap_or(-12_000) as f32)
                    + (pz.env_release_tc.unwrap_or(0) as f32);

                let key_to_hold_tc = (iz.key_to_hold_tc.unwrap_or(0) as f32)
                    + (pz.key_to_hold_tc.unwrap_or(0) as f32);
                let key_to_decay_tc = (iz.key_to_decay_tc.unwrap_or(0) as f32)
                    + (pz.key_to_decay_tc.unwrap_or(0) as f32);

                let sustain_cb = (iz.env_sustain_cb.unwrap_or(0) as i32
                    + pz.env_sustain_cb.unwrap_or(0) as i32)
                    .clamp(0, 1440);
                let env_sustain = if sustain_cb >= 1000 {
                    0.0
                } else {
                    10.0f32.powf(-(sustain_cb as f32) / 200.0)
                };

                let filter_fc_cents = (iz.filter_fc_cents.unwrap_or(13_500) as i32
                    + pz.filter_fc_cents.unwrap_or(0) as i32)
                    .clamp(1_500, 13_500) as f32;
                let filter_q_cb = (iz.filter_q_cb.unwrap_or(0) as i32
                    + pz.filter_q_cb.unwrap_or(0) as i32)
                    .clamp(0, 960) as f32;

                let vib_delay_tc = (iz.vib_delay_tc.unwrap_or(-12_000) as f32)
                    + (pz.vib_delay_tc.unwrap_or(0) as f32);
                let vib_freq_cents = (iz.vib_freq_cents.unwrap_or(0) as f32)
                    + (pz.vib_freq_cents.unwrap_or(0) as f32);
                let vib_freq_hz = (8.175_799f32 * 2.0f32.powf(vib_freq_cents / 1_200.0))
                    .clamp(0.1, 100.0);
                let vib_to_pitch_cents = (iz.vib_to_pitch_cents.unwrap_or(0) as f32)
                    + (pz.vib_to_pitch_cents.unwrap_or(0) as f32);

                let sample_type = match shdr.sample_type {
                    SampleLink::LeftSample
                    | SampleLink::RomLeftSample
                    | SampleLink::VorbisLeftSample => -1,
                    SampleLink::RightSample
                    | SampleLink::RomRightSample
                    | SampleLink::VorbisRightSample => 1,
                    _ => 0,
                };

                pending.push(Sf2PendingRegion {
                    region: SynthRegion {
                        sample_idx: asset_idx,
                        right_sample_idx: NO_SAMPLE,
                        key_low,
                        key_high,
                        vel_low,
                        vel_high,
                        root_key,
                        exclusive_class: iz.exclusive_class.unwrap_or(0),
                        loop_mode,
                        loop_start,
                        loop_end,
                        offset,
                        gain,
                        pan,
                        tune_cents,
                        scale_tuning,
                        env_delay_tc,
                        env_attack_tc,
                        env_hold_tc,
                        env_decay_tc,
                        env_release_tc,
                        key_to_hold_tc,
                        key_to_decay_tc,
                        env_sustain,
                        filter_fc_cents,
                        filter_q_cb,
                        vel_to_filter_cents: -2_400.0, // SF2 2.01 default modulator (Section 8.4.2)
                        vib_delay_secs: timecents_to_secs(vib_delay_tc),
                        vib_freq_hz,
                        vib_to_pitch_cents,
                    },
                    shdr_idx,
                    link_idx: shdr.sample_link,
                    sample_type,
                    pan_raw,
                });
            }
        }

        // Pair linked LeftSample / RightSample zones into phase-locked stereo regions
        // while never dropping unpaired or asymmetric zones.
        let mut consumed = vec![false; pending.len()];
        let mut built_regions = Vec::with_capacity(pending.len());

        for i in 0..pending.len() {
            if consumed[i] {
                continue;
            }
            let p = pending[i];
            if p.sample_type != 0 && p.pan_raw.abs() >= 450 {
                let partner = (i + 1..pending.len()).find(|&j| {
                    if consumed[j] {
                        return false;
                    }
                    let q = pending[j];
                    q.sample_type == -p.sample_type
                        && (q.shdr_idx == p.link_idx || p.shdr_idx == q.link_idx)
                        && q.pan_raw == -p.pan_raw
                        && q.region.key_low == p.region.key_low
                        && q.region.key_high == p.region.key_high
                        && q.region.vel_low == p.region.vel_low
                        && q.region.vel_high == p.region.vel_high
                        && q.region.root_key == p.region.root_key
                        && q.region.loop_mode == p.region.loop_mode
                        && q.region.exclusive_class == p.region.exclusive_class
                });

                if let Some(j) = partner {
                    consumed[j] = true;
                    let q = pending[j];
                    let (left_idx, right_idx, base) = if p.sample_type == -1 {
                        (p.region.sample_idx, q.region.sample_idx, p.region)
                    } else {
                        (q.region.sample_idx, p.region.sample_idx, q.region)
                    };
                    let mut merged = base;
                    merged.sample_idx = left_idx;
                    merged.right_sample_idx = right_idx;
                    merged.pan = ((p.pan_raw as f32 + q.pan_raw as f32) * 0.5) / 500.0;
                    built_regions.push(merged);
                    continue;
                }
            }
            built_regions.push(p.region);
        }

        if !built_regions.is_empty() {
            preset_map.insert(key, built_regions);
        }
    }
}

fn load_sfz_into(
    path: &Path,
    output_sample_rate: u32,
    samples: &mut Vec<SampleAsset>,
    preset_map: &mut HashMap<(u16, u16), Vec<SynthRegion>>,
    dedup: &mut HashMap<PathBuf, (u32, u32)>,
) {
    let Ok(sfz_regions) = xsynth_soundfonts::sfz::parse_soundfont(path) else {
        return;
    };

    // SFZ maps into GM bank 0 / preset 0 unless already claimed by a higher-priority SoundFont.
    let key = (0u16, 0u16);
    if preset_map.contains_key(&key) {
        return;
    }

    let mut built_regions = Vec::with_capacity(sfz_regions.len());
    for reg in sfz_regions {
        let (sample_idx, right_sample_idx) = if let Some(&existing) = dedup.get(&reg.sample_path) {
            existing
        } else {
            let Some((left_asset, right_asset)) = decode_audio_sample_file(&reg.sample_path) else {
                continue;
            };
            let l_idx = samples.len() as u32;
            samples.push(left_asset);
            let r_idx = if let Some(ra) = right_asset {
                let idx = samples.len() as u32;
                samples.push(ra);
                idx
            } else {
                NO_SAMPLE
            };
            dedup.insert(reg.sample_path.clone(), (l_idx, r_idx));
            (l_idx, r_idx)
        };

        let sample_len = samples
            .get(sample_idx as usize)
            .map(|s| s.sample_len)
            .unwrap_or(0);
        if sample_len < 2 {
            continue;
        }

        let loop_start = reg.loop_start.min(sample_len.saturating_sub(1));
        let loop_end = if reg.loop_end == 0 {
            sample_len
        } else {
            reg.loop_end.min(sample_len)
        };
        let mut loop_mode = RegionLoopMode::from(reg.loop_mode);
        if matches!(
            loop_mode,
            RegionLoopMode::LoopContinuous | RegionLoopMode::LoopSustain
        ) && loop_end <= loop_start.saturating_add(1)
        {
            loop_mode = RegionLoopMode::NoLoop;
        }

        let env = &reg.ampeg_envelope;
        let gain = 10.0f32.powf(reg.volume as f32 / 20.0).clamp(0.0, 4.0);
        let _ = output_sample_rate;

        built_regions.push(SynthRegion {
            sample_idx,
            right_sample_idx,
            key_low: (*reg.keyrange.start()).clamp(0, 127) as u8,
            key_high: (*reg.keyrange.end()).clamp(0, 127) as u8,
            vel_low: *reg.velrange.start(),
            vel_high: *reg.velrange.end(),
            root_key: reg.pitch_keycenter.clamp(0, 127) as u8,
            exclusive_class: 0,
            loop_mode,
            loop_start,
            loop_end,
            offset: reg.offset.min(sample_len.saturating_sub(1)),
            gain,
            pan: (reg.pan as f32 / 100.0).clamp(-1.0, 1.0),
            tune_cents: reg.tune as f32,
            scale_tuning: 100.0,
            env_delay_tc: secs_to_timecents(env.ampeg_delay.max(0.0)),
            env_attack_tc: secs_to_timecents(env.ampeg_attack.max(0.001)),
            env_hold_tc: secs_to_timecents(env.ampeg_hold.max(0.0)),
            env_decay_tc: secs_to_timecents(env.ampeg_decay.max(0.001)),
            env_release_tc: secs_to_timecents(env.ampeg_release.max(0.005)),
            key_to_hold_tc: 0.0,
            key_to_decay_tc: 0.0,
            env_sustain: (env.ampeg_sustain / 100.0).clamp(0.0, 1.0),
            filter_fc_cents: 13_500.0,
            filter_q_cb: 0.0,
            vel_to_filter_cents: 0.0,
            vib_delay_secs: 0.0,
            vib_freq_hz: 8.176,
            vib_to_pitch_cents: 0.0,
        });
    }

    if !built_regions.is_empty() {
        preset_map.insert(key, built_regions);
    }
}

/// Decode an `.sfz` referenced audio file (`.wav`, `.flac`, `.ogg`) via `symphonia` on the cold path.
fn decode_audio_sample_file(path: &Path) -> Option<(SampleAsset, Option<SampleAsset>)> {
    let file = File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;
    let track = format.default_track()?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut is_stereo = false;

    loop {
        let packet = match format.next_packet() {
            Ok(pkt) => pkt,
            Err(SymphoniaError::IoError(_)) => break,
            Err(SymphoniaError::ResetRequired) => continue,
            Err(_) => break,
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) => break,
        };

        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let frames = decoded.frames();
        let mut sbuf = SymphoniaSampleBuffer::<f32>::new(frames as u64, spec);
        sbuf.copy_interleaved_ref(decoded);
        let interleaved = sbuf.samples();

        if channels >= 2 {
            is_stereo = true;
            for frame in interleaved.chunks_exact(channels) {
                left.push(frame[0]);
                right.push(frame[1]);
            }
        } else {
            left.extend_from_slice(interleaved);
        }
    }

    if left.len() < 2 {
        return None;
    }

    let left_asset = SampleAsset::from_raw(&left, sample_rate);
    let right_asset = if is_stereo && right.len() == left.len() && !right.is_empty() {
        Some(SampleAsset::from_raw(&right, sample_rate))
    } else {
        None
    };
    Some((left_asset, right_asset))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum EnvStage {
    Inactive = 0,
    Delay = 1,
    Attack = 2,
    Hold = 3,
    Decay = 4,
    Sustain = 5,
    Release = 6,
}

/// 2-pole resonant LPF coefficients & Direct Form II Transposed state (3 coeffs, 0 branches).
#[derive(Clone, Copy, Default)]
struct VoiceFilter {
    b0: f32,
    a1: f32,
    a2: f32,
    l_z1: f32,
    l_z2: f32,
    r_z1: f32,
    r_z2: f32,
}

impl VoiceFilter {
    #[inline]
    fn from_fc_q(fc_hz: f64, q_cb: f32, sr: f64) -> Self {
        let fc = fc_hz.clamp(20.0, 0.45 * sr);
        let q_db = (q_cb / 10.0).clamp(0.0, 96.0) as f64;
        let q_lin = 10.0f64.powf(q_db / 20.0) * core::f64::consts::FRAC_1_SQRT_2;
        let w0 = 2.0 * PI * fc / sr;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q_lin);
        let inv_a0 = 1.0 / (1.0 + alpha);
        Self {
            b0: (((1.0 - cos_w0) * 0.5) * inv_a0) as f32,
            a1: ((-2.0 * cos_w0) * inv_a0) as f32,
            a2: ((1.0 - alpha) * inv_a0) as f32,
            l_z1: 0.0,
            l_z2: 0.0,
            r_z1: 0.0,
            r_z2: 0.0,
        }
    }

    #[inline(always)]
    fn from_cos_sin_inv2q(cos_w0: f32, sin_w0: f32, inv_2q: f32) -> Self {
        let alpha = sin_w0 * inv_2q;
        let inv_a0 = 1.0 / (1.0 + alpha);
        Self {
            b0: ((1.0 - cos_w0) * 0.5) * inv_a0,
            a1: (-2.0 * cos_w0) * inv_a0,
            a2: (1.0 - alpha) * inv_a0,
            l_z1: 0.0,
            l_z2: 0.0,
            r_z1: 0.0,
            r_z2: 0.0,
        }
    }

    #[inline(always)]
    fn step_left(&mut self, x: f32) -> f32 {
        let b0x = self.b0 * x;
        let y = b0x + self.l_z1;
        self.l_z1 = (b0x + b0x) - self.a1 * y + self.l_z2;
        self.l_z2 = b0x - self.a2 * y;
        y
    }

    #[inline(always)]
    fn step_right(&mut self, x: f32) -> f32 {
        let b0x = self.b0 * x;
        let y = b0x + self.r_z1;
        self.r_z1 = (b0x + b0x) - self.a1 * y + self.r_z2;
        self.r_z2 = b0x - self.a2 * y;
        y
    }
}

/// Compact hot per-sample voice fields separated from cold metadata to maximize L1/L2 cache density.
#[derive(Clone, Copy)]
struct HotVoice {
    /// Pointer to logical sample index 0 in pre-padded left buffer (`-32..sample_len+32` readable).
    sample_left_ptr: *const f32,
    /// Pointer to logical sample index 0 in pre-padded right buffer, or null if mono.
    sample_right_ptr: *const f32,
    /// 32.32 fixed-point sample read phase.
    pos_fp: u64,
    /// Base sample phase increment per output frame (before pitch bend & vibrato).
    base_step: f64,
    sample_len: u32,
    loop_start: u32,
    loop_end: u32,
    gain_l: f32,
    gain_r: f32,
    env_level: f32,
    attack_inc: f32,
    decay_mul: f32,
    sustain_level: f32,
    release_mul: f32,
    env_samples_left: u32,
    hold_samples: u32,
    /// Bypassable 2-pole resonant LPF (`filter_active == false` skips 100% of filter work).
    filter: VoiceFilter,
    /// Bypassable Vibrato LFO (skipped completely when depth == 0).
    vib_phase: f32,
    vib_inc_per_sample: f32,
    vib_delay_samples: u32,
    vib_to_pitch_cents: f32,
    channel: u8,
    loop_mode: RegionLoopMode,
    env_stage: EnvStage,
    filter_active: bool,
    /// True once this voice has rendered at least one audio block (protects fresh voices from stealing
    /// and prevents same-block Note-On + Note-Off instant death).
    has_rendered: bool,
    /// Deferred Note-Off flag set when Note-Off arrives in the same audio block before first render.
    pending_release: bool,
}

// SAFETY: `HotVoice` raw pointers refer into immutable heap-allocated `SampleAsset` buffers
// owned by `SoundFontSynth::bank`, invalidated via `reset()` before bank replacement.
unsafe impl Send for HotVoice {}
unsafe impl Sync for HotVoice {}

impl Default for HotVoice {
    fn default() -> Self {
        Self {
            sample_left_ptr: core::ptr::null(),
            sample_right_ptr: core::ptr::null(),
            pos_fp: 0,
            base_step: 1.0,
            sample_len: 0,
            loop_start: 0,
            loop_end: 0,
            gain_l: 0.0,
            gain_r: 0.0,
            env_level: 0.0,
            attack_inc: 1.0,
            decay_mul: 0.999,
            sustain_level: 1.0,
            release_mul: 0.995,
            env_samples_left: 0,
            hold_samples: 0,
            filter: VoiceFilter::default(),
            vib_phase: 0.0,
            vib_inc_per_sample: 0.0,
            vib_delay_samples: 0,
            vib_to_pitch_cents: 0.0,
            channel: 0,
            loop_mode: RegionLoopMode::NoLoop,
            env_stage: EnvStage::Inactive,
            filter_active: false,
            has_rendered: false,
            pending_release: false,
        }
    }
}

/// Cold voice metadata touched only on MIDI control events (`note_on`, `note_off`, voice stealing,
/// `exclusiveClass` chokes, pedals, `reset`), never read by worker threads during sample rendering.
#[derive(Clone, Copy)]
struct ColdVoice {
    active: bool,
    channel: u8,
    key: u8,
    velocity: u8,
    exclusive_class: u8,
    sustained: bool,
    sostenuto: bool,
    key_released: bool,
    in_key_list: bool,
    age: u64,
    prev_on_key: u32,
    next_on_key: u32,
}

impl Default for ColdVoice {
    fn default() -> Self {
        Self {
            active: false,
            channel: 0,
            key: 0,
            velocity: 0,
            exclusive_class: 0,
            sustained: false,
            sostenuto: false,
            key_released: false,
            in_key_list: false,
            age: 0,
            prev_on_key: NO_VOICE_LINK,
            next_on_key: NO_VOICE_LINK,
        }
    }
}

#[derive(Clone, Copy)]
struct ChannelState {
    bank_msb: u8,
    bank_lsb: u8,
    program: u8,
    /// Concave linear gain from MIDI CC7 (Channel Volume).
    volume: f32,
    /// Concave linear gain from MIDI CC11 (Expression).
    expression: f32,
    /// MIDI CC10 Pan (`-1.0..=+1.0`).
    pan: f32,
    /// MIDI CC1 Modulation Wheel (`0.0..=1.0`).
    mod_wheel: f32,
    /// MIDI CC64 Sustain / Damper pedal.
    sustain: bool,
    /// MIDI CC66 Sostenuto pedal.
    sostenuto: bool,
    pitch_bend_raw: f32,         // -1.0 ..= +1.0
    pitch_bend_range_semis: f32, // semitones (RPN 0 MSB, default 2.0)
    pitch_bend_range_cents: f32, // fractional cents (RPN 0 LSB, default 0.0)
    pitch_bend_ratio: f64,       // precomputed 2^(semitones / 12)
    rpn_msb: u8,
    rpn_lsb: u8,
}

impl ChannelState {
    fn new(is_percussion: bool, concave_lut: &[f32; 128]) -> Self {
        Self {
            bank_msb: if is_percussion { 128 } else { 0 },
            bank_lsb: 0,
            program: 0,
            volume: concave_lut[100],
            expression: concave_lut[127],
            pan: 0.0,
            mod_wheel: 0.0,
            sustain: false,
            sostenuto: false,
            pitch_bend_raw: 0.0,
            pitch_bend_range_semis: 2.0,
            pitch_bend_range_cents: 0.0,
            pitch_bend_ratio: 1.0,
            rpn_msb: 127,
            rpn_lsb: 127,
        }
    }

    #[inline]
    fn effective_bank(&self, channel_idx: usize) -> u16 {
        if channel_idx == 9 {
            128
        } else {
            self.bank_msb as u16
        }
    }

    #[inline]
    fn update_pitch_ratio(&mut self) {
        let range = self.pitch_bend_range_semis + self.pitch_bend_range_cents * 0.01;
        let semis = (self.pitch_bend_raw * range) as f64;
        self.pitch_bend_ratio = 2.0f64.powf(semis / 12.0);
    }
}

/// Cache-line padded atomic flag to prevent false sharing between workers.
#[repr(align(64))]
struct PaddedAtomicBool(AtomicBool);

/// Cache-line padded pre-allocated stereo scratch buffer owned by a single worker.
#[repr(align(64))]
struct WorkerScratch {
    buf: UnsafeCell<Box<[f32]>>,
}

#[repr(align(64))]
struct PaddedAtomicU64(AtomicU64);

#[repr(align(64))]
struct PaddedAtomicUsize(AtomicUsize);

/// Synchronous per-callback render job descriptor shared across fixed pool workers.
#[derive(Clone, Copy)]
struct RenderJob {
    hot_voices_ptr: *mut HotVoice,
    voices_len: usize,
    active_list_ptr: *const u32,
    active_len: usize,
    alive_flags_ptr: *mut u8,
    pitch_ratios: [f64; 16],
    mod_wheels: [f32; 16],
    sinc_tables_ptr: *const SincTables,
    interp: InterpolationMode,
    use_avx2_fma: bool,
    enable_vibrato_lfo: bool,
    frames: usize,
    work_chunk: usize,
}

impl Default for RenderJob {
    fn default() -> Self {
        Self {
            hot_voices_ptr: core::ptr::null_mut(),
            voices_len: 0,
            active_list_ptr: core::ptr::null(),
            active_len: 0,
            alive_flags_ptr: core::ptr::null_mut(),
            pitch_ratios: [1.0; 16],
            mod_wheels: [0.0; 16],
            sinc_tables_ptr: core::ptr::null(),
            interp: InterpolationMode::Linear,
            use_avx2_fma: false,
            enable_vibrato_lfo: true,
            frames: 0,
            work_chunk: VOICE_WORK_CHUNK,
        }
    }
}

struct SharedRenderState {
    /// Encodes `(epoch << 1) | open_bit`.
    /// - `open_bit == 1`: epoch is open for workers to join.
    /// - `open_bit == 0`: epoch is closed; sleeping/late workers never touch `job` or block the RT thread.
    job_state: PaddedAtomicU64,
    next_idx: PaddedAtomicUsize,
    /// Count of workers currently inside the active epoch's critical render window.
    in_flight: PaddedAtomicUsize,
    /// Number of workers currently parked in OS sleep (zero-syscall fast-check when 0).
    parked_count: PaddedAtomicUsize,
    ready_workers: PaddedAtomicUsize,
    shutdown: AtomicBool,
    worker_parked: Box<[PaddedAtomicBool]>,
    worker_did_work: Box<[PaddedAtomicBool]>,
    scratches: Box<[WorkerScratch]>,
    job: UnsafeCell<RenderJob>,
}

// SAFETY: Access to `job`, `worker_did_work[w]`, and `scratches[w]` is strictly synchronized via
// the Dekker SeqCst fence pair on `job_state` (open/close) and `in_flight` (Acquire/Release).
// Active voice indices are pairwise unique and dynamically claimed in disjoint chunks via `next_idx.fetch_add`.
unsafe impl Send for SharedRenderState {}
unsafe impl Sync for SharedRenderState {}

/// Fixed-size, zero-allocation, lock-free audio worker pool initialized strictly on the cold path.
struct RenderPool {
    shared: Arc<SharedRenderState>,
    handles: Box<[Option<JoinHandle<()>>]>,
    threads: Box<[Thread]>,
    worker_count: usize,
    max_frames: usize,
    epoch: u64,
}

impl RenderPool {
    fn new(worker_count: usize, max_frames: usize) -> Option<Self> {
        if worker_count == 0 || max_frames == 0 {
            return None;
        }

        let worker_parked: Box<[PaddedAtomicBool]> = (0..worker_count)
            .map(|_| PaddedAtomicBool(AtomicBool::new(false)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let worker_did_work: Box<[PaddedAtomicBool]> = (0..worker_count)
            .map(|_| PaddedAtomicBool(AtomicBool::new(false)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let scratches: Box<[WorkerScratch]> = (0..worker_count)
            .map(|_| WorkerScratch {
                buf: UnsafeCell::new(vec![0.0f32; max_frames * 2].into_boxed_slice()),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let shared = Arc::new(SharedRenderState {
            job_state: PaddedAtomicU64(AtomicU64::new(0)),
            next_idx: PaddedAtomicUsize(AtomicUsize::new(0)),
            in_flight: PaddedAtomicUsize(AtomicUsize::new(0)),
            parked_count: PaddedAtomicUsize(AtomicUsize::new(0)),
            ready_workers: PaddedAtomicUsize(AtomicUsize::new(0)),
            shutdown: AtomicBool::new(false),
            worker_parked,
            worker_did_work,
            scratches,
            job: UnsafeCell::new(RenderJob::default()),
        });

        let mut handles = Vec::with_capacity(worker_count);
        let mut threads = Vec::with_capacity(worker_count);

        for w in 0..worker_count {
            let worker_shared = Arc::clone(&shared);
            let spawned = thread::Builder::new()
                .name(format!("kdmapiv3-dsp-{}", w))
                .spawn(move || worker_loop(w, worker_shared));

            match spawned {
                Ok(handle) => {
                    threads.push(handle.thread().clone());
                    handles.push(Some(handle));
                }
                Err(_) => {
                    shared.shutdown.store(true, Ordering::Release);
                    shared.job_state.0.store(2, Ordering::Release);
                    fence(Ordering::SeqCst);
                    for t in &threads {
                        t.unpark();
                    }
                    for h in handles.into_iter().flatten() {
                        let _ = h.join();
                    }
                    return None;
                }
            }
        }

        while shared.ready_workers.0.load(Ordering::Acquire) < worker_count {
            thread::yield_now();
        }

        Some(Self {
            shared,
            handles: handles.into_boxed_slice(),
            threads: threads.into_boxed_slice(),
            worker_count,
            max_frames,
            epoch: 0,
        })
    }

    #[inline]
    fn render_parallel(&mut self, mut job: RenderJob, output: &mut [f32]) {
        let total_threads = self.worker_count + 1;
        job.work_chunk = (job.active_len / total_threads).clamp(1, VOICE_WORK_CHUNK);

        let next_epoch = self.epoch.wrapping_add(1).max(1);
        self.epoch = next_epoch;

        // SAFETY: Previous epoch was closed and `in_flight` reached 0 before returning,
        // so no worker is reading `job` or writing `worker_did_work` / `scratches`.
        unsafe {
            *self.shared.job.get() = job;
        }
        for w in 0..self.worker_count {
            self.shared.worker_did_work[w]
                .0
                .store(false, Ordering::Relaxed);
        }
        self.shared.next_idx.0.store(0, Ordering::Relaxed);

        let open_state = (next_epoch << 1) | 1;
        self.shared.job_state.0.store(open_state, Ordering::Release);
        fence(Ordering::SeqCst);

        // Zero-syscall fast path when workers are already warm (`parked_count == 0`) or
        // when all voices fit in a single chunk handled directly by the audio thread.
        if job.active_len > job.work_chunk && self.shared.parked_count.0.load(Ordering::Acquire) > 0
        {
            for w in 0..self.worker_count {
                if self.shared.worker_parked[w].0.swap(false, Ordering::AcqRel) {
                    self.shared.parked_count.0.fetch_sub(1, Ordering::AcqRel);
                    self.threads[w].unpark();
                }
            }
        }

        execute_render_chunks(&self.shared, &job, output, false);

        // Close this epoch immediately once all voice chunks are claimed so sleeping or
        // OS-preempted workers that never joined cannot delay the RT audio thread.
        self.shared
            .job_state
            .0
            .store(next_epoch << 1, Ordering::Release);
        fence(Ordering::SeqCst);

        // Wait ONLY for workers that already entered the epoch; yield after bounded spins
        // so a preempted worker on an oversubscribed CPU is never starved by RT busy-waiting.
        let mut wait_spins = 0u32;
        while self.shared.in_flight.0.load(Ordering::Acquire) > 0 {
            if wait_spins < RT_WAIT_SPIN_ITERS {
                core::hint::spin_loop();
                wait_spins += 1;
            } else {
                thread::yield_now();
            }
        }

        let sample_count = (job.frames * 2).min(output.len());
        let dst = &mut output[..sample_count];
        for w in 0..self.worker_count {
            if self.shared.worker_did_work[w].0.load(Ordering::Relaxed) {
                // SAFETY: Worker `w` completed and published `Release` on `in_flight`, observed with `Acquire` above.
                let scratch = unsafe { &*self.shared.scratches[w].buf.get() };
                let src = &scratch[..sample_count];
                sum_f32_slices(dst, src, job.use_avx2_fma);
            }
        }
    }
}

impl Drop for RenderPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        let next_epoch = self.epoch.wrapping_add(1).max(1);
        self.epoch = next_epoch;
        self.shared
            .job_state
            .0
            .store(next_epoch << 1, Ordering::Release);
        fence(Ordering::SeqCst);
        for t in self.threads.iter() {
            t.unpark();
        }
        for h in self.handles.iter_mut() {
            if let Some(handle) = h.take() {
                let _ = handle.join();
            }
        }
    }
}

#[inline(always)]
pub(crate) fn enable_ftz_daz() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        const FTZ_DAZ_MASK: u32 = 0x8040;
        let mut csr: u32 = 0;
        core::arch::asm!(
            "stmxcsr dword ptr [{}]",
            in(reg) &mut csr as *mut u32,
            options(nostack, preserves_flags),
        );
        if (csr & FTZ_DAZ_MASK) != FTZ_DAZ_MASK {
            let new_csr = csr | FTZ_DAZ_MASK;
            core::arch::asm!(
                "ldmxcsr dword ptr [{}]",
                in(reg) &new_csr as *const u32,
                options(nostack, preserves_flags),
            );
        }
    }
}

fn worker_loop(worker_id: usize, shared: Arc<SharedRenderState>) {
    let _ = crate::backend::platform::configure_realtime_thread();
    enable_ftz_daz();
    shared.ready_workers.0.fetch_add(1, Ordering::Release);

    let mut last_epoch = 0u64;
    loop {
        let mut state = shared.job_state.0.load(Ordering::Acquire);
        let mut spins = 0u32;
        let mut yields = 0u32;

        while (state & 1) == 0 || (state >> 1) == last_epoch {
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            if spins < WORKER_HOT_SPIN_ITERS {
                core::hint::spin_loop();
                spins += 1;
                state = shared.job_state.0.load(Ordering::Acquire);
            } else if yields < WORKER_YIELD_ITERS {
                thread::yield_now();
                yields += 1;
                state = shared.job_state.0.load(Ordering::Acquire);
            } else {
                // Register parked state: increment `parked_count` BEFORE setting `worker_parked[id]`
                // so the audio thread can never underflow `parked_count`.
                shared.parked_count.0.fetch_add(1, Ordering::AcqRel);
                shared.worker_parked[worker_id]
                    .0
                    .store(true, Ordering::Release);
                fence(Ordering::SeqCst);

                state = shared.job_state.0.load(Ordering::Acquire);
                if shared.shutdown.load(Ordering::Acquire)
                    || ((state & 1) != 0 && (state >> 1) != last_epoch)
                {
                    if shared.worker_parked[worker_id]
                        .0
                        .swap(false, Ordering::AcqRel)
                    {
                        shared.parked_count.0.fetch_sub(1, Ordering::AcqRel);
                    }
                    break;
                }

                thread::park();

                if shared.worker_parked[worker_id]
                    .0
                    .swap(false, Ordering::AcqRel)
                {
                    shared.parked_count.0.fetch_sub(1, Ordering::AcqRel);
                }
                spins = 0;
                yields = 0;
                state = shared.job_state.0.load(Ordering::Acquire);
            }
        }

        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        let epoch = state >> 1;
        last_epoch = epoch;

        // Dekker SeqCst registration: claim `in_flight` BEFORE re-verifying `job_state == state`.
        // If the epoch closed concurrently, exit immediately without touching `job` or `scratches`.
        shared.in_flight.0.fetch_add(1, Ordering::AcqRel);
        fence(Ordering::SeqCst);

        if shared.job_state.0.load(Ordering::Acquire) == state {
            // SAFETY: `job` was published before `job_state` Release store, and `in_flight > 0`
            // guarantees the audio thread cannot return until our `fetch_sub(1, Release)` below.
            let job = unsafe { *shared.job.get() };
            // SAFETY: Worker `worker_id` has exclusive access to its own scratch buffer.
            let scratch = unsafe { &mut *shared.scratches[worker_id].buf.get() };
            let sample_count = (job.frames * 2).min(scratch.len());
            let out_slice = &mut scratch[..sample_count];

            let did_work = execute_render_chunks(&shared, &job, out_slice, true);
            if did_work {
                shared.worker_did_work[worker_id]
                    .0
                    .store(true, Ordering::Relaxed);
            }
        }

        shared.in_flight.0.fetch_sub(1, Ordering::Release);
    }
}

/// Claim disjoint voice chunks lock-free and render them into `out_slice`.
/// When `lazy_zero_scratch` is `true`, zeroes `out_slice` only if at least one chunk is claimed.
/// Returns `true` if this thread rendered at least one voice.
#[inline]
fn execute_render_chunks(
    shared: &SharedRenderState,
    job: &RenderJob,
    out_slice: &mut [f32],
    lazy_zero_scratch: bool,
) -> bool {
    let active_len = job.active_len;
    let frames = job.frames;
    let interp = job.interp;
    let use_avx2_fma = job.use_avx2_fma;
    let enable_vibrato_lfo = job.enable_vibrato_lfo;
    let work_chunk = job.work_chunk.max(1);

    // SAFETY: `active_list_ptr` and `sinc_tables_ptr` are valid immutable refs for the job duration.
    let active_list = unsafe { core::slice::from_raw_parts(job.active_list_ptr, active_len) };
    let sinc_tables = unsafe { &*job.sinc_tables_ptr };
    let mut rendered_any = false;

    loop {
        let start = shared.next_idx.0.fetch_add(work_chunk, Ordering::Relaxed);
        if start >= active_len {
            break;
        }
        if lazy_zero_scratch && !rendered_any {
            out_slice.fill(0.0);
        }
        rendered_any = true;
        let end = (start + work_chunk).min(active_len);

        for active_idx in start..end {
            let slot = active_list[active_idx] as usize;
            let alive = if slot < job.voices_len {
                // SAFETY: Every `active_list[active_idx]` is unique across `0..active_len`,
                // and `[start..end)` is exclusively claimed by this thread.
                let voice = unsafe { &mut *job.hot_voices_ptr.add(slot) };
                let ch = (voice.channel as usize).min(15);
                let pitch_ratio = job.pitch_ratios[ch];
                let mod_wheel = job.mod_wheels[ch];
                render_hot_voice(
                    voice,
                    pitch_ratio,
                    mod_wheel,
                    enable_vibrato_lfo,
                    interp,
                    use_avx2_fma,
                    sinc_tables,
                    out_slice,
                    frames,
                )
            } else {
                false
            };

            // SAFETY: `active_idx < active_len` is exclusively owned by this thread.
            unsafe {
                *job.alive_flags_ptr.add(active_idx) = if alive { 1 } else { 0 };
            }
        }
    }

    rendered_any
}

/// Hard real-time, pre-allocated SoundFont synthesizer implementing [`SynthBackend`].
pub struct SoundFontSynth {
    config: SynthConfig,
    bank: SoundFontBank,
    sinc_tables: SincTables,
    /// Precomputed 128-entry SF2 concave attenuation LUT (Velocity, CC7, CC11).
    concave_lut: [f32; 128],
    channels: [ChannelState; 16],
    /// Hot voice DSP state (cache-dense, accessed inside worker render loops).
    hot_voices: Box<[HotVoice]>,
    /// Cold voice metadata (accessed only on Note On/Off, stealing, chokes, pedals, reset).
    cold_voices: Box<[ColdVoice]>,
    active_list: Box<[u32]>,
    alive_flags: Box<[u8]>,
    active_len: usize,
    /// Prefix count of already-rendered (`has_rendered == true`) voices in `active_list[0..rendered_count]`.
    /// Suffix `active_list[rendered_count..active_len]` holds newly spawned unrendered voices in this block.
    rendered_count: usize,
    free_stack: Box<[u32]>,
    free_len: usize,
    /// Intrusive doubly-linked list heads indexed by `(channel << 7) | key` for O(k) Note Off.
    key_heads: [u32; 16 * 128],
    age_counter: u64,
    steal_cursor: usize,
    master_gain: f32,
    /// Precomputed sample-rate-based decay multipliers for `exclusiveClass` chokes & CC120 All Sound Off.
    choke_release_mul: f32,
    kill_release_mul: f32,
    /// Cached once at `init()` (zero config reads / locks on hot path).
    enable_lowpass_filter: bool,
    enable_vibrato_lfo: bool,
    use_avx2_fma: bool,
    force_scalar: bool,
    parallel_enabled: bool,
    render_pool: Option<RenderPool>,
}

#[inline]
fn detect_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

impl SoundFontSynth {
    pub fn new() -> Self {
        let default_cfg = SynthConfig::default();
        let concave_lut = build_sf2_concave_lut();
        let mut synth = Self {
            config: default_cfg,
            bank: SoundFontBank::empty(),
            sinc_tables: SincTables::build(),
            concave_lut,
            channels: default_channels(&concave_lut),
            hot_voices: Box::new([]),
            cold_voices: Box::new([]),
            active_list: Box::new([]),
            alive_flags: Box::new([]),
            active_len: 0,
            rendered_count: 0,
            free_stack: Box::new([]),
            free_len: 0,
            key_heads: [NO_VOICE_LINK; 16 * 128],
            age_counter: 0,
            steal_cursor: 0,
            master_gain: 0.35,
            choke_release_mul: exp_decay_mul_for_secs(CHOKE_RELEASE_SECS, default_cfg.sample_rate),
            kill_release_mul: exp_decay_mul_for_secs(KILL_RELEASE_SECS, default_cfg.sample_rate),
            enable_lowpass_filter: default_cfg.lowpass_filter_active(),
            enable_vibrato_lfo: default_cfg.vibrato_lfo_active(),
            use_avx2_fma: detect_avx2_fma(),
            force_scalar: false,
            parallel_enabled: true,
            render_pool: None,
        };
        synth.allocate_pool(default_cfg.max_voices);
        synth
    }

    fn allocate_pool(&mut self, max_voices: u32) {
        let cap = (max_voices as usize).clamp(MIN_VOICE_LIMIT as usize, MAX_VOICE_LIMIT as usize);
        self.hot_voices = vec![HotVoice::default(); cap].into_boxed_slice();
        self.cold_voices = vec![ColdVoice::default(); cap].into_boxed_slice();
        self.active_list = vec![0u32; cap].into_boxed_slice();
        self.alive_flags = vec![0u8; cap].into_boxed_slice();
        let mut free = Vec::with_capacity(cap);
        for i in (0..cap).rev() {
            free.push(i as u32);
        }
        self.free_stack = free.into_boxed_slice();
        self.free_len = cap;
        self.active_len = 0;
        self.rendered_count = 0;
        self.key_heads.fill(NO_VOICE_LINK);
        self.steal_cursor = 0;
    }

    fn ensure_render_pool(&mut self, render_threads: u32, buffer_frames: u32) {
        let total_threads =
            render_threads.clamp(MIN_RENDER_THREADS, MAX_RENDER_THREADS) as usize;
        let worker_count = total_threads.saturating_sub(1);
        let needed_frames =
            (buffer_frames as usize).clamp(MIN_BUFFER_FRAMES as usize, MAX_BUFFER_FRAMES as usize);

        if worker_count == 0 {
            self.render_pool = None;
            return;
        }

        if let Some(ref pool) = self.render_pool {
            if pool.worker_count == worker_count && pool.max_frames >= needed_frames {
                return;
            }
        }
        self.render_pool = RenderPool::new(worker_count, needed_frames);
    }

    pub fn loaded_preset_count(&self) -> usize {
        self.bank.preset_count()
    }

    pub fn simd_avx2_enabled(&self) -> bool {
        self.use_avx2_fma && !self.force_scalar
    }

    pub fn render_thread_count(&self) -> usize {
        if !self.parallel_enabled {
            return 1;
        }
        self.render_pool
            .as_ref()
            .map(|p| p.worker_count + 1)
            .unwrap_or(1)
    }

    pub fn set_force_scalar(&mut self, scalar: bool) {
        self.force_scalar = scalar;
    }

    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.parallel_enabled = enabled;
    }

    #[inline]
    fn link_voice_key(&mut self, slot: usize, channel: usize, key: u8) {
        if channel >= 16 || slot >= self.cold_voices.len() {
            return;
        }
        if self.cold_voices[slot].in_key_list {
            self.unlink_voice_key(slot);
        }
        let head_idx = (channel << 7) | ((key as usize) & 0x7F);
        let old_head = self.key_heads[head_idx];
        self.cold_voices[slot].prev_on_key = NO_VOICE_LINK;
        self.cold_voices[slot].next_on_key = old_head;
        self.cold_voices[slot].in_key_list = true;
        if old_head != NO_VOICE_LINK {
            if let Some(next_v) = self.cold_voices.get_mut(old_head as usize) {
                next_v.prev_on_key = slot as u32;
            }
        }
        self.key_heads[head_idx] = slot as u32;
    }

    #[inline]
    fn unlink_voice_key(&mut self, slot: usize) {
        let Some(v) = self.cold_voices.get(slot) else {
            return;
        };
        if !v.in_key_list {
            return;
        }
        let ch = v.channel as usize;
        let key = (v.key as usize) & 0x7F;
        let prev = v.prev_on_key;
        let next = v.next_on_key;

        if prev != NO_VOICE_LINK {
            if let Some(prev_v) = self.cold_voices.get_mut(prev as usize) {
                prev_v.next_on_key = next;
            }
        } else if ch < 16 {
            self.key_heads[(ch << 7) | key] = next;
        }

        if next != NO_VOICE_LINK {
            if let Some(next_v) = self.cold_voices.get_mut(next as usize) {
                next_v.prev_on_key = prev;
            }
        }

        if let Some(v) = self.cold_voices.get_mut(slot) {
            v.prev_on_key = NO_VOICE_LINK;
            v.next_on_key = NO_VOICE_LINK;
            v.in_key_list = false;
        }
    }

    /// Force-choke active voices on the same channel sharing `exclusive_class` (gen 57).
    #[inline]
    fn choke_exclusive_class(&mut self, channel: usize, excl_class: u8, note_age_cutoff: u64) {
        if excl_class == 0 || channel >= 16 {
            return;
        }
        let ch_u8 = channel as u8;
        for i in 0..self.active_len {
            let slot = match self.active_list.get(i) {
                Some(&s) => s as usize,
                None => continue,
            };
            if let Some(cv) = self.cold_voices.get_mut(slot) {
                if cv.active
                    && cv.channel == ch_u8
                    && cv.exclusive_class == excl_class
                    && cv.age <= note_age_cutoff
                {
                    cv.sustained = false;
                    cv.sostenuto = false;
                    cv.key_released = true;
                    if let Some(hv) = self.hot_voices.get_mut(slot) {
                        // Sample-rate-scaled click-free choke release (exponential in dB).
                        hv.release_mul = self.choke_release_mul;
                        hv.env_stage = EnvStage::Release;
                    }
                }
            }
        }
    }

    /// Hot path: strictly O(1) voice slot acquisition via Free-list or consistent newest-note-priority
    /// voice stealing (Release/pending_release -> lowest env_level -> lowest velocity -> oldest age).
    ///
    /// Prefers already-rendered voices (`active_list[0..rendered_count]`) when available; when all
    /// slots were spawned in the current block (`rendered_count == 0`), steals from `active_list[0..active_len]`
    /// using the exact same policy instead of dropping incoming Note-Ons.
    #[inline]
    fn acquire_voice_slot(&mut self) -> Option<usize> {
        if self.free_len > 0 {
            self.free_len -= 1;
            let slot = *self.free_stack.get(self.free_len)? as usize;
            if let Some(dst) = self.active_list.get_mut(self.active_len) {
                *dst = slot as u32;
                self.active_len += 1;
            }
            return Some(slot);
        }

        if self.active_len == 0 {
            return None;
        }

        let stealing_from_rendered = self.rendered_count > 0;
        let search_len = if stealing_from_rendered {
            self.rendered_count
        } else {
            self.active_len
        };

        let probes = search_len.min(STEAL_RR_PROBES);
        let base = self.steal_cursor % search_len;
        let mut best_active_idx = base;
        let mut best_slot = self.active_list[base] as usize;
        let mut best_score = f32::INFINITY;
        let mut best_age = u64::MAX;

        for step in 0..probes {
            let active_idx = (base + step) % search_len;
            let slot = self.active_list[active_idx] as usize;
            let hv = &self.hot_voices[slot];
            let cv = &self.cold_voices[slot];
            let stage_bias = if hv.env_stage == EnvStage::Release || hv.pending_release {
                0.0
            } else {
                10.0
            };
            let score = stage_bias + hv.env_level + (cv.velocity as f32 * 1.0e-3);
            if score < best_score || ((score - best_score).abs() < 1.0e-6 && cv.age < best_age) {
                best_score = score;
                best_age = cv.age;
                best_active_idx = active_idx;
                best_slot = slot;
            }
        }

        if stealing_from_rendered {
            // Move stolen slot from rendered prefix `[0..rendered_count)` into the
            // newly-spawned suffix `[rendered_count..active_len)`.
            self.rendered_count -= 1;
            if best_active_idx < self.rendered_count {
                self.active_list.swap(best_active_idx, self.rendered_count);
            }
            self.steal_cursor = if self.rendered_count > 0 {
                (base + 1) % self.rendered_count
            } else {
                0
            };
        } else {
            self.steal_cursor = (base + 1) % self.active_len;
        }

        Some(best_slot)
    }

    #[inline]
    fn note_on(&mut self, channel: usize, key: u8, velocity: u8) {
        if self.hot_voices.is_empty() || self.bank.presets.is_empty() || channel >= 16 {
            return;
        }
        let ch = self.channels[channel];

        // SF2 2.01 concave attenuation LUT for Velocity, CC7 Volume, and CC11 Expression.
        // Evaluate acoustic silence threshold BEFORE applying `master_gain` so low-velocity
        // notes (e.g., vel=1 with normal CC7/CC11) are never inadvertently culled.
        let vel_gain = self.concave_lut[(velocity & 0x7F) as usize];
        let acoustic_ch_gain = vel_gain * ch.volume * ch.expression;
        if acoustic_ch_gain <= 0.0 {
            return;
        }

        let bank = ch.effective_bank(channel);
        let preset = ch.program as u16;
        let sr = self.config.sample_rate.max(8_000) as f64;

        let preset_entry = match self.bank.find_preset_entry(bank, preset) {
            Some(entry) => entry,
            None => return,
        };
        let bucket_idx = preset_entry.key_bucket_base as usize + ((key as usize) & 0x7F);
        let (bucket_start, bucket_len) = match self.bank.key_region_ranges.get(bucket_idx) {
            Some(&(st, ln)) if ln > 0 => (st as usize, ln as usize),
            _ => return,
        };
        let use_precomp = self.config.sample_rate.max(8_000) == self.bank.precomp_sr;

        // Snapshot age before spawning multi-layer zones so exclusiveClass only chokes prior notes.
        let age_before_note = self.age_counter;
        let mut last_choked_class = 0u8;

        for b_off in 0..bucket_len {
            let region_idx = self.bank.key_region_indices[bucket_start + b_off] as usize;
            let Some(&region) = self.bank.regions.get(region_idx) else {
                break;
            };

            if velocity < region.vel_low || velocity > region.vel_high {
                continue;
            }

            if region.exclusive_class != 0 && region.exclusive_class != last_choked_class {
                self.choke_exclusive_class(channel, region.exclusive_class, age_before_note);
                last_choked_class = region.exclusive_class;
            }

            let acoustic_gain = acoustic_ch_gain * region.gain;
            if acoustic_gain < MIN_ENV_AMP {
                continue;
            }
            let combined_gain = acoustic_gain * self.master_gain;

            let Some(left_asset) = self.bank.samples.get(region.sample_idx as usize) else {
                continue;
            };
            if left_asset.sample_len < 2 {
                continue;
            }

            let sample_left_ptr = left_asset.data_ptr();
            let mut sample_len = left_asset.sample_len;
            let sample_sr = left_asset.sample_rate.max(1) as f64;

            let sample_right_ptr = if region.right_sample_idx != NO_SAMPLE {
                match self.bank.samples.get(region.right_sample_idx as usize) {
                    Some(ra) if ra.sample_len >= 2 => {
                        sample_len = sample_len.min(ra.sample_len);
                        ra.data_ptr()
                    }
                    _ => core::ptr::null(),
                }
            } else {
                core::ptr::null()
            };

            let Some(slot) = self.acquire_voice_slot() else {
                break;
            };

            if self.cold_voices[slot].in_key_list {
                self.unlink_voice_key(slot);
            }

            let pre = self.bank.region_precomp[region_idx];

            // Pitch calculation with SF2 `scaleTuning` (gen 56), `coarseTune`, `fineTune`, `overridingRootKey`.
            let base_step = if use_precomp && region.scale_tuning == 100.0 {
                let lut_idx = (key as i32 - region.root_key as i32 + 127).clamp(0, 254) as usize;
                self.bank.semitone_ratios[lut_idx] * pre.tune_sr_mul
            } else {
                let semitones = ((key as f64 - region.root_key as f64)
                    * (region.scale_tuning as f64 / 100.0))
                    + (region.tune_cents as f64 / 100.0);
                2.0f64.powf(semitones / 12.0) * (sample_sr / sr)
            };

            // Constant-power stereo panning combining SF2 `pan` (gen 17) and MIDI CC10.
            let total_pan = (ch.pan + region.pan).clamp(-1.0, 1.0);
            let (sin_a, cos_a) = ((total_pan + 1.0) * core::f32::consts::FRAC_PI_4).sin_cos();
            let (gain_l, gain_r) = if sample_right_ptr.is_null() {
                (combined_gain * cos_a, combined_gain * sin_a)
            } else {
                // Linked stereo pair: unity L/R at center without 3 dB mono collapse,
                // constant-power balance when panned left or right.
                let gl = (cos_a * core::f32::consts::SQRT_2).min(1.0);
                let gr = (sin_a * core::f32::consts::SQRT_2).min(1.0);
                (combined_gain * gl, combined_gain * gr)
            };

            let sr_f32 = sr as f32;
            let (delay_samples, attack_inc, hold_samples, decay_mul, release_mul) = if use_precomp {
                let key_delta = 60.0 - (key as f32);
                let hold_samples = if pre.has_key_to_hold {
                    (timecents_to_secs(region.env_hold_tc + key_delta * region.key_to_hold_tc)
                        * sr_f32) as u32
                } else {
                    pre.base_hold_samples
                };
                let decay_mul = if pre.has_key_to_decay {
                    let decay_samples = (timecents_to_secs(
                        region.env_decay_tc + key_delta * region.key_to_decay_tc,
                    )
                    .max(0.001)
                        * sr_f32)
                        .max(1.0);
                    (LN_ENV_ATTEN / decay_samples).exp().clamp(0.5, 0.999_999)
                } else {
                    pre.base_decay_mul
                };
                (
                    pre.delay_samples,
                    pre.attack_inc,
                    hold_samples,
                    decay_mul,
                    pre.release_mul,
                )
            } else {
                let key_delta = 60.0 - (key as f32);
                let delay_samples = (timecents_to_secs(region.env_delay_tc) * sr_f32) as u32;
                let attack_samples =
                    (timecents_to_secs(region.env_attack_tc).max(0.001) * sr_f32).max(1.0);
                let hold_samples = (timecents_to_secs(
                    region.env_hold_tc + key_delta * region.key_to_hold_tc,
                ) * sr_f32) as u32;
                let decay_samples = (timecents_to_secs(
                    region.env_decay_tc + key_delta * region.key_to_decay_tc,
                )
                .max(0.001)
                    * sr_f32)
                    .max(1.0);
                let release_samples =
                    (timecents_to_secs(region.env_release_tc).max(0.005) * sr_f32).max(16.0);
                (
                    delay_samples,
                    1.0 / attack_samples,
                    hold_samples,
                    (LN_ENV_ATTEN / decay_samples).exp().clamp(0.5, 0.999_999),
                    (LN_ENV_ATTEN / release_samples).exp().clamp(0.5, 0.999_999),
                )
            };

            // Bypassable resonant 2-pole LPF + SF2 2.01 default Velocity-to-Filter modulator.
            // Completely skipped (0% Note-On and 0% render overhead) when disabled via config.
            let (filter_active, filter) = if self.enable_lowpass_filter {
                let vel_fc_mod = region.vel_to_filter_cents * (1.0 - (velocity as f32) / 127.0);
                let eff_fc_cents = (region.filter_fc_cents + vel_fc_mod).clamp(1_500.0, 13_500.0);
                if use_precomp {
                    let active = eff_fc_cents < self.bank.filter_max_active_cents
                        || region.filter_q_cb > 0.5;
                    if active {
                        let t = (eff_fc_cents - 1_500.0)
                            * (FILTER_TRIG_LUT_STEPS as f32 / 12_000.0);
                        let idx = (t as usize).min(FILTER_TRIG_LUT_STEPS - 1);
                        let frac = t - (idx as f32);
                        let (c0, s0) = self.bank.filter_trig_lut[idx];
                        let (c1, s1) = self.bank.filter_trig_lut[idx + 1];
                        let cos_w0 = c0 + (c1 - c0) * frac;
                        let sin_w0 = s0 + (s1 - s0) * frac;
                        (
                            true,
                            VoiceFilter::from_cos_sin_inv2q(cos_w0, sin_w0, pre.filter_inv_2q),
                        )
                    } else {
                        (false, VoiceFilter::default())
                    }
                } else if eff_fc_cents >= 13_499.0 && region.filter_q_cb <= 0.5 {
                    (false, VoiceFilter::default())
                } else {
                    let fc_hz = 8.175_798_915_6 * 2.0f64.powf(eff_fc_cents as f64 / 1_200.0);
                    let active =
                        (eff_fc_cents < 13_499.0 && fc_hz < 0.45 * sr) || region.filter_q_cb > 0.5;
                    if active {
                        (true, VoiceFilter::from_fc_q(fc_hz, region.filter_q_cb, sr))
                    } else {
                        (false, VoiceFilter::default())
                    }
                }
            } else {
                (false, VoiceFilter::default())
            };

            let (vib_delay_samples, vib_inc_per_sample, vib_to_pitch_cents) =
                if self.enable_vibrato_lfo {
                    (
                        (region.vib_delay_secs * sr_f32) as u32,
                        (region.vib_freq_hz / sr_f32).clamp(1.0e-6, 0.05),
                        region.vib_to_pitch_cents,
                    )
                } else {
                    (0, 0.0, 0.0)
                };

            self.age_counter = self.age_counter.wrapping_add(1);
            let (stage, samples_left) = if delay_samples > 0 {
                (EnvStage::Delay, delay_samples)
            } else {
                (EnvStage::Attack, 0)
            };

            if let Some(hv) = self.hot_voices.get_mut(slot) {
                *hv = HotVoice {
                    sample_left_ptr,
                    sample_right_ptr,
                    pos_fp: (region.offset.min(sample_len - 1) as u64) << 32,
                    base_step,
                    sample_len,
                    loop_start: region.loop_start.min(sample_len - 1),
                    loop_end: region.loop_end.min(sample_len),
                    gain_l,
                    gain_r,
                    env_level: 0.0,
                    attack_inc,
                    decay_mul,
                    sustain_level: region.env_sustain,
                    release_mul,
                    env_samples_left: samples_left,
                    hold_samples,
                    filter,
                    vib_phase: 0.0,
                    vib_inc_per_sample,
                    vib_delay_samples,
                    vib_to_pitch_cents,
                    channel: channel as u8,
                    loop_mode: region.loop_mode,
                    env_stage: stage,
                    filter_active,
                    has_rendered: false,
                    pending_release: false,
                };
            }
            if let Some(cv) = self.cold_voices.get_mut(slot) {
                *cv = ColdVoice {
                    active: true,
                    channel: channel as u8,
                    key,
                    velocity,
                    exclusive_class: region.exclusive_class,
                    sustained: false,
                    sostenuto: false,
                    key_released: false,
                    in_key_list: false,
                    age: self.age_counter,
                    prev_on_key: NO_VOICE_LINK,
                    next_on_key: NO_VOICE_LINK,
                };
            }
            self.link_voice_key(slot, channel, key);
        }
    }

    #[inline]
    fn note_off(&mut self, channel: usize, key: u8) {
        if channel >= 16 {
            return;
        }
        let damper_down = self.channels[channel].sustain;
        let head_idx = (channel << 7) | ((key as usize) & 0x7F);
        let mut cur = self.key_heads[head_idx];
        self.key_heads[head_idx] = NO_VOICE_LINK;
        while cur != NO_VOICE_LINK {
            let slot = cur as usize;
            let Some(cv) = self.cold_voices.get_mut(slot) else {
                break;
            };
            cur = cv.next_on_key;
            cv.prev_on_key = NO_VOICE_LINK;
            cv.next_on_key = NO_VOICE_LINK;
            cv.in_key_list = false;
            if cv.active {
                cv.key_released = true;
                if let Some(hv) = self.hot_voices.get_mut(slot) {
                    if hv.loop_mode == RegionLoopMode::OneShot {
                        continue;
                    }
                    if damper_down {
                        cv.sustained = true;
                    } else if cv.sostenuto {
                        // Held by CC66 Sostenuto pedal until CC66 releases.
                    } else if !hv.has_rendered {
                        // Same-block Note-On + Note-Off: defer Release until attack/first block renders.
                        hv.pending_release = true;
                    } else if hv.env_stage != EnvStage::Release {
                        hv.env_stage = EnvStage::Release;
                    }
                }
            }
        }
    }

    #[inline]
    fn release_channel_sustained(&mut self, channel: usize) {
        let ch_u8 = channel as u8;
        for i in 0..self.active_len {
            let slot = match self.active_list.get(i) {
                Some(&s) => s as usize,
                None => continue,
            };
            if let Some(cv) = self.cold_voices.get_mut(slot) {
                if cv.active && cv.channel == ch_u8 && cv.sustained {
                    cv.sustained = false;
                    if !cv.sostenuto {
                        if let Some(hv) = self.hot_voices.get_mut(slot) {
                            if !hv.has_rendered {
                                hv.pending_release = true;
                            } else {
                                hv.env_stage = EnvStage::Release;
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn capture_channel_sostenuto(&mut self, channel: usize) {
        let ch_u8 = channel as u8;
        for i in 0..self.active_len {
            let slot = match self.active_list.get(i) {
                Some(&s) => s as usize,
                None => continue,
            };
            if let Some(cv) = self.cold_voices.get_mut(slot) {
                if cv.active && cv.channel == ch_u8 && !cv.key_released {
                    if let Some(hv) = self.hot_voices.get(slot) {
                        if hv.env_stage != EnvStage::Release && hv.env_stage != EnvStage::Inactive {
                            cv.sostenuto = true;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn release_channel_sostenuto(&mut self, channel: usize) {
        let ch_u8 = channel as u8;
        let damper_down = self.channels[channel].sustain;
        for i in 0..self.active_len {
            let slot = match self.active_list.get(i) {
                Some(&s) => s as usize,
                None => continue,
            };
            if let Some(cv) = self.cold_voices.get_mut(slot) {
                if cv.active && cv.channel == ch_u8 && cv.sostenuto {
                    cv.sostenuto = false;
                    if cv.key_released && !damper_down && !cv.sustained {
                        if let Some(hv) = self.hot_voices.get_mut(slot) {
                            if !hv.has_rendered {
                                hv.pending_release = true;
                            } else {
                                hv.env_stage = EnvStage::Release;
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn release_all_channel_notes(&mut self, channel: usize, immediate_kill: bool) {
        let ch_u8 = channel as u8;
        for i in 0..self.active_len {
            let slot = match self.active_list.get(i) {
                Some(&s) => s as usize,
                None => continue,
            };
            if let Some(cv) = self.cold_voices.get_mut(slot) {
                if cv.active && cv.channel == ch_u8 {
                    cv.sustained = false;
                    cv.sostenuto = false;
                    cv.key_released = true;
                    if let Some(hv) = self.hot_voices.get_mut(slot) {
                        if immediate_kill {
                            hv.release_mul = self.kill_release_mul;
                            hv.pending_release = false;
                            hv.env_stage = EnvStage::Release;
                        } else if !hv.has_rendered {
                            hv.pending_release = true;
                        } else {
                            hv.env_stage = EnvStage::Release;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    fn handle_cc(&mut self, channel: usize, cc: u8, val: u8) {
        if channel >= 16 {
            return;
        }
        match cc {
            0 => self.channels[channel].bank_msb = val,
            1 => self.channels[channel].mod_wheel = (val as f32) / 127.0,
            6 => {
                // Data Entry MSB for RPN 0 (Pitch Bend Sensitivity semitones)
                if self.channels[channel].rpn_msb == 0 && self.channels[channel].rpn_lsb == 0 {
                    self.channels[channel].pitch_bend_range_semis = (val as f32).clamp(0.0, 24.0);
                    self.channels[channel].update_pitch_ratio();
                }
            }
            7 => self.channels[channel].volume = self.concave_lut[(val & 0x7F) as usize],
            10 => self.channels[channel].pan = ((val as f32) - 64.0) / 63.0,
            11 => self.channels[channel].expression = self.concave_lut[(val & 0x7F) as usize],
            32 => self.channels[channel].bank_lsb = val,
            38 => {
                // Data Entry LSB for RPN 0 (Pitch Bend Sensitivity cents)
                if self.channels[channel].rpn_msb == 0 && self.channels[channel].rpn_lsb == 0 {
                    self.channels[channel].pitch_bend_range_cents = (val as f32).clamp(0.0, 99.0);
                    self.channels[channel].update_pitch_ratio();
                }
            }
            64 => {
                let down = val >= 64;
                let was_down = self.channels[channel].sustain;
                self.channels[channel].sustain = down;
                if was_down && !down {
                    self.release_channel_sustained(channel);
                }
            }
            66 => {
                let down = val >= 64;
                let was_down = self.channels[channel].sostenuto;
                self.channels[channel].sostenuto = down;
                if !was_down && down {
                    self.capture_channel_sostenuto(channel);
                } else if was_down && !down {
                    self.release_channel_sostenuto(channel);
                }
            }
            100 => self.channels[channel].rpn_lsb = val,
            101 => self.channels[channel].rpn_msb = val,
            120 => self.release_all_channel_notes(channel, true),
            121 => {
                let prev_prog = self.channels[channel].program;
                let prev_bank = self.channels[channel].bank_msb;
                self.channels[channel] = ChannelState::new(channel == 9, &self.concave_lut);
                self.channels[channel].program = prev_prog;
                self.channels[channel].bank_msb = prev_bank;
            }
            123 => self.release_all_channel_notes(channel, false),
            _ => {}
        }
    }
}

impl Default for SoundFontSynth {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthBackend for SoundFontSynth {
    fn name(&self) -> &'static str {
        "SoundFontSynth (SF2 2.01 + AVX2/FMA)"
    }

    fn init(&mut self, config: SynthConfig) -> Result<(), BackendError> {
        self.config = config;
        self.enable_lowpass_filter = config.lowpass_filter_active();
        self.enable_vibrato_lfo = config.vibrato_lfo_active();
        self.choke_release_mul = exp_decay_mul_for_secs(CHOKE_RELEASE_SECS, config.sample_rate);
        self.kill_release_mul = exp_decay_mul_for_secs(KILL_RELEASE_SECS, config.sample_rate);
        self.use_avx2_fma = detect_avx2_fma();
        self.allocate_pool(config.max_voices);
        self.ensure_render_pool(config.render_threads, config.buffer_frames);
        self.reset();
        Ok(())
    }

    fn shutdown(&mut self) {
        self.reset();
    }

    #[inline]
    fn reset(&mut self) {
        for i in 0..self.active_len {
            if let Some(&slot) = self.active_list.get(i) {
                let s = slot as usize;
                if let Some(hv) = self.hot_voices.get_mut(s) {
                    hv.env_stage = EnvStage::Inactive;
                    hv.env_level = 0.0;
                    hv.sample_left_ptr = core::ptr::null();
                    hv.sample_right_ptr = core::ptr::null();
                }
                if let Some(cv) = self.cold_voices.get_mut(s) {
                    cv.active = false;
                    cv.in_key_list = false;
                    cv.prev_on_key = NO_VOICE_LINK;
                    cv.next_on_key = NO_VOICE_LINK;
                }
                if self.free_len < self.free_stack.len() {
                    self.free_stack[self.free_len] = slot;
                    self.free_len += 1;
                }
            }
        }
        self.active_len = 0;
        self.rendered_count = 0;
        self.steal_cursor = 0;
        self.key_heads.fill(NO_VOICE_LINK);
        self.channels = default_channels(&self.concave_lut);
    }

    #[inline]
    fn process_short_event(&mut self, dw_msg: u32) {
        let status = (dw_msg & 0xFF) as u8;
        let msg_type = status & 0xF0;
        let channel = (status & 0x0F) as usize;
        let data1 = ((dw_msg >> 8) & 0x7F) as u8;
        let data2 = ((dw_msg >> 16) & 0x7F) as u8;

        match msg_type {
            0x90 if data2 > 0 => self.note_on(channel, data1, data2),
            0x80 | 0x90 => self.note_off(channel, data1),
            0xB0 => self.handle_cc(channel, data1, data2),
            0xC0 => {
                if channel < 16 {
                    self.channels[channel].program = data1;
                }
            }
            0xE0 => {
                if channel < 16 {
                    let raw14 = ((data2 as i32) << 7) | (data1 as i32);
                    self.channels[channel].pitch_bend_raw = ((raw14 - 8192) as f32) / 8192.0;
                    self.channels[channel].update_pitch_ratio();
                }
            }
            _ => {}
        }
    }

    #[inline]
    fn process_sysex(&mut self, data: &[u8]) {
        if data == [0xF0, 0x7E, 0x7F, 0x09, 0x01, 0xF7]
            || data == [0xF0, 0x41, 0x10, 0x42, 0x12, 0x40, 0x00, 0x7F, 0x00, 0x41, 0xF7]
            || data == [0xF0, 0x43, 0x10, 0x4C, 0x00, 0x00, 0x7E, 0x00, 0xF7]
        {
            self.reset();
        }
    }

    #[inline]
    fn process_custom_event(&mut self, _event_type: u32, _chan: u32, _param: u32) {}

    #[inline]
    fn render_audio(&mut self, output: &mut [f32]) {
        enable_ftz_daz();
        output.fill(0.0);

        let total_frames = output.len() / 2;
        if total_frames == 0 || self.active_len == 0 {
            return;
        }

        let interp = self.config.interpolation;
        let use_avx2_fma = self.use_avx2_fma && !self.force_scalar;
        let enable_vibrato_lfo = self.enable_vibrato_lfo;
        let mut pitch_ratios = [1.0f64; 16];
        let mut mod_wheels = [0.0f32; 16];
        for ch in 0..16 {
            pitch_ratios[ch] = self.channels[ch].pitch_bend_ratio;
            if enable_vibrato_lfo {
                mod_wheels[ch] = self.channels[ch].mod_wheel;
            }
        }

        let max_chunk_frames = self
            .render_pool
            .as_ref()
            .map(|p| p.max_frames)
            .unwrap_or(total_frames)
            .max(1);

        let mut frame_offset = 0usize;
        while frame_offset < total_frames && self.active_len > 0 {
            let frames = (total_frames - frame_offset).min(max_chunk_frames);
            let out_chunk = &mut output[frame_offset * 2..(frame_offset + frames) * 2];

            let use_parallel = self.parallel_enabled
                && self
                    .render_pool
                    .as_ref()
                    .map(|p| p.worker_count > 0)
                    .unwrap_or(false);

            if use_parallel {
                let active_len = self.active_len;
                let voices_len = self.hot_voices.len();
                let hot_voices_ptr = self.hot_voices.as_mut_ptr();
                let active_list_ptr = self.active_list.as_ptr();
                let alive_flags_ptr = self.alive_flags.as_mut_ptr();
                let sinc_tables_ptr = &self.sinc_tables as *const SincTables;

                if let Some(ref mut pool) = self.render_pool {
                    let job = RenderJob {
                        hot_voices_ptr,
                        voices_len,
                        active_list_ptr,
                        active_len,
                        alive_flags_ptr,
                        pitch_ratios,
                        mod_wheels,
                        sinc_tables_ptr,
                        interp,
                        use_avx2_fma,
                        enable_vibrato_lfo,
                        frames,
                        work_chunk: VOICE_WORK_CHUNK,
                    };
                    pool.render_parallel(job, out_chunk);
                }

                let mut i = 0usize;
                while i < self.active_len {
                    if self.alive_flags[i] != 0 {
                        i += 1;
                    } else {
                        let slot = self.active_list[i] as usize;
                        self.unlink_voice_key(slot);
                        if let Some(hv) = self.hot_voices.get_mut(slot) {
                            hv.env_stage = EnvStage::Inactive;
                        }
                        if let Some(cv) = self.cold_voices.get_mut(slot) {
                            cv.active = false;
                        }
                        self.active_len -= 1;
                        if i < self.active_len {
                            self.active_list[i] = self.active_list[self.active_len];
                            self.alive_flags[i] = self.alive_flags[self.active_len];
                        }
                        if self.free_len < self.free_stack.len() {
                            self.free_stack[self.free_len] = slot as u32;
                            self.free_len += 1;
                        }
                    }
                }
            } else {
                let mut active_idx = 0usize;
                while active_idx < self.active_len {
                    let slot = match self.active_list.get(active_idx) {
                        Some(&s) => s as usize,
                        None => break,
                    };

                    let voice_alive = if let Some(hv) = self.hot_voices.get_mut(slot) {
                        let ch = (hv.channel as usize).min(15);
                        render_hot_voice(
                            hv,
                            pitch_ratios[ch],
                            mod_wheels[ch],
                            enable_vibrato_lfo,
                            interp,
                            use_avx2_fma,
                            &self.sinc_tables,
                            out_chunk,
                            frames,
                        )
                    } else {
                        false
                    };

                    if voice_alive {
                        active_idx += 1;
                    } else {
                        self.unlink_voice_key(slot);
                        if let Some(hv) = self.hot_voices.get_mut(slot) {
                            hv.env_stage = EnvStage::Inactive;
                        }
                        if let Some(cv) = self.cold_voices.get_mut(slot) {
                            cv.active = false;
                        }
                        self.active_len -= 1;
                        if active_idx < self.active_len {
                            self.active_list[active_idx] = self.active_list[self.active_len];
                        }
                        if self.free_len < self.free_stack.len() {
                            self.free_stack[self.free_len] = slot as u32;
                            self.free_len += 1;
                        }
                    }
                }
            }

            frame_offset += frames;
        }

        // All surviving voices in `active_list[0..active_len]` have now rendered >= 1 block.
        self.rendered_count = self.active_len;
        if self.rendered_count > 0 {
            self.steal_cursor %= self.rendered_count;
        } else {
            self.steal_cursor = 0;
        }

        // Master output NaN/Inf guard + smooth C1 tanh soft-clipper strictly clamped to [-1.0, 1.0].
        // Prevents PipeWire/PulseAudio/ALSA from muting the audio sink on any non-finite sample.
        sanitize_and_soft_clip(output);
    }

    fn load_soundfont_list(&mut self, path: &Path) -> Result<(), BackendError> {
        if path.as_os_str().is_empty() {
            return Err(BackendError::InvalidPath);
        }
        self.load_soundfonts(&[path.to_path_buf()])
    }

    fn load_soundfonts(&mut self, paths: &[PathBuf]) -> Result<(), BackendError> {
        let new_bank = SoundFontBank::load_ordered(paths, self.config.sample_rate);
        if new_bank.preset_count() == 0 && !self.bank.presets.is_empty() {
            return Err(BackendError::InvalidPath);
        }
        self.reset();
        self.bank = new_bank;
        Ok(())
    }

    #[inline]
    fn stats(&self) -> SynthStats {
        SynthStats {
            active_voices: self.active_len as u64,
            rendering_time_pct: 0.0,
            xrun_detected: false,
        }
    }
}

#[inline]
fn default_channels(concave_lut: &[f32; 128]) -> [ChannelState; 16] {
    let mut ch = [ChannelState::new(false, concave_lut); 16];
    ch[9] = ChannelState::new(true, concave_lut);
    ch
}

/// Master output sanitizer & smooth C1 `tanh` soft-clipper strictly bounded to `[-1.0, 1.0]`.
///
/// Guarantees zero `NaN` / `Inf` samples ever reach Linux PipeWire/ALSA or Windows WASAPI.
#[inline]
pub fn sanitize_and_soft_clip(output: &mut [f32]) {
    for sample in output.iter_mut() {
        let mut x = *sample;
        if !x.is_finite() {
            x = 0.0;
        } else if x > 0.75 || x < -0.75 {
            let sign = x.signum();
            let excess = (x.abs() - 0.75) * 4.0;
            x = sign * (0.75 + 0.25 * excess.tanh());
        }
        *sample = x.clamp(-1.0, 1.0);
    }
}

// ============================================================================
// SIMD Scratch-Buffer Summation
// ============================================================================

#[inline]
fn sum_f32_slices(dst: &mut [f32], src: &[f32], use_avx2_fma: bool) {
    let n = dst.len().min(src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2_fma {
            // SAFETY: `use_avx2_fma` was verified at init via `is_x86_feature_detected!`.
            unsafe {
                sum_f32_slices_avx2(dst.as_mut_ptr(), src.as_ptr(), n);
            }
            return;
        }
    }
    let _ = use_avx2_fma;
    for i in 0..n {
        dst[i] += src[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_f32_slices_avx2(dst: *mut f32, src: *const f32, n: usize) {
    let mut i = 0usize;
    while i + 32 <= n {
        let a0 = _mm256_loadu_ps(dst.add(i));
        let b0 = _mm256_loadu_ps(src.add(i));
        _mm256_storeu_ps(dst.add(i), _mm256_add_ps(a0, b0));

        let a1 = _mm256_loadu_ps(dst.add(i + 8));
        let b1 = _mm256_loadu_ps(src.add(i + 8));
        _mm256_storeu_ps(dst.add(i + 8), _mm256_add_ps(a1, b1));

        let a2 = _mm256_loadu_ps(dst.add(i + 16));
        let b2 = _mm256_loadu_ps(src.add(i + 16));
        _mm256_storeu_ps(dst.add(i + 16), _mm256_add_ps(a2, b2));

        let a3 = _mm256_loadu_ps(dst.add(i + 24));
        let b3 = _mm256_loadu_ps(src.add(i + 24));
        _mm256_storeu_ps(dst.add(i + 24), _mm256_add_ps(a3, b3));
        i += 32;
    }
    while i + 8 <= n {
        let a = _mm256_loadu_ps(dst.add(i));
        let b = _mm256_loadu_ps(src.add(i));
        _mm256_storeu_ps(dst.add(i), _mm256_add_ps(a, b));
        i += 8;
    }
    while i < n {
        *dst.add(i) += *src.add(i);
        i += 1;
    }
}

// ============================================================================
// Branchless Fast-Path Voice Renderer (Precomputed `safe_frames` + AVX2/FMA)
// ============================================================================

#[derive(Clone, Copy)]
enum EnvRunMode {
    /// Constant amplitude (Sustain or Hold stage).
    Constant(f32),
    /// Linear ramp (Attack stage).
    LinearAttack { level: f32, inc: f32 },
    /// Exponential-in-dB decay toward `sustain` (pure `level *= mul`).
    ExpDecay { level: f32, sustain: f32, mul: f32 },
    /// Exponential-in-dB release toward 0 with `-96 dB` culling (pure `level *= mul`).
    ExpRelease { level: f32, mul: f32 },
}

/// Render a single `HotVoice` for `frames` stereo samples. Returns `false` once finished/culled.
#[inline]
#[allow(clippy::too_many_arguments)]
fn render_hot_voice(
    voice: &mut HotVoice,
    pitch_ratio: f64,
    mod_wheel: f32,
    enable_vibrato_lfo: bool,
    interp: InterpolationMode,
    use_avx2_fma: bool,
    tables: &SincTables,
    output: &mut [f32],
    frames: usize,
) -> bool {
    let sample_len = voice.sample_len as usize;
    if voice.sample_left_ptr.is_null() || sample_len < 2 || output.len() < frames * 2 {
        return false;
    }

    if voice.env_stage == EnvStage::Inactive {
        return false;
    }
    // Same-block Note-On + Note-Off on a delayed zone: skip delay so the attack+release still plays.
    if voice.pending_release && !voice.has_rendered && voice.env_stage == EnvStage::Delay {
        voice.env_samples_left = 0;
        voice.env_stage = EnvStage::Attack;
    }
    if voice.env_stage == EnvStage::Release && voice.env_level <= MIN_ENV_AMP {
        voice.env_level = 0.0;
        voice.env_stage = EnvStage::Inactive;
        return false;
    }
    if voice.env_stage == EnvStage::Sustain
        && voice.env_level <= MIN_ENV_AMP
        && voice.sustain_level <= MIN_ENV_AMP
    {
        voice.env_level = 0.0;
        voice.env_stage = EnvStage::Inactive;
        return false;
    }

    // Zero-overhead bypassable Vibrato LFO (CC1 Mod Wheel + SF2 `vibLfoToPitch`).
    let step = if !enable_vibrato_lfo {
        voice.base_step * pitch_ratio
    } else {
        let vib_cents = voice.vib_to_pitch_cents + mod_wheel * 50.0;
        if vib_cents.abs() < 0.01 {
            voice.base_step * pitch_ratio
        } else if voice.vib_delay_samples > 0 {
            voice.vib_delay_samples = voice.vib_delay_samples.saturating_sub(frames as u32);
            voice.base_step * pitch_ratio
        } else {
            let lfo = (voice.vib_phase * (2.0 * core::f32::consts::PI)).sin();
            voice.vib_phase =
                (voice.vib_phase + voice.vib_inc_per_sample * (frames as f32)).fract();
            voice.base_step * pitch_ratio * 2.0f64.powf((lfo * vib_cents) as f64 / 1_200.0)
        }
    };

    let looping = matches!(
        voice.loop_mode,
        RegionLoopMode::LoopContinuous | RegionLoopMode::LoopSustain
    ) && (voice.loop_mode == RegionLoopMode::LoopContinuous
        || voice.env_stage != EnvStage::Release)
        && (voice.loop_end as usize) > (voice.loop_start as usize + 1)
        && (voice.loop_end as usize) <= sample_len;

    let loop_start = voice.loop_start as usize;
    let loop_end = voice.loop_end as usize;
    let loop_len = loop_end.saturating_sub(loop_start);
    let loop_end_fp = (loop_end as u64) << 32;
    let loop_len_fp = (loop_len as u64) << 32;
    let max_pos_fp = ((sample_len - 1) as u64) << 32;
    let step_fp = (step * FP_SCALE).max(1.0) as u64;

    let tap_right: usize = match interp {
        InterpolationMode::Linear => 1,
        InterpolationMode::Sinc8 => 4,
        InterpolationMode::Sinc16 => 8,
        InterpolationMode::Sinc32 => 16,
    };
    let fast_limit_fp = if looping && loop_end > loop_start + tap_right + 1 {
        ((loop_end - tap_right) as u64) << 32
    } else if looping {
        loop_end_fp
    } else {
        max_pos_fp
    };

    let filter_active = voice.filter_active;
    let mut frame = 0usize;
    while frame < frames {
        let rem = frames - frame;

        if voice.env_stage == EnvStage::Delay {
            if voice.env_samples_left == 0 {
                voice.env_stage = EnvStage::Attack;
                continue;
            }
            let skip = rem.min(voice.env_samples_left as usize);
            voice.env_samples_left -= skip as u32;
            frame += skip;
            if voice.env_samples_left == 0 {
                voice.env_stage = EnvStage::Attack;
            }
            continue;
        }

        if looping {
            if voice.pos_fp >= loop_end_fp && loop_len_fp > 0 {
                voice.pos_fp = loop_end_fp
                    - loop_len_fp
                    + ((voice.pos_fp - loop_end_fp) % loop_len_fp);
            }
        } else if voice.pos_fp >= max_pos_fp {
            voice.env_stage = EnvStage::Inactive;
            return false;
        }

        let (stage_frames, env_mode) = match voice.env_stage {
            EnvStage::Inactive => return false,
            EnvStage::Delay => continue,
            EnvStage::Attack => {
                if voice.env_level >= 1.0 {
                    voice.env_level = 1.0;
                    if voice.pending_release {
                        voice.pending_release = false;
                        voice.env_stage = EnvStage::Release;
                    } else if voice.hold_samples > 0 {
                        voice.env_samples_left = voice.hold_samples;
                        voice.env_stage = EnvStage::Hold;
                    } else {
                        voice.env_stage = EnvStage::Decay;
                    }
                    continue;
                }
                let inc = voice.attack_inc.max(1.0e-6);
                let steps_to_peak = (((1.0 - voice.env_level) / inc).ceil() as usize).max(1);
                (
                    rem.min(steps_to_peak),
                    EnvRunMode::LinearAttack {
                        level: voice.env_level,
                        inc,
                    },
                )
            }
            EnvStage::Hold => {
                if voice.env_samples_left == 0 {
                    voice.env_stage = EnvStage::Decay;
                    continue;
                }
                (
                    rem.min(voice.env_samples_left as usize),
                    EnvRunMode::Constant(voice.env_level),
                )
            }
            EnvStage::Decay => {
                if voice.env_level <= voice.sustain_level {
                    voice.env_level = voice.sustain_level;
                    if voice.sustain_level <= MIN_ENV_AMP {
                        voice.env_level = 0.0;
                        voice.env_stage = EnvStage::Inactive;
                        return false;
                    }
                    voice.env_stage = EnvStage::Sustain;
                    continue;
                }
                (
                    rem,
                    EnvRunMode::ExpDecay {
                        level: voice.env_level,
                        sustain: voice.sustain_level,
                        mul: voice.decay_mul,
                    },
                )
            }
            EnvStage::Sustain => {
                if voice.env_level <= MIN_ENV_AMP && voice.sustain_level <= MIN_ENV_AMP {
                    voice.env_level = 0.0;
                    voice.env_stage = EnvStage::Inactive;
                    return false;
                }
                (rem, EnvRunMode::Constant(voice.env_level))
            }
            EnvStage::Release => {
                if voice.env_level <= MIN_ENV_AMP {
                    voice.env_level = 0.0;
                    voice.env_stage = EnvStage::Inactive;
                    return false;
                }
                (
                    rem,
                    EnvRunMode::ExpRelease {
                        level: voice.env_level,
                        mul: voice.release_mul,
                    },
                )
            }
        };

        if voice.pos_fp < fast_limit_fp {
            let dist_fp = fast_limit_fp - voice.pos_fp;
            let safe_frames = (((dist_fp - 1) / step_fp) as usize).saturating_add(1);
            let run = stage_frames.min(safe_frames);

            let out_ptr = unsafe { output.as_mut_ptr().add(frame * 2) };
            let (new_pos_fp, end_env, rendered) = if filter_active {
                render_fast_block::<true>(
                    voice.sample_left_ptr,
                    voice.sample_right_ptr,
                    voice.pos_fp,
                    step_fp,
                    voice.gain_l,
                    voice.gain_r,
                    env_mode,
                    &mut voice.filter,
                    interp,
                    use_avx2_fma,
                    tables,
                    out_ptr,
                    run,
                )
            } else {
                render_fast_block::<false>(
                    voice.sample_left_ptr,
                    voice.sample_right_ptr,
                    voice.pos_fp,
                    step_fp,
                    voice.gain_l,
                    voice.gain_r,
                    env_mode,
                    &mut voice.filter,
                    interp,
                    use_avx2_fma,
                    tables,
                    out_ptr,
                    run,
                )
            };

            voice.pos_fp = new_pos_fp;
            voice.env_level = end_env;
            frame += rendered;

            match voice.env_stage {
                EnvStage::Attack => {
                    if voice.env_level >= 1.0 {
                        voice.env_level = 1.0;
                        if voice.pending_release {
                            voice.pending_release = false;
                            voice.env_stage = EnvStage::Release;
                        } else if voice.hold_samples > 0 {
                            voice.env_samples_left = voice.hold_samples;
                            voice.env_stage = EnvStage::Hold;
                        } else {
                            voice.env_stage = EnvStage::Decay;
                        }
                    }
                }
                EnvStage::Hold => {
                    voice.env_samples_left = voice.env_samples_left.saturating_sub(rendered as u32);
                    if voice.env_samples_left == 0 {
                        voice.env_stage = EnvStage::Decay;
                    }
                }
                EnvStage::Decay => {
                    if voice.env_level <= voice.sustain_level {
                        voice.env_level = voice.sustain_level;
                        if voice.sustain_level <= MIN_ENV_AMP {
                            voice.env_level = 0.0;
                            voice.env_stage = EnvStage::Inactive;
                            return false;
                        }
                        voice.env_stage = EnvStage::Sustain;
                    }
                }
                EnvStage::Release => {
                    if voice.env_level <= MIN_ENV_AMP {
                        voice.env_level = 0.0;
                        voice.env_stage = EnvStage::Inactive;
                        return false;
                    }
                }
                _ => {}
            }
        } else {
            let seam_rendered = render_loop_seam_sample(
                voice,
                step_fp,
                loop_start,
                loop_end,
                loop_end_fp,
                loop_len_fp,
                interp,
                tables,
                &mut output[frame * 2..frame * 2 + 2],
            );
            if !seam_rendered {
                voice.env_stage = EnvStage::Inactive;
                return false;
            }
            frame += 1;
        }
    }

    // Mark voice as having rendered at least one audio block.
    voice.has_rendered = true;
    // If a same-block Note-Off arrived and Attack lasted the full block, transition cleanly
    // into Release now so the note decays naturally on subsequent blocks instead of sticking.
    if voice.pending_release {
        voice.pending_release = false;
        if voice.env_stage != EnvStage::Inactive {
            voice.env_stage = EnvStage::Release;
        }
    }
    // Guard filter state against any numerical NaN/Inf runaway.
    if voice.filter_active
        && (!voice.filter.l_z1.is_finite()
            || !voice.filter.l_z2.is_finite()
            || !voice.filter.r_z1.is_finite()
            || !voice.filter.r_z2.is_finite())
    {
        voice.filter.l_z1 = 0.0;
        voice.filter.l_z2 = 0.0;
        voice.filter.r_z1 = 0.0;
        voice.filter.r_z2 = 0.0;
    }

    true
}

/// Dispatch a branchless `run`-frame block monomorphized on `FILTER` so `FILTER == false`
/// compiles with literally 0% filter overhead.
#[inline]
#[allow(clippy::too_many_arguments)]
fn render_fast_block<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    interp: InterpolationMode,
    use_avx2_fma: bool,
    tables: &SincTables,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let is_stereo = !right_ptr.is_null();

    #[cfg(target_arch = "x86_64")]
    if use_avx2_fma {
        match (interp, is_stereo) {
            (InterpolationMode::Sinc8, false) => {
                return unsafe {
                    render_fast_sinc8_mono_avx2::<FILTER>(
                        left_ptr,
                        pos_fp,
                        step_fp,
                        gain_l,
                        gain_r,
                        env_mode,
                        flt,
                        &tables.sinc8,
                        out_ptr,
                        run,
                    )
                };
            }
            (InterpolationMode::Sinc8, true) => {
                return unsafe {
                    render_fast_sinc8_stereo_avx2::<FILTER>(
                        left_ptr,
                        right_ptr,
                        pos_fp,
                        step_fp,
                        gain_l,
                        gain_r,
                        env_mode,
                        flt,
                        &tables.sinc8,
                        out_ptr,
                        run,
                    )
                };
            }
            (InterpolationMode::Sinc16, false) => {
                return unsafe {
                    render_fast_sinc16_mono_avx2::<FILTER>(
                        left_ptr,
                        pos_fp,
                        step_fp,
                        gain_l,
                        gain_r,
                        env_mode,
                        flt,
                        &tables.sinc16,
                        out_ptr,
                        run,
                    )
                };
            }
            (InterpolationMode::Sinc16, true) => {
                return unsafe {
                    render_fast_sinc16_stereo_avx2::<FILTER>(
                        left_ptr,
                        right_ptr,
                        pos_fp,
                        step_fp,
                        gain_l,
                        gain_r,
                        env_mode,
                        flt,
                        &tables.sinc16,
                        out_ptr,
                        run,
                    )
                };
            }
            (InterpolationMode::Sinc32, false) => {
                return unsafe {
                    render_fast_sinc32_mono_avx2::<FILTER>(
                        left_ptr,
                        pos_fp,
                        step_fp,
                        gain_l,
                        gain_r,
                        env_mode,
                        flt,
                        &tables.sinc32,
                        out_ptr,
                        run,
                    )
                };
            }
            (InterpolationMode::Sinc32, true) => {
                return unsafe {
                    render_fast_sinc32_stereo_avx2::<FILTER>(
                        left_ptr,
                        right_ptr,
                        pos_fp,
                        step_fp,
                        gain_l,
                        gain_r,
                        env_mode,
                        flt,
                        &tables.sinc32,
                        out_ptr,
                        run,
                    )
                };
            }
            (InterpolationMode::Linear, _) => {}
        }
    }

    let _ = use_avx2_fma;
    match (interp, is_stereo) {
        (InterpolationMode::Linear, false) => unsafe {
            render_fast_linear_mono::<FILTER>(
                left_ptr, pos_fp, step_fp, gain_l, gain_r, env_mode, flt, out_ptr, run,
            )
        },
        (InterpolationMode::Linear, true) => unsafe {
            render_fast_linear_stereo::<FILTER>(
                left_ptr, right_ptr, pos_fp, step_fp, gain_l, gain_r, env_mode, flt, out_ptr, run,
            )
        },
        (InterpolationMode::Sinc8, false) => unsafe {
            render_fast_sinc8_mono_scalar::<FILTER>(
                left_ptr,
                pos_fp,
                step_fp,
                gain_l,
                gain_r,
                env_mode,
                flt,
                &tables.sinc8,
                out_ptr,
                run,
            )
        },
        (InterpolationMode::Sinc8, true) => unsafe {
            render_fast_sinc8_stereo_scalar::<FILTER>(
                left_ptr,
                right_ptr,
                pos_fp,
                step_fp,
                gain_l,
                gain_r,
                env_mode,
                flt,
                &tables.sinc8,
                out_ptr,
                run,
            )
        },
        (InterpolationMode::Sinc16, false) => unsafe {
            render_fast_sinc16_mono_scalar::<FILTER>(
                left_ptr,
                pos_fp,
                step_fp,
                gain_l,
                gain_r,
                env_mode,
                flt,
                &tables.sinc16,
                out_ptr,
                run,
            )
        },
        (InterpolationMode::Sinc16, true) => unsafe {
            render_fast_sinc16_stereo_scalar::<FILTER>(
                left_ptr,
                right_ptr,
                pos_fp,
                step_fp,
                gain_l,
                gain_r,
                env_mode,
                flt,
                &tables.sinc16,
                out_ptr,
                run,
            )
        },
        (InterpolationMode::Sinc32, false) => unsafe {
            render_fast_sinc32_mono_scalar::<FILTER>(
                left_ptr,
                pos_fp,
                step_fp,
                gain_l,
                gain_r,
                env_mode,
                flt,
                &tables.sinc32,
                out_ptr,
                run,
            )
        },
        (InterpolationMode::Sinc32, true) => unsafe {
            render_fast_sinc32_stereo_scalar::<FILTER>(
                left_ptr,
                right_ptr,
                pos_fp,
                step_fp,
                gain_l,
                gain_r,
                env_mode,
                flt,
                &tables.sinc32,
                out_ptr,
                run,
            )
        },
    }
}

#[inline(always)]
fn step_env_scalar(mode: &mut EnvRunMode) -> f32 {
    match *mode {
        EnvRunMode::Constant(v) => v,
        EnvRunMode::LinearAttack { ref mut level, inc } => {
            *level = (*level + inc).min(1.0);
            *level
        }
        EnvRunMode::ExpDecay {
            ref mut level,
            sustain,
            mul,
        } => {
            *level = (*level * mul).max(sustain);
            *level
        }
        EnvRunMode::ExpRelease { ref mut level, mul } => {
            *level *= mul;
            *level
        }
    }
}

#[inline(always)]
fn current_env_level(mode: &EnvRunMode) -> f32 {
    match *mode {
        EnvRunMode::Constant(v) => v,
        EnvRunMode::LinearAttack { level, .. } => level,
        EnvRunMode::ExpDecay { level, .. } => level,
        EnvRunMode::ExpRelease { level, .. } => level,
    }
}

#[inline(always)]
fn should_stop_env(mode: &EnvRunMode) -> bool {
    match *mode {
        EnvRunMode::ExpRelease { level, .. } => level <= MIN_ENV_AMP,
        EnvRunMode::ExpDecay { level, sustain, .. } => level <= sustain,
        _ => false,
    }
}

// ----------------------------------------------------------------------------
// Explicit AVX2 + FMA Kernels (Sinc8, Sinc16, Sinc32 — 4-frame batched SIMD)
// ----------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn hsum4_avx2(p0: __m256, p1: __m256, p2: __m256, p3: __m256) -> __m128 {
    let t01 = _mm256_hadd_ps(p0, p1);
    let t23 = _mm256_hadd_ps(p2, p3);
    let t0123 = _mm256_hadd_ps(t01, t23);
    let lo = _mm256_castps256_ps128(t0123);
    let hi = _mm256_extractf128_ps(t0123, 1);
    _mm_add_ps(lo, hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn hsum1_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 1);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn accumulate_stereo4_avx2(dst_ptr: *mut f32, l4: __m128, r4: __m128) {
    let inter_lo = _mm_unpacklo_ps(l4, r4);
    let inter_hi = _mm_unpackhi_ps(l4, r4);
    let stereo4 = _mm256_insertf128_ps(_mm256_castps128_ps256(inter_lo), inter_hi, 1);
    let cur = _mm256_loadu_ps(dst_ptr);
    _mm256_storeu_ps(dst_ptr, _mm256_add_ps(cur, stereo4));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn apply_lpf4_left_avx2(samples: __m128, flt: &mut VoiceFilter) -> __m128 {
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), samples);
    let y0 = flt.step_left(tmp[0]);
    let y1 = flt.step_left(tmp[1]);
    let y2 = flt.step_left(tmp[2]);
    let y3 = flt.step_left(tmp[3]);
    _mm_set_ps(y3, y2, y1, y0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn apply_lpf4_stereo_avx2(
    sl4: __m128,
    sr4: __m128,
    flt: &mut VoiceFilter,
) -> (__m128, __m128) {
    use core::arch::x86_64::*;

    let b0 = _mm_set1_ps(flt.b0);
    let a1 = _mm_set1_ps(flt.a1);
    let a2 = _mm_set1_ps(flt.a2);

    // Interleave [L0, R0, L1, R1] & [L2, R2, L3, R3], pre-scaled by b0 and 2*b0.
    let p01 = _mm_unpacklo_ps(sl4, sr4);
    let p23 = _mm_unpackhi_ps(sl4, sr4);
    let bp01 = _mm_mul_ps(p01, b0);
    let bp23 = _mm_mul_ps(p23, b0);
    let b1p01 = _mm_add_ps(bp01, bp01);
    let b1p23 = _mm_add_ps(bp23, bp23);

    let b0x0 = bp01;
    let b0x1 = _mm_movehl_ps(bp01, bp01);
    let b0x2 = bp23;
    let b0x3 = _mm_movehl_ps(bp23, bp23);

    let b1x0 = b1p01;
    let b1x1 = _mm_movehl_ps(b1p01, b1p01);
    let b1x2 = b1p23;
    let b1x3 = _mm_movehl_ps(b1p23, b1p23);

    let mut z1 = _mm_set_ps(0.0, 0.0, flt.r_z1, flt.l_z1);
    let mut z2 = _mm_set_ps(0.0, 0.0, flt.r_z2, flt.l_z2);

    let y0 = _mm_add_ps(b0x0, z1);
    z1 = _mm_fnmadd_ps(a1, y0, _mm_add_ps(b1x0, z2));
    z2 = _mm_fnmadd_ps(a2, y0, b0x0);

    let y1 = _mm_add_ps(b0x1, z1);
    z1 = _mm_fnmadd_ps(a1, y1, _mm_add_ps(b1x1, z2));
    z2 = _mm_fnmadd_ps(a2, y1, b0x1);

    let y2 = _mm_add_ps(b0x2, z1);
    z1 = _mm_fnmadd_ps(a1, y2, _mm_add_ps(b1x2, z2));
    z2 = _mm_fnmadd_ps(a2, y2, b0x2);

    let y3 = _mm_add_ps(b0x3, z1);
    z1 = _mm_fnmadd_ps(a1, y3, _mm_add_ps(b1x3, z2));
    z2 = _mm_fnmadd_ps(a2, y3, b0x3);

    flt.l_z1 = _mm_cvtss_f32(z1);
    flt.r_z1 = _mm_cvtss_f32(_mm_shuffle_ps(z1, z1, 1));
    flt.l_z2 = _mm_cvtss_f32(z2);
    flt.r_z2 = _mm_cvtss_f32(_mm_shuffle_ps(z2, z2, 1));

    let y01 = _mm_movelh_ps(y0, y1);
    let y23 = _mm_movelh_ps(y2, y3);
    let out_l4 = _mm_shuffle_ps(y01, y23, 0x88);
    let out_r4 = _mm_shuffle_ps(y01, y23, 0xDD);
    (out_l4, out_r4)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn env4_avx2(mode: &mut EnvRunMode) -> __m128 {
    match *mode {
        EnvRunMode::Constant(v) => _mm_set1_ps(v),
        EnvRunMode::LinearAttack { ref mut level, inc } => {
            let e0 = (*level + inc).min(1.0);
            let e1 = (e0 + inc).min(1.0);
            let e2 = (e1 + inc).min(1.0);
            let e3 = (e2 + inc).min(1.0);
            *level = e3;
            _mm_set_ps(e3, e2, e1, e0)
        }
        EnvRunMode::ExpDecay {
            ref mut level,
            sustain,
            mul,
        } => {
            let e0 = (*level * mul).max(sustain);
            let e1 = (e0 * mul).max(sustain);
            let e2 = (e1 * mul).max(sustain);
            let e3 = (e2 * mul).max(sustain);
            *level = e3;
            _mm_set_ps(e3, e2, e1, e0)
        }
        EnvRunMode::ExpRelease { ref mut level, mul } => {
            let e0 = *level * mul;
            let e1 = e0 * mul;
            let e2 = e1 * mul;
            let e3 = e2 * mul;
            *level = e3;
            _mm_set_ps(e3, e2, e1, e0)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc8_mono_avx2<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc8,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let lut_base = lut.rows.as_ptr() as *const f32;
    let gl_vec = _mm_set1_ps(gain_l);
    let gr_vec = _mm_set1_ps(gain_r);
    let mut f = 0usize;

    while f + 4 <= run {
        let p0 = pos_fp;
        let p1 = p0.wrapping_add(step_fp);
        let p2 = p1.wrapping_add(step_fp);
        let p3 = p2.wrapping_add(step_fp);
        pos_fp = p3.wrapping_add(step_fp);

        let m0 = _mm256_mul_ps(
            _mm256_loadu_ps(left_ptr.add((p0 >> 32) as usize).sub(3)),
            _mm256_load_ps(lut_base.add((((p0 >> 24) & 0xFF) as usize) * 8)),
        );
        let m1 = _mm256_mul_ps(
            _mm256_loadu_ps(left_ptr.add((p1 >> 32) as usize).sub(3)),
            _mm256_load_ps(lut_base.add((((p1 >> 24) & 0xFF) as usize) * 8)),
        );
        let m2 = _mm256_mul_ps(
            _mm256_loadu_ps(left_ptr.add((p2 >> 32) as usize).sub(3)),
            _mm256_load_ps(lut_base.add((((p2 >> 24) & 0xFF) as usize) * 8)),
        );
        let m3 = _mm256_mul_ps(
            _mm256_loadu_ps(left_ptr.add((p3 >> 32) as usize).sub(3)),
            _mm256_load_ps(lut_base.add((((p3 >> 24) & 0xFF) as usize) * 8)),
        );

        let mut samples4 = hsum4_avx2(m0, m1, m2, m3);
        if FILTER {
            samples4 = apply_lpf4_left_avx2(samples4, flt);
        }
        let env4 = env4_avx2(&mut env_mode);
        let scaled = _mm_mul_ps(samples4, env4);
        accumulate_stereo4_avx2(
            out_ptr.add(f * 2),
            _mm_mul_ps(scaled, gl_vec),
            _mm_mul_ps(scaled, gr_vec),
        );

        f += 4;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let mut s = hsum1_avx2(_mm256_mul_ps(
            _mm256_loadu_ps(left_ptr.add(idx).sub(3)),
            _mm256_load_ps(lut_base.add(ph * 8)),
        ));
        if FILTER {
            s = flt.step_left(s);
        }
        let val = s * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += val * gain_l;
        *dst.add(1) += val * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    (pos_fp, current_env_level(&env_mode), f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc8_stereo_avx2<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc8,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let lut_base = lut.rows.as_ptr() as *const f32;
    let gl_vec = _mm_set1_ps(gain_l);
    let gr_vec = _mm_set1_ps(gain_r);
    let mut f = 0usize;

    while f + 4 <= run {
        let p0 = pos_fp;
        let p1 = p0.wrapping_add(step_fp);
        let p2 = p1.wrapping_add(step_fp);
        let p3 = p2.wrapping_add(step_fp);
        pos_fp = p3.wrapping_add(step_fp);

        let i0 = (p0 >> 32) as usize;
        let i1 = (p1 >> 32) as usize;
        let i2 = (p2 >> 32) as usize;
        let i3 = (p3 >> 32) as usize;

        let k0 = _mm256_load_ps(lut_base.add((((p0 >> 24) & 0xFF) as usize) * 8));
        let k1 = _mm256_load_ps(lut_base.add((((p1 >> 24) & 0xFF) as usize) * 8));
        let k2 = _mm256_load_ps(lut_base.add((((p2 >> 24) & 0xFF) as usize) * 8));
        let k3 = _mm256_load_ps(lut_base.add((((p3 >> 24) & 0xFF) as usize) * 8));

        let mut sl4 = hsum4_avx2(
            _mm256_mul_ps(_mm256_loadu_ps(left_ptr.add(i0).sub(3)), k0),
            _mm256_mul_ps(_mm256_loadu_ps(left_ptr.add(i1).sub(3)), k1),
            _mm256_mul_ps(_mm256_loadu_ps(left_ptr.add(i2).sub(3)), k2),
            _mm256_mul_ps(_mm256_loadu_ps(left_ptr.add(i3).sub(3)), k3),
        );
        let mut sr4 = hsum4_avx2(
            _mm256_mul_ps(_mm256_loadu_ps(right_ptr.add(i0).sub(3)), k0),
            _mm256_mul_ps(_mm256_loadu_ps(right_ptr.add(i1).sub(3)), k1),
            _mm256_mul_ps(_mm256_loadu_ps(right_ptr.add(i2).sub(3)), k2),
            _mm256_mul_ps(_mm256_loadu_ps(right_ptr.add(i3).sub(3)), k3),
        );
        if FILTER {
            let (fl, fr) = apply_lpf4_stereo_avx2(sl4, sr4, flt);
            sl4 = fl;
            sr4 = fr;
        }

        let env4 = env4_avx2(&mut env_mode);
        accumulate_stereo4_avx2(
            out_ptr.add(f * 2),
            _mm_mul_ps(sl4, _mm_mul_ps(env4, gl_vec)),
            _mm_mul_ps(sr4, _mm_mul_ps(env4, gr_vec)),
        );

        f += 4;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = _mm256_load_ps(lut_base.add(ph * 8));
        let mut sl = hsum1_avx2(_mm256_mul_ps(_mm256_loadu_ps(left_ptr.add(idx).sub(3)), k));
        let mut sr = hsum1_avx2(_mm256_mul_ps(_mm256_loadu_ps(right_ptr.add(idx).sub(3)), k));
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    (pos_fp, current_env_level(&env_mode), f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc16_mono_avx2<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc16,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let lut_base = lut.rows.as_ptr() as *const f32;
    let gl_vec = _mm_set1_ps(gain_l);
    let gr_vec = _mm_set1_ps(gain_r);
    let mut f = 0usize;

    while f + 4 <= run {
        let p0 = pos_fp;
        let p1 = p0.wrapping_add(step_fp);
        let p2 = p1.wrapping_add(step_fp);
        let p3 = p2.wrapping_add(step_fp);
        pos_fp = p3.wrapping_add(step_fp);

        let kp0 = lut_base.add((((p0 >> 24) & 0xFF) as usize) * 16);
        let kp1 = lut_base.add((((p1 >> 24) & 0xFF) as usize) * 16);
        let kp2 = lut_base.add((((p2 >> 24) & 0xFF) as usize) * 16);
        let kp3 = lut_base.add((((p3 >> 24) & 0xFF) as usize) * 16);

        let sp0 = left_ptr.add((p0 >> 32) as usize).sub(7);
        let sp1 = left_ptr.add((p1 >> 32) as usize).sub(7);
        let sp2 = left_ptr.add((p2 >> 32) as usize).sub(7);
        let sp3 = left_ptr.add((p3 >> 32) as usize).sub(7);

        let m0 = _mm256_fmadd_ps(
            _mm256_loadu_ps(sp0.add(8)),
            _mm256_load_ps(kp0.add(8)),
            _mm256_mul_ps(_mm256_loadu_ps(sp0), _mm256_load_ps(kp0)),
        );
        let m1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(sp1.add(8)),
            _mm256_load_ps(kp1.add(8)),
            _mm256_mul_ps(_mm256_loadu_ps(sp1), _mm256_load_ps(kp1)),
        );
        let m2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(sp2.add(8)),
            _mm256_load_ps(kp2.add(8)),
            _mm256_mul_ps(_mm256_loadu_ps(sp2), _mm256_load_ps(kp2)),
        );
        let m3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(sp3.add(8)),
            _mm256_load_ps(kp3.add(8)),
            _mm256_mul_ps(_mm256_loadu_ps(sp3), _mm256_load_ps(kp3)),
        );

        let mut samples4 = hsum4_avx2(m0, m1, m2, m3);
        if FILTER {
            samples4 = apply_lpf4_left_avx2(samples4, flt);
        }
        let env4 = env4_avx2(&mut env_mode);
        let scaled = _mm_mul_ps(samples4, env4);
        accumulate_stereo4_avx2(
            out_ptr.add(f * 2),
            _mm_mul_ps(scaled, gl_vec),
            _mm_mul_ps(scaled, gr_vec),
        );

        f += 4;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let kp = lut_base.add(ph * 16);
        let sp = left_ptr.add(idx).sub(7);
        let mut s = hsum1_avx2(_mm256_fmadd_ps(
            _mm256_loadu_ps(sp.add(8)),
            _mm256_load_ps(kp.add(8)),
            _mm256_mul_ps(_mm256_loadu_ps(sp), _mm256_load_ps(kp)),
        ));
        if FILTER {
            s = flt.step_left(s);
        }
        let val = s * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += val * gain_l;
        *dst.add(1) += val * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    (pos_fp, current_env_level(&env_mode), f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc16_stereo_avx2<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc16,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let lut_base = lut.rows.as_ptr() as *const f32;
    let gl_vec = _mm_set1_ps(gain_l);
    let gr_vec = _mm_set1_ps(gain_r);
    let mut f = 0usize;

    while f + 4 <= run {
        let p0 = pos_fp;
        let p1 = p0.wrapping_add(step_fp);
        let p2 = p1.wrapping_add(step_fp);
        let p3 = p2.wrapping_add(step_fp);
        pos_fp = p3.wrapping_add(step_fp);

        let i0 = (p0 >> 32) as usize;
        let i1 = (p1 >> 32) as usize;
        let i2 = (p2 >> 32) as usize;
        let i3 = (p3 >> 32) as usize;

        let kp0 = lut_base.add((((p0 >> 24) & 0xFF) as usize) * 16);
        let kp1 = lut_base.add((((p1 >> 24) & 0xFF) as usize) * 16);
        let kp2 = lut_base.add((((p2 >> 24) & 0xFF) as usize) * 16);
        let kp3 = lut_base.add((((p3 >> 24) & 0xFF) as usize) * 16);

        let k0a = _mm256_load_ps(kp0);
        let k0b = _mm256_load_ps(kp0.add(8));
        let k1a = _mm256_load_ps(kp1);
        let k1b = _mm256_load_ps(kp1.add(8));
        let k2a = _mm256_load_ps(kp2);
        let k2b = _mm256_load_ps(kp2.add(8));
        let k3a = _mm256_load_ps(kp3);
        let k3b = _mm256_load_ps(kp3.add(8));

        let lp0 = left_ptr.add(i0).sub(7);
        let lp1 = left_ptr.add(i1).sub(7);
        let lp2 = left_ptr.add(i2).sub(7);
        let lp3 = left_ptr.add(i3).sub(7);
        let mut sl4 = hsum4_avx2(
            _mm256_fmadd_ps(_mm256_loadu_ps(lp0.add(8)), k0b, _mm256_mul_ps(_mm256_loadu_ps(lp0), k0a)),
            _mm256_fmadd_ps(_mm256_loadu_ps(lp1.add(8)), k1b, _mm256_mul_ps(_mm256_loadu_ps(lp1), k1a)),
            _mm256_fmadd_ps(_mm256_loadu_ps(lp2.add(8)), k2b, _mm256_mul_ps(_mm256_loadu_ps(lp2), k2a)),
            _mm256_fmadd_ps(_mm256_loadu_ps(lp3.add(8)), k3b, _mm256_mul_ps(_mm256_loadu_ps(lp3), k3a)),
        );

        let rp0 = right_ptr.add(i0).sub(7);
        let rp1 = right_ptr.add(i1).sub(7);
        let rp2 = right_ptr.add(i2).sub(7);
        let rp3 = right_ptr.add(i3).sub(7);
        let mut sr4 = hsum4_avx2(
            _mm256_fmadd_ps(_mm256_loadu_ps(rp0.add(8)), k0b, _mm256_mul_ps(_mm256_loadu_ps(rp0), k0a)),
            _mm256_fmadd_ps(_mm256_loadu_ps(rp1.add(8)), k1b, _mm256_mul_ps(_mm256_loadu_ps(rp1), k1a)),
            _mm256_fmadd_ps(_mm256_loadu_ps(rp2.add(8)), k2b, _mm256_mul_ps(_mm256_loadu_ps(rp2), k2a)),
            _mm256_fmadd_ps(_mm256_loadu_ps(rp3.add(8)), k3b, _mm256_mul_ps(_mm256_loadu_ps(rp3), k3a)),
        );

        if FILTER {
            let (fl, fr) = apply_lpf4_stereo_avx2(sl4, sr4, flt);
            sl4 = fl;
            sr4 = fr;
        }
        let env4 = env4_avx2(&mut env_mode);
        accumulate_stereo4_avx2(
            out_ptr.add(f * 2),
            _mm_mul_ps(sl4, _mm_mul_ps(env4, gl_vec)),
            _mm_mul_ps(sr4, _mm_mul_ps(env4, gr_vec)),
        );

        f += 4;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let kp = lut_base.add(ph * 16);
        let ka = _mm256_load_ps(kp);
        let kb = _mm256_load_ps(kp.add(8));
        let lp = left_ptr.add(idx).sub(7);
        let rp = right_ptr.add(idx).sub(7);

        let mut sl = hsum1_avx2(_mm256_fmadd_ps(
            _mm256_loadu_ps(lp.add(8)),
            kb,
            _mm256_mul_ps(_mm256_loadu_ps(lp), ka),
        ));
        let mut sr = hsum1_avx2(_mm256_fmadd_ps(
            _mm256_loadu_ps(rp.add(8)),
            kb,
            _mm256_mul_ps(_mm256_loadu_ps(rp), ka),
        ));
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    (pos_fp, current_env_level(&env_mode), f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn sinc32_dot_avx2(sp: *const f32, kp: *const f32) -> __m256 {
    let mut acc = _mm256_mul_ps(_mm256_loadu_ps(sp), _mm256_load_ps(kp));
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(sp.add(8)), _mm256_load_ps(kp.add(8)), acc);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(sp.add(16)), _mm256_load_ps(kp.add(16)), acc);
    _mm256_fmadd_ps(_mm256_loadu_ps(sp.add(24)), _mm256_load_ps(kp.add(24)), acc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc32_mono_avx2<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc32,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let lut_base = lut.rows.as_ptr() as *const f32;
    let gl_vec = _mm_set1_ps(gain_l);
    let gr_vec = _mm_set1_ps(gain_r);
    let mut f = 0usize;

    while f + 4 <= run {
        let p0 = pos_fp;
        let p1 = p0.wrapping_add(step_fp);
        let p2 = p1.wrapping_add(step_fp);
        let p3 = p2.wrapping_add(step_fp);
        pos_fp = p3.wrapping_add(step_fp);

        let m0 = sinc32_dot_avx2(
            left_ptr.add((p0 >> 32) as usize).sub(15),
            lut_base.add((((p0 >> 24) & 0xFF) as usize) * 32),
        );
        let m1 = sinc32_dot_avx2(
            left_ptr.add((p1 >> 32) as usize).sub(15),
            lut_base.add((((p1 >> 24) & 0xFF) as usize) * 32),
        );
        let m2 = sinc32_dot_avx2(
            left_ptr.add((p2 >> 32) as usize).sub(15),
            lut_base.add((((p2 >> 24) & 0xFF) as usize) * 32),
        );
        let m3 = sinc32_dot_avx2(
            left_ptr.add((p3 >> 32) as usize).sub(15),
            lut_base.add((((p3 >> 24) & 0xFF) as usize) * 32),
        );

        let mut samples4 = hsum4_avx2(m0, m1, m2, m3);
        if FILTER {
            samples4 = apply_lpf4_left_avx2(samples4, flt);
        }
        let env4 = env4_avx2(&mut env_mode);
        let scaled = _mm_mul_ps(samples4, env4);
        accumulate_stereo4_avx2(
            out_ptr.add(f * 2),
            _mm_mul_ps(scaled, gl_vec),
            _mm_mul_ps(scaled, gr_vec),
        );

        f += 4;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let mut s = hsum1_avx2(sinc32_dot_avx2(
            left_ptr.add(idx).sub(15),
            lut_base.add(ph * 32),
        ));
        if FILTER {
            s = flt.step_left(s);
        }
        let val = s * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += val * gain_l;
        *dst.add(1) += val * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    (pos_fp, current_env_level(&env_mode), f)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc32_stereo_avx2<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc32,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let lut_base = lut.rows.as_ptr() as *const f32;
    let gl_vec = _mm_set1_ps(gain_l);
    let gr_vec = _mm_set1_ps(gain_r);
    let mut f = 0usize;

    while f + 4 <= run {
        let p0 = pos_fp;
        let p1 = p0.wrapping_add(step_fp);
        let p2 = p1.wrapping_add(step_fp);
        let p3 = p2.wrapping_add(step_fp);
        pos_fp = p3.wrapping_add(step_fp);

        let i0 = (p0 >> 32) as usize;
        let i1 = (p1 >> 32) as usize;
        let i2 = (p2 >> 32) as usize;
        let i3 = (p3 >> 32) as usize;

        let kp0 = lut_base.add((((p0 >> 24) & 0xFF) as usize) * 32);
        let kp1 = lut_base.add((((p1 >> 24) & 0xFF) as usize) * 32);
        let kp2 = lut_base.add((((p2 >> 24) & 0xFF) as usize) * 32);
        let kp3 = lut_base.add((((p3 >> 24) & 0xFF) as usize) * 32);

        let mut sl4 = hsum4_avx2(
            sinc32_dot_avx2(left_ptr.add(i0).sub(15), kp0),
            sinc32_dot_avx2(left_ptr.add(i1).sub(15), kp1),
            sinc32_dot_avx2(left_ptr.add(i2).sub(15), kp2),
            sinc32_dot_avx2(left_ptr.add(i3).sub(15), kp3),
        );
        let mut sr4 = hsum4_avx2(
            sinc32_dot_avx2(right_ptr.add(i0).sub(15), kp0),
            sinc32_dot_avx2(right_ptr.add(i1).sub(15), kp1),
            sinc32_dot_avx2(right_ptr.add(i2).sub(15), kp2),
            sinc32_dot_avx2(right_ptr.add(i3).sub(15), kp3),
        );

        if FILTER {
            let (fl, fr) = apply_lpf4_stereo_avx2(sl4, sr4, flt);
            sl4 = fl;
            sr4 = fr;
        }
        let env4 = env4_avx2(&mut env_mode);
        accumulate_stereo4_avx2(
            out_ptr.add(f * 2),
            _mm_mul_ps(sl4, _mm_mul_ps(env4, gl_vec)),
            _mm_mul_ps(sr4, _mm_mul_ps(env4, gr_vec)),
        );

        f += 4;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let kp = lut_base.add(ph * 32);
        let mut sl = hsum1_avx2(sinc32_dot_avx2(left_ptr.add(idx).sub(15), kp));
        let mut sr = hsum1_avx2(sinc32_dot_avx2(right_ptr.add(idx).sub(15), kp));
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }

    (pos_fp, current_env_level(&env_mode), f)
}

// ----------------------------------------------------------------------------
// Portable Branchless Scalar Fast-Path Kernels (Linear, Sinc8, Sinc16, Sinc32)
// ----------------------------------------------------------------------------

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_linear_mono<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    const INV_FRAC32: f32 = 1.0 / 4_294_967_296.0;
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let frac = ((pos_fp & 0xFFFF_FFFF) as u32) as f32 * INV_FRAC32;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let s0 = *left_ptr.add(idx);
        let s1 = *left_ptr.add(idx + 1);
        let mut s = s0 + frac * (s1 - s0);
        if FILTER {
            s = flt.step_left(s);
        }
        let v = s * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += v * gain_l;
        *dst.add(1) += v * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_linear_stereo<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    const INV_FRAC32: f32 = 1.0 / 4_294_967_296.0;
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let frac = ((pos_fp & 0xFFFF_FFFF) as u32) as f32 * INV_FRAC32;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let l0 = *left_ptr.add(idx);
        let l1 = *left_ptr.add(idx + 1);
        let r0 = *right_ptr.add(idx);
        let r1 = *right_ptr.add(idx + 1);
        let mut sl = l0 + frac * (l1 - l0);
        let mut sr = r0 + frac * (r1 - r0);
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc8_mono_scalar<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc8,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = lut.rows.get_unchecked(ph);
        let sp = left_ptr.add(idx).sub(3);
        let mut acc = *sp * k[0]
            + *sp.add(1) * k[1]
            + *sp.add(2) * k[2]
            + *sp.add(3) * k[3]
            + *sp.add(4) * k[4]
            + *sp.add(5) * k[5]
            + *sp.add(6) * k[6]
            + *sp.add(7) * k[7];
        if FILTER {
            acc = flt.step_left(acc);
        }
        let v = acc * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += v * gain_l;
        *dst.add(1) += v * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc8_stereo_scalar<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc8,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = lut.rows.get_unchecked(ph);
        let lp = left_ptr.add(idx).sub(3);
        let rp = right_ptr.add(idx).sub(3);
        let mut sl = *lp * k[0]
            + *lp.add(1) * k[1]
            + *lp.add(2) * k[2]
            + *lp.add(3) * k[3]
            + *lp.add(4) * k[4]
            + *lp.add(5) * k[5]
            + *lp.add(6) * k[6]
            + *lp.add(7) * k[7];
        let mut sr = *rp * k[0]
            + *rp.add(1) * k[1]
            + *rp.add(2) * k[2]
            + *rp.add(3) * k[3]
            + *rp.add(4) * k[4]
            + *rp.add(5) * k[5]
            + *rp.add(6) * k[6]
            + *rp.add(7) * k[7];
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc16_mono_scalar<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc16,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = lut.rows.get_unchecked(ph);
        let sp = left_ptr.add(idx).sub(7);
        let mut acc = 0.0f32;
        for t in 0..16 {
            acc += *sp.add(t) * k[t];
        }
        if FILTER {
            acc = flt.step_left(acc);
        }
        let v = acc * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += v * gain_l;
        *dst.add(1) += v * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc16_stereo_scalar<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc16,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = lut.rows.get_unchecked(ph);
        let lp = left_ptr.add(idx).sub(7);
        let rp = right_ptr.add(idx).sub(7);
        let mut sl = 0.0f32;
        let mut sr = 0.0f32;
        for t in 0..16 {
            let kt = k[t];
            sl += *lp.add(t) * kt;
            sr += *rp.add(t) * kt;
        }
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc32_mono_scalar<const FILTER: bool>(
    left_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc32,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = lut.rows.get_unchecked(ph);
        let sp = left_ptr.add(idx).sub(15);
        let mut acc = 0.0f32;
        for t in 0..32 {
            acc += *sp.add(t) * k[t];
        }
        if FILTER {
            acc = flt.step_left(acc);
        }
        let v = acc * step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += v * gain_l;
        *dst.add(1) += v * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn render_fast_sinc32_stereo_scalar<const FILTER: bool>(
    left_ptr: *const f32,
    right_ptr: *const f32,
    mut pos_fp: u64,
    step_fp: u64,
    gain_l: f32,
    gain_r: f32,
    mut env_mode: EnvRunMode,
    flt: &mut VoiceFilter,
    lut: &AlignedSinc32,
    out_ptr: *mut f32,
    run: usize,
) -> (u64, f32, usize) {
    let mut f = 0usize;
    while f < run {
        let idx = (pos_fp >> 32) as usize;
        let ph = ((pos_fp >> 24) & 0xFF) as usize;
        pos_fp = pos_fp.wrapping_add(step_fp);

        let k = lut.rows.get_unchecked(ph);
        let lp = left_ptr.add(idx).sub(15);
        let rp = right_ptr.add(idx).sub(15);
        let mut sl = 0.0f32;
        let mut sr = 0.0f32;
        for t in 0..32 {
            let kt = k[t];
            sl += *lp.add(t) * kt;
            sr += *rp.add(t) * kt;
        }
        if FILTER {
            sl = flt.step_left(sl);
            sr = flt.step_right(sr);
        }
        let env = step_env_scalar(&mut env_mode);

        let dst = out_ptr.add(f * 2);
        *dst += sl * env * gain_l;
        *dst.add(1) += sr * env * gain_r;
        f += 1;
        if should_stop_env(&env_mode) {
            return (pos_fp, current_env_level(&env_mode), f);
        }
    }
    (pos_fp, current_env_level(&env_mode), f)
}

// ----------------------------------------------------------------------------
// Exact Loop-Seam Boundary Fallback (only executed for final 1..16 samples before wrap)
// ----------------------------------------------------------------------------

#[inline]
#[allow(clippy::too_many_arguments)]
fn render_loop_seam_sample(
    voice: &mut HotVoice,
    step_fp: u64,
    loop_start: usize,
    loop_end: usize,
    loop_end_fp: u64,
    loop_len_fp: u64,
    interp: InterpolationMode,
    tables: &SincTables,
    out_pair: &mut [f32],
) -> bool {
    let env = advance_envelope_hot(voice);
    if voice.env_stage == EnvStage::Inactive {
        return false;
    }
    if voice.env_stage == EnvStage::Delay {
        return true;
    }

    let sample_len = voice.sample_len as usize;
    let idx = (voice.pos_fp >> 32) as usize;
    let frac = ((voice.pos_fp & 0xFFFF_FFFF) as u32) as f32 * (1.0 / 4_294_967_296.0);
    let ph = ((voice.pos_fp >> 24) & 0xFF) as usize;

    let left_buf = unsafe { core::slice::from_raw_parts(voice.sample_left_ptr, sample_len) };
    let right_buf = if voice.sample_right_ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(voice.sample_right_ptr, sample_len) })
    };

    let (mut sl, mut sr) = match interp {
        InterpolationMode::Linear => {
            let l = interp_linear_wrap(left_buf, idx, frac, loop_start, loop_end);
            let r = match right_buf {
                Some(rb) => interp_linear_wrap(rb, idx, frac, loop_start, loop_end),
                None => l,
            };
            (l, r)
        }
        InterpolationMode::Sinc8 => {
            let k = &tables.sinc8.rows[ph];
            let l = interp_sinc_wrap(left_buf, idx, k, 3, loop_start, loop_end);
            let r = match right_buf {
                Some(rb) => interp_sinc_wrap(rb, idx, k, 3, loop_start, loop_end),
                None => l,
            };
            (l, r)
        }
        InterpolationMode::Sinc16 => {
            let k = &tables.sinc16.rows[ph];
            let l = interp_sinc_wrap(left_buf, idx, k, 7, loop_start, loop_end);
            let r = match right_buf {
                Some(rb) => interp_sinc_wrap(rb, idx, k, 7, loop_start, loop_end),
                None => l,
            };
            (l, r)
        }
        InterpolationMode::Sinc32 => {
            let k = &tables.sinc32.rows[ph];
            let l = interp_sinc_wrap(left_buf, idx, k, 15, loop_start, loop_end);
            let r = match right_buf {
                Some(rb) => interp_sinc_wrap(rb, idx, k, 15, loop_start, loop_end),
                None => l,
            };
            (l, r)
        }
    };

    if voice.filter_active {
        sl = voice.filter.step_left(sl);
        sr = if right_buf.is_some() {
            voice.filter.step_right(sr)
        } else {
            sl
        };
    }

    out_pair[0] += sl * env * voice.gain_l;
    out_pair[1] += sr * env * voice.gain_r;

    voice.pos_fp = voice.pos_fp.wrapping_add(step_fp);
    if voice.pos_fp >= loop_end_fp && loop_len_fp > 0 {
        voice.pos_fp =
            loop_end_fp - loop_len_fp + ((voice.pos_fp - loop_end_fp) % loop_len_fp);
    }

    true
}

#[inline]
fn advance_envelope_hot(voice: &mut HotVoice) -> f32 {
    match voice.env_stage {
        EnvStage::Inactive => 0.0,
        EnvStage::Delay => {
            if voice.env_samples_left > 0 {
                voice.env_samples_left -= 1;
                0.0
            } else {
                voice.env_stage = EnvStage::Attack;
                0.0
            }
        }
        EnvStage::Attack => {
            voice.env_level += voice.attack_inc;
            if voice.env_level >= 1.0 {
                voice.env_level = 1.0;
                if voice.pending_release {
                    voice.pending_release = false;
                    voice.env_stage = EnvStage::Release;
                } else if voice.hold_samples > 0 {
                    voice.env_samples_left = voice.hold_samples;
                    voice.env_stage = EnvStage::Hold;
                } else {
                    voice.env_stage = EnvStage::Decay;
                }
            }
            voice.env_level
        }
        EnvStage::Hold => {
            if voice.env_samples_left > 0 {
                voice.env_samples_left -= 1;
            } else {
                voice.env_stage = EnvStage::Decay;
            }
            voice.env_level
        }
        EnvStage::Decay => {
            voice.env_level *= voice.decay_mul;
            if voice.env_level <= voice.sustain_level {
                voice.env_level = voice.sustain_level;
                if voice.sustain_level <= MIN_ENV_AMP {
                    voice.env_level = 0.0;
                    voice.env_stage = EnvStage::Inactive;
                    return 0.0;
                }
                voice.env_stage = EnvStage::Sustain;
            }
            voice.env_level
        }
        EnvStage::Sustain => {
            if voice.env_level <= MIN_ENV_AMP && voice.sustain_level <= MIN_ENV_AMP {
                voice.env_stage = EnvStage::Inactive;
                0.0
            } else {
                voice.env_level
            }
        }
        EnvStage::Release => {
            voice.env_level *= voice.release_mul;
            if voice.env_level <= MIN_ENV_AMP {
                voice.env_level = 0.0;
                voice.env_stage = EnvStage::Inactive;
            }
            voice.env_level
        }
    }
}

#[inline]
fn sample_at_wrap(buf: &[f32], pos: isize, loop_start: usize, loop_end: usize) -> f32 {
    let len = buf.len() as isize;
    if len == 0 {
        return 0.0;
    }
    let ls = loop_start as isize;
    let le = loop_end as isize;
    let idx = if le > ls && pos >= le {
        (ls + (pos - ls) % (le - ls)) as usize
    } else {
        pos.clamp(0, len - 1) as usize
    };
    buf.get(idx).copied().unwrap_or(0.0)
}

#[inline]
fn interp_linear_wrap(
    buf: &[f32],
    idx: usize,
    frac: f32,
    loop_start: usize,
    loop_end: usize,
) -> f32 {
    let s0 = sample_at_wrap(buf, idx as isize, loop_start, loop_end);
    let s1 = sample_at_wrap(buf, idx as isize + 1, loop_start, loop_end);
    s0 + frac * (s1 - s0)
}

#[inline]
fn interp_sinc_wrap<const TAPS: usize>(
    buf: &[f32],
    idx: usize,
    k: &[f32; TAPS],
    left_radius: isize,
    loop_start: usize,
    loop_end: usize,
) -> f32 {
    let base = idx as isize - left_radius;
    let mut acc = 0.0f32;
    for tap in 0..TAPS {
        acc += sample_at_wrap(buf, base + tap as isize, loop_start, loop_end) * k[tap];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal valid 16-bit PCM WAV + SFZ fixture in temp dir and verify
    /// real synthesis across Linear, Sinc8, Sinc16, and Sinc32 + voice stealing.
    #[test]
    fn sfz_real_synthesis_all_interpolation_modes_and_voice_limit() {
        let tmp = std::env::temp_dir().join(format!("kdmapiv3_sfz_synth_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let wav_path = tmp.join("tone.wav");
        let sfz_path = tmp.join("piano.sfz");

        write_sine_wav(&wav_path, 44_100, 440.0, 22_050);
        std::fs::write(
            &sfz_path,
            "<region> sample=tone.wav lokey=0 hikey=127 pitch_keycenter=69 loop_mode=loop_continuous loop_start=100 loop_end=10000 ampeg_attack=0.001 ampeg_release=0.01\n",
        )
        .expect("write sfz");

        for mode in [
            InterpolationMode::Linear,
            InterpolationMode::Sinc8,
            InterpolationMode::Sinc16,
            InterpolationMode::Sinc32,
        ] {
            let mut synth = SoundFontSynth::new();
            let cfg = SynthConfig {
                sample_rate: 48_000,
                max_voices: 4,
                buffer_frames: 512,
                interpolation: mode,
                ..SynthConfig::default()
            };
            synth.init(cfg).expect("init synth");
            synth
                .load_soundfonts(std::slice::from_ref(&sfz_path))
                .expect("load sfz");
            assert_eq!(synth.loaded_preset_count(), 1);

            for note in 60..66u32 {
                synth.process_short_event(0x007F_0090 | (note << 8));
            }
            assert_eq!(synth.stats().active_voices, 4);

            let mut stereo = [0.0f32; 1024];
            synth.render_audio(&mut stereo);

            let peak = stereo.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            assert!(peak > 0.05, "mode {:?} produced silence (peak={})", mode, peak);
            assert!(stereo.iter().all(|s| s.is_finite()));

            synth.process_short_event(0x0000_7BB0);
            for _ in 0..32 {
                synth.render_audio(&mut stereo);
            }
            assert_eq!(synth.stats().active_voices, 0);
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Verify AVX2/FMA SIMD + multi-threaded rendering matches 1T scalar reference within 1e-4
    /// across Sinc8, Sinc16, and Sinc32.
    #[test]
    fn avx2_and_parallel_match_scalar_reference() {
        let tmp = std::env::temp_dir().join(format!("kdmapiv3_simd_eq_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("create temp dir");

        let wav_path = tmp.join("tone.wav");
        let sfz_path = tmp.join("piano.sfz");
        write_sine_wav(&wav_path, 44_100, 440.0, 22_050);
        std::fs::write(
            &sfz_path,
            "<region> sample=tone.wav lokey=0 hikey=127 pitch_keycenter=69 loop_mode=loop_continuous loop_start=100 loop_end=10000 ampeg_attack=0.002 ampeg_release=0.02\n",
        )
        .expect("write sfz");

        for mode in [
            InterpolationMode::Sinc8,
            InterpolationMode::Sinc16,
            InterpolationMode::Sinc32,
        ] {
            let mut ref_synth = SoundFontSynth::new();
            let mut fast_synth = SoundFontSynth::new();
            let cfg = SynthConfig {
                sample_rate: 48_000,
                max_voices: 256,
                buffer_frames: 512,
                interpolation: mode,
                ..SynthConfig::default()
            };
            ref_synth.init(cfg).expect("init ref");
            fast_synth.init(cfg).expect("init fast");
            ref_synth
                .load_soundfonts(std::slice::from_ref(&sfz_path))
                .expect("load ref");
            fast_synth
                .load_soundfonts(std::slice::from_ref(&sfz_path))
                .expect("load fast");

            ref_synth.set_force_scalar(true);
            ref_synth.set_parallel_enabled(false);
            fast_synth.set_force_scalar(false);
            fast_synth.set_parallel_enabled(true);

            for i in 0..160u32 {
                let note = 36 + (i % 60);
                let ch = i & 0x0F;
                let msg = 0x0060_0090 | (note << 8) | ch;
                ref_synth.process_short_event(msg);
                fast_synth.process_short_event(msg);
            }

            let mut ref_out = [0.0f32; 1024];
            let mut fast_out = [0.0f32; 1024];
            for _ in 0..4 {
                ref_synth.render_audio(&mut ref_out);
                fast_synth.render_audio(&mut fast_out);
                let mut max_err = 0.0f32;
                for (a, b) in ref_out.iter().zip(fast_out.iter()) {
                    max_err = max_err.max((a - b).abs());
                }
                assert!(
                    max_err < 1.0e-4,
                    "mode {:?}: AVX2+MT diverged from scalar+1T (max_err={})",
                    mode,
                    max_err
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Verify SF2 2.01 concave attenuation LUT, bypassable resonant LPF vs active LPF,
    /// exclusiveClass choke groups, and CC66 Sostenuto pedal behavior.
    #[test]
    fn sf2_concave_luts_lpf_bypass_and_exclusive_class_choke() {
        // 1. Concave LUT invariants
        let lut = build_sf2_concave_lut();
        assert_eq!(lut[0], 0.0);
        assert_eq!(lut[127], 1.0);
        // Midpoint 64/127 squared ~= 0.254 (-11.9 dB concave curve, NOT 0.5 linear)
        assert!((lut[64] - (64.0f32 / 127.0).powi(2)).abs() < 1.0e-6);
        assert!(lut[64] < 0.30);

        // Build an in-memory high-frequency square/saw wave (rich harmonics for LPF test)
        let sr = 48_000u32;
        let mut raw = vec![0.0f32; 4800];
        for (i, s) in raw.iter_mut().enumerate() {
            // 4 kHz square wave -> strong high-frequency content that a 400 Hz LPF heavily attenuates
            *s = if ((i * 4_000) / (sr as usize)) % 2 == 0 {
                0.8
            } else {
                -0.8
            };
        }
        let sample_asset = SampleAsset::from_raw(&raw, sr);

        let make_region = |fc_cents: f32, q_cb: f32, excl: u8, key_lo: u8, key_hi: u8| SynthRegion {
            sample_idx: 0,
            right_sample_idx: NO_SAMPLE,
            key_low: key_lo,
            key_high: key_hi,
            vel_low: 0,
            vel_high: 127,
            root_key: 60,
            exclusive_class: excl,
            loop_mode: RegionLoopMode::LoopContinuous,
            loop_start: 64,
            loop_end: 4000,
            offset: 0,
            gain: 1.0,
            pan: 0.0,
            tune_cents: 0.0,
            scale_tuning: 100.0,
            env_delay_tc: -12_000.0,
            env_attack_tc: secs_to_timecents(0.001),
            env_hold_tc: -12_000.0,
            env_decay_tc: secs_to_timecents(2.0),
            env_release_tc: secs_to_timecents(0.5),
            key_to_hold_tc: 0.0,
            key_to_decay_tc: 0.0,
            env_sustain: 1.0,
            filter_fc_cents: fc_cents,
            filter_q_cb: q_cb,
            vel_to_filter_cents: 0.0,
            vib_delay_secs: 0.0,
            vib_freq_hz: 8.176,
            vib_to_pitch_cents: 0.0,
        };

        // 2. LPF Bypass (13500 cents, Q=0) vs Active LPF (~400 Hz = ~6735 cents)
        let mut synth = SoundFontSynth::new();
        synth
            .init(SynthConfig {
                sample_rate: sr,
                max_voices: 32,
                buffer_frames: 512,
                interpolation: InterpolationMode::Sinc8,
                ..SynthConfig::default()
            })
            .expect("init");
        synth.bank = SoundFontBank::from_parts(
            Box::new([sample_asset]),
            Box::new([
                make_region(13_500.0, 0.0, 0, 60, 60), // key 60: bypassed filter
                make_region(6_735.0, 0.0, 0, 61, 61),  // key 61: ~400 Hz active LPF
                make_region(13_500.0, 0.0, 1, 42, 46), // keys 42..46: hi-hat exclusiveClass = 1
            ]),
            Box::new([PresetEntry {
                bank: 0,
                preset: 0,
                region_start: 0,
                region_count: 3,
                key_bucket_base: 0,
            }]),
            synth.config.sample_rate,
        );

        // Key 60 (bypassed filter) -> filter_active MUST be false
        synth.process_short_event(0x007F_3C90);
        let slot_bypass = synth.active_list[0] as usize;
        assert!(!synth.hot_voices[slot_bypass].filter_active);
        let mut out_bypass = [0.0f32; 1024];
        synth.render_audio(&mut out_bypass);
        let rms_bypass =
            (out_bypass.iter().map(|s| s * s).sum::<f32>() / out_bypass.len() as f32).sqrt();

        // Key 61 (~400 Hz active LPF) -> filter_active MUST be true and heavily attenuate 4 kHz wave
        synth.reset();
        synth.process_short_event(0x007F_3D90);
        let slot_lpf = synth.active_list[0] as usize;
        assert!(synth.hot_voices[slot_lpf].filter_active);
        let mut out_lpf = [0.0f32; 1024];
        synth.render_audio(&mut out_lpf);
        let rms_lpf = (out_lpf.iter().map(|s| s * s).sum::<f32>() / out_lpf.len() as f32).sqrt();
        assert!(
            rms_lpf < rms_bypass * 0.25,
            "active 400 Hz LPF should strongly attenuate 4 kHz signal (bypass={:.4}, lpf={:.4})",
            rms_bypass,
            rms_lpf
        );

        // 3. ExclusiveClass choke: open hi-hat (key 46) choked immediately by closed hi-hat (key 42)
        synth.reset();
        synth.process_short_event(0x007F_2E90); // key 46 Note On (excl=1)
        let mut buf = [0.0f32; 1024];
        synth.render_audio(&mut buf);
        assert_eq!(synth.stats().active_voices, 1);
        let open_hh_slot = synth.active_list[0] as usize;
        assert_ne!(synth.hot_voices[open_hh_slot].env_stage, EnvStage::Release);

        synth.process_short_event(0x007F_2A90); // key 42 Note On (excl=1) -> MUST choke key 46
        assert_eq!(synth.hot_voices[open_hh_slot].env_stage, EnvStage::Release);
        for _ in 0..4 {
            synth.render_audio(&mut buf);
        }
        // Choked open hi-hat voice is culled cleanly; only closed hi-hat remains active.
        assert_eq!(synth.stats().active_voices, 1);

        // 4. CC66 Sostenuto holds pre-existing note across Note Off, but not newly pressed note
        synth.reset();
        synth.process_short_event(0x007F_3C90); // key 60 held before Sostenuto
        synth.render_audio(&mut buf);          // render 1 block so voice is established
        synth.process_short_event(0x007F_42B0); // CC66 Sostenuto ON
        synth.process_short_event(0x0000_3C80); // key 60 Note Off -> stays held!
        let s60 = synth.active_list[0] as usize;
        assert_ne!(synth.hot_voices[s60].env_stage, EnvStage::Release);
        synth.process_short_event(0x0000_42B0); // CC66 Sostenuto OFF -> now releases!
        assert_eq!(synth.hot_voices[s60].env_stage, EnvStage::Release);

        // 5. Low-Spec Effect Toggles: `disable_all_effects` & `enable_lowpass_filter = false`
        //    must completely bypass the 400 Hz LPF and Vibrato LFO at Note On (0% overhead).
        synth
            .init(SynthConfig {
                sample_rate: sr,
                max_voices: 32,
                buffer_frames: 512,
                render_threads: 4,
                interpolation: InterpolationMode::Sinc8,
                enable_lowpass_filter: true,
                enable_vibrato_lfo: true,
                disable_all_effects: true,
            })
            .expect("init low-spec override");
        synth.process_short_event(0x007F_01B0); // CC1 Mod Wheel = 127
        synth.process_short_event(0x007F_3D90); // key 61 (400 Hz LPF zone)
        let s_disabled = synth.active_list[0] as usize;
        assert!(
            !synth.hot_voices[s_disabled].filter_active,
            "disable_all_effects must force LPF bypass"
        );
        assert_eq!(synth.hot_voices[s_disabled].vib_to_pitch_cents, 0.0);
        let mut out_disabled = [0.0f32; 1024];
        synth.render_audio(&mut out_disabled);
        let rms_disabled =
            (out_disabled.iter().map(|s| s * s).sum::<f32>() / out_disabled.len() as f32).sqrt();
        assert!(
            rms_disabled > rms_lpf * 3.0,
            "bypassed LPF via disable_all_effects should preserve full 4 kHz energy"
        );
    }

    /// Verify master NaN/Inf guard + tanh soft-clipper, same-block Note-On+Note-Off survival
    /// (at least 1 block rendered instead of instant death), and unrendered voice steal protection.
    #[test]
    fn nan_guard_same_block_survival_and_unrendered_steal_protection() {
        // 1. Master NaN/Inf sanitizer + tanh soft-clipper strictly bounded in [-1.0, 1.0].
        let mut samples = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.5,
            -0.5,
            2.5,
            -2.5,
            1_000.0,
            -1_000.0,
        ];
        sanitize_and_soft_clip(&mut samples);
        assert_eq!(samples[0], 0.0);
        assert_eq!(samples[1], 0.0);
        assert_eq!(samples[2], 0.0);
        assert_eq!(samples[4], 0.5);
        assert_eq!(samples[5], -0.5);
        for &s in &samples {
            assert!(s.is_finite(), "output must never contain NaN/Inf");
            assert!((-1.0..=1.0).contains(&s), "output {} out of [-1, 1]", s);
        }
        assert!(samples[6] > 0.75 && samples[6] <= 1.0);
        assert!(samples[7] < -0.75 && samples[7] >= -1.0);

        // 2. Same-block Note-On + immediate Note-Off MUST survive & render audible attack+release
        //    in a 1024-frame block, then cleanly decay back to 0 active voices.
        let sr = 48_000u32;
        let raw: Vec<f32> = (0..8192)
            .map(|i| (2.0 * core::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.8)
            .collect();
        let make_bank = || SoundFontBank::from_parts(
            Box::new([SampleAsset::from_raw(&raw, sr)]),
            Box::new([SynthRegion {
                sample_idx: 0,
                right_sample_idx: NO_SAMPLE,
                key_low: 0,
                key_high: 127,
                vel_low: 0,
                vel_high: 127,
                root_key: 69,
                exclusive_class: 0,
                loop_mode: RegionLoopMode::LoopContinuous,
                loop_start: 64,
                loop_end: 4096,
                offset: 0,
                gain: 1.0,
                pan: 0.0,
                tune_cents: 0.0,
                scale_tuning: 100.0,
                env_delay_tc: -12_000.0,
                env_attack_tc: secs_to_timecents(0.002),
                env_hold_tc: -12_000.0,
                env_decay_tc: secs_to_timecents(1.0),
                env_release_tc: secs_to_timecents(0.03),
                key_to_hold_tc: 0.0,
                key_to_decay_tc: 0.0,
                env_sustain: 1.0,
                filter_fc_cents: 13_500.0,
                filter_q_cb: 0.0,
                vel_to_filter_cents: 0.0,
                vib_delay_secs: 0.0,
                vib_freq_hz: 8.176,
                vib_to_pitch_cents: 0.0,
            }]),
            Box::new([PresetEntry {
                bank: 0,
                preset: 0,
                region_start: 0,
                region_count: 1,
                key_bucket_base: 0,
            }]),
            sr,
        );

        let mut synth = SoundFontSynth::new();
        synth
            .init(SynthConfig {
                sample_rate: sr,
                max_voices: 4,
                buffer_frames: 1024,
                interpolation: InterpolationMode::Sinc8,
                ..SynthConfig::default()
            })
            .expect("init");
        synth.bank = make_bank();

        // Trigger Note-On and immediate Note-Off within the SAME 1024-frame block!
        synth.process_short_event(0x007F_4590); // key 69 Note-On
        synth.process_short_event(0x0000_4580); // key 69 Note-Off (same block!)
        let mut block1024 = [0.0f32; 2048];
        synth.render_audio(&mut block1024);

        let peak = block1024.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            peak > 0.05,
            "same-block Note-On+Note-Off must render audible attack/release (peak={})",
            peak
        );
        // Should decay naturally back to 0 voices over subsequent blocks (not stuck).
        for _ in 0..16 {
            synth.render_audio(&mut block1024);
        }
        assert_eq!(synth.stats().active_voices, 0);

        // 3. Unrendered voice steal protection:
        //    Fill 2 slots with rendered voices, then 2 slots with brand-new unrendered voices.
        //    A 5th Note-On MUST steal one of the rendered voices and NEVER steal an unrendered one.
        synth.reset();
        synth.process_short_event(0x007F_3C90); // key 60 (will render)
        synth.process_short_event(0x007F_3E90); // key 62 (will render)
        synth.render_audio(&mut block1024);
        assert_eq!(synth.stats().active_voices, 2);

        synth.process_short_event(0x007F_4090); // key 64 (unrendered!)
        synth.process_short_event(0x007F_4190); // key 65 (unrendered!)
        assert_eq!(synth.stats().active_voices, 4);

        // 5th Note-On (key 67): pool full -> steals a rendered voice (60 or 62), keeping 64 & 65 intact!
        synth.process_short_event(0x007F_4390);
        let active_keys: Vec<u8> = (0..synth.active_len)
            .map(|i| synth.cold_voices[synth.active_list[i] as usize].key)
            .collect();
        assert!(active_keys.contains(&64), "unrendered key 64 must not be stolen");
        assert!(active_keys.contains(&65), "unrendered key 65 must not be stolen");
        assert!(active_keys.contains(&67), "new key 67 should be spawned by stealing rendered voice");

        // 6th & 7th Note-On before render:
        // - 6th (key 69) steals the remaining rendered voice.
        // - 7th (key 71) MUST NOT be dropped; with all 4 voices unrendered at equal velocity,
        //   newest-note-priority voice stealing replaces the oldest voice (key 64).
        synth.process_short_event(0x007F_4590); // key 69 steals remaining rendered voice
        synth.process_short_event(0x007F_4790); // key 71 steals oldest unrendered voice (key 64)
        let active_keys2: Vec<u8> = (0..synth.active_len)
            .map(|i| synth.cold_voices[synth.active_list[i] as usize].key)
            .collect();
        assert!(!active_keys2.contains(&64), "oldest unrendered key 64 should be stolen");
        assert!(active_keys2.contains(&65));
        assert!(active_keys2.contains(&67));
        assert!(active_keys2.contains(&69));
        assert!(active_keys2.contains(&71), "newest key 71 must NEVER be dropped");
    }

    /// Verify O(1) newest-note-priority voice stealing (no block-level early-drop),
    /// low-velocity survival independent of `master_gain`, sample-rate-scaled choke/kill decay,
    /// and 2048/4096-frame buffer rendering.
    #[test]
    fn o1_voice_stealing_newest_note_priority_and_large_buffers() {
        let sr = 48_000u32;
        let raw: Vec<f32> = (0..8192)
            .map(|i| (2.0 * core::f32::consts::PI * 440.0 * (i as f32) / (sr as f32)).sin() * 0.6)
            .collect();
        let make_bank = || SoundFontBank::from_parts(
            Box::new([SampleAsset::from_raw(&raw, sr)]),
            Box::new([SynthRegion {
                sample_idx: 0,
                right_sample_idx: NO_SAMPLE,
                key_low: 0,
                key_high: 127,
                vel_low: 0,
                vel_high: 127,
                root_key: 69,
                exclusive_class: 0,
                loop_mode: RegionLoopMode::LoopContinuous,
                loop_start: 64,
                loop_end: 4096,
                offset: 0,
                gain: 1.0,
                pan: 0.0,
                tune_cents: 0.0,
                scale_tuning: 100.0,
                env_delay_tc: -12_000.0,
                env_attack_tc: secs_to_timecents(0.002),
                env_hold_tc: -12_000.0,
                env_decay_tc: secs_to_timecents(1.0),
                env_release_tc: secs_to_timecents(0.05),
                key_to_hold_tc: 0.0,
                key_to_decay_tc: 0.0,
                env_sustain: 1.0,
                filter_fc_cents: 13_500.0,
                filter_q_cb: 0.0,
                vel_to_filter_cents: 0.0,
                vib_delay_secs: 0.0,
                vib_freq_hz: 8.176,
                vib_to_pitch_cents: 0.0,
            }]),
            Box::new([PresetEntry {
                bank: 0,
                preset: 0,
                region_start: 0,
                region_count: 1,
                key_bucket_base: 0,
            }]),
            sr,
        );

        let mut synth = SoundFontSynth::new();
        synth
            .init(SynthConfig {
                sample_rate: sr,
                max_voices: 64,
                buffer_frames: 2048,
                interpolation: InterpolationMode::Linear,
                ..SynthConfig::default()
            })
            .expect("init 2048");
        synth.bank = make_bank();
        if let Some(ref pool) = synth.render_pool {
            assert!(pool.max_frames >= 2048);
        }

        // 1. Burst 5,000 Note-Ons in one block before rendering:
        //    First 64 allocate from free-list, subsequent notes legitimately steal older voices
        //    so the newest Note-Ons survive instead of being dropped.
        for i in 0..5_000u32 {
            let key = (i % 128) as u32;
            synth.process_short_event(0x007F_0090 | (key << 8));
        }
        assert_eq!(synth.active_len, 64);
        assert_eq!(synth.rendered_count, 0);
        let final_key = (4_999u32 % 128) as u8;
        let has_final_key = (0..synth.active_len)
            .any(|i| synth.cold_voices[synth.active_list[i] as usize].key == final_key);
        assert!(
            has_final_key,
            "newest Note-On (key {}) must survive via legitimate voice stealing",
            final_key
        );

        // Render one 2048-frame block -> all 64 voices transition into the stealable rendered prefix.
        let mut out2048 = vec![0.0f32; 2048 * 2];
        synth.render_audio(&mut out2048);
        assert_eq!(synth.active_len, 64);
        assert_eq!(synth.rendered_count, 64);
        assert!(out2048.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        assert!(out2048.iter().any(|&s| s.abs() > 0.01));

        // 2. Now spawn 32 new notes -> steals 32 rendered slots in O(1) via round-robin cursor.
        for i in 0..32u32 {
            let key = (60 + (i % 24)) as u32;
            synth.process_short_event(0x007F_0090 | (key << 8));
        }
        assert_eq!(synth.active_len, 64);
        assert_eq!(synth.rendered_count, 32);
        for idx in 0..synth.rendered_count {
            let slot = synth.active_list[idx] as usize;
            assert!(synth.hot_voices[slot].has_rendered);
        }
        for idx in synth.rendered_count..synth.active_len {
            let slot = synth.active_list[idx] as usize;
            assert!(!synth.hot_voices[slot].has_rendered);
        }

        // 3. Low-velocity survival: vel=1 at CC7=64 and CC7=85 MUST NOT be dropped by master_gain=0.35.
        synth.reset();
        synth.process_short_event(0x0040_07B0); // CC7 Volume = 64
        synth.process_short_event(0x0001_3C90); // vel=1, key=60
        assert_eq!(
            synth.stats().active_voices,
            1,
            "vel=1 at CC7=64 must not be prematurely culled by master_gain"
        );
        // Muted channel (CC7=0) is still cleanly skipped.
        synth.reset();
        synth.process_short_event(0x0000_07B0); // CC7 Volume = 0
        synth.process_short_event(0x007F_3C90);
        assert_eq!(synth.stats().active_voices, 0);

        // 4. Sample-rate-scaled choke/kill release multipliers: higher sample rate -> closer to 1.0
        //    so physical decay duration stays constant across sample rates.
        let mul_48k = exp_decay_mul_for_secs(CHOKE_RELEASE_SECS, 48_000);
        let mul_192k = exp_decay_mul_for_secs(CHOKE_RELEASE_SECS, 192_000);
        assert!(mul_192k > mul_48k);
        // After 2ms of samples at either sample rate, amplitude reaches ~MIN_ENV_AMP (1e-5).
        let amp_48k = mul_48k.powf(CHOKE_RELEASE_SECS * 48_000.0);
        let amp_192k = mul_192k.powf(CHOKE_RELEASE_SECS * 192_000.0);
        assert!((amp_48k - MIN_ENV_AMP).abs() < 2.0e-6);
        assert!((amp_192k - MIN_ENV_AMP).abs() < 2.0e-6);

        // 5. Re-init with 4096-frame buffer and confirm RenderPool resizes and renders cleanly.
        synth
            .init(SynthConfig {
                sample_rate: sr,
                max_voices: 128,
                buffer_frames: 4096,
                interpolation: InterpolationMode::Sinc8,
                ..SynthConfig::default()
            })
            .expect("init 4096");
        if let Some(ref pool) = synth.render_pool {
            assert!(pool.max_frames >= 4096);
        }
        for i in 0..96u32 {
            let key = (36 + (i % 60)) as u32;
            synth.process_short_event(0x007F_0090 | (key << 8));
        }
        let mut out4096 = vec![0.0f32; 4096 * 2];
        synth.render_audio(&mut out4096);
        assert_eq!(synth.active_len, 96);
        assert_eq!(synth.rendered_count, 96);
        assert!(out4096.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        assert!(out4096.iter().any(|&s| s.abs() > 0.01));
    }

    /// Verify manual `render_threads` (including 1T and >16T) and `max_voices > 100_000`
    /// are honored strictly without hidden auto-scaling or silent 100k/15-worker caps.
    #[test]
    fn manual_render_threads_and_voice_limit_honored_without_hidden_caps() {
        let mut synth = SoundFontSynth::new();

        // 1 thread -> strictly single-threaded, no worker pool spawned.
        synth
            .init(SynthConfig {
                max_voices: 150_000,
                buffer_frames: 192,
                render_threads: 1,
                ..SynthConfig::default()
            })
            .expect("init 1T 150k");
        assert_eq!(synth.hot_voices.len(), 150_000);
        assert_eq!(synth.render_thread_count(), 1);
        assert!(synth.render_pool.is_none());

        // 20 threads -> strictly 1 audio thread + 19 workers (no hidden 15-worker cap).
        synth
            .init(SynthConfig {
                max_voices: 64,
                buffer_frames: 64,
                render_threads: 20,
                ..SynthConfig::default()
            })
            .expect("init 20T");
        assert_eq!(synth.render_thread_count(), 20);
        assert_eq!(
            synth.render_pool.as_ref().map(|p| p.max_frames),
            Some(64),
            "worker scratch must honor exact buffer_frames without hidden upscaling"
        );
    }

    fn write_sine_wav(path: &Path, sr: u32, freq: f32, frames: usize) {
        let data_bytes = (frames * 2) as u32;
        let riff_size = 36 + data_bytes;
        let mut f = File::create(path).expect("create wav");
        f.write_all(b"RIFF").expect("wav header");
        f.write_all(&riff_size.to_le_bytes()).expect("riff size");
        f.write_all(b"WAVEfmt ").expect("fmt chunk");
        f.write_all(&16u32.to_le_bytes()).expect("fmt len");
        f.write_all(&1u16.to_le_bytes()).expect("pcm format");
        f.write_all(&1u16.to_le_bytes()).expect("mono");
        f.write_all(&sr.to_le_bytes()).expect("sample rate");
        f.write_all(&(sr * 2).to_le_bytes()).expect("byte rate");
        f.write_all(&2u16.to_le_bytes()).expect("block align");
        f.write_all(&16u16.to_le_bytes()).expect("bits per sample");
        f.write_all(b"data").expect("data tag");
        f.write_all(&data_bytes.to_le_bytes()).expect("data size");
        for i in 0..frames {
            let t = i as f32 / sr as f32;
            let sample = (2.0 * core::f32::consts::PI * freq * t).sin();
            let s16 = (sample * 28_000.0) as i16;
            f.write_all(&s16.to_le_bytes()).expect("sample");
        }
    }
}

#[cfg(test)]
mod dsp_hotpath_fix_tests {
    use super::*;

    #[test]
    fn filter_trig_lut_matches_exact_from_fc_q() {
        let sr = 48_000u32;
        let lut = build_filter_trig_lut(sr);
        for &q_cb in &[0.0f32, 30.0, 96.0] {
            let q_db = (q_cb / 10.0).clamp(0.0, 96.0) as f64;
            let q_lin = 10.0f64.powf(q_db / 20.0) * core::f64::consts::FRAC_1_SQRT_2;
            let inv_2q = (0.5 / q_lin) as f32;

            for step in 0..=240 {
                let eff_fc_cents = 1_500.0 + (step as f32) * 50.0;
                let fc_hz = 8.175_798_915_6 * 2.0f64.powf(eff_fc_cents as f64 / 1_200.0);
                let exact = VoiceFilter::from_fc_q(fc_hz, q_cb, sr as f64);

                let t = (eff_fc_cents - 1_500.0) * (FILTER_TRIG_LUT_STEPS as f32 / 12_000.0);
                let idx = (t as usize).min(FILTER_TRIG_LUT_STEPS - 1);
                let frac = t - (idx as f32);
                let (c0, s0) = lut[idx];
                let (c1, s1) = lut[idx + 1];
                let approx =
                    VoiceFilter::from_cos_sin_inv2q(c0 + (c1 - c0) * frac, s0 + (s1 - s0) * frac, inv_2q);

                assert!((exact.b0 - approx.b0).abs() < 1.0e-3, "b0 mismatch at {eff_fc_cents}c");
                assert!((exact.a1 - approx.a1).abs() < 1.0e-3, "a1 mismatch at {eff_fc_cents}c");
                assert!((exact.a2 - approx.a2).abs() < 1.0e-3, "a2 mismatch at {eff_fc_cents}c");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn stereo_biquad_avx2_matches_scalar_and_ftz_daz_active() {
        enable_ftz_daz();
        let tiny = core::hint::black_box(f32::from_bits(1));
        assert_eq!(core::hint::black_box(tiny * 0.5f32), 0.0);

        if !detect_avx2_fma() {
            return;
        }
        let mut flt_scalar = VoiceFilter::from_fc_q(1_800.0, 45.0, 48_000.0);
        let mut flt_simd = flt_scalar;

        for blk in 0..32usize {
            let mut l_in = [0.0f32; 4];
            let mut r_in = [0.0f32; 4];
            for k in 0..4 {
                let n = (blk * 4 + k) as f32;
                l_in[k] = (n * 0.17).sin() * 0.8;
                r_in[k] = (n * 0.31 + 0.7).cos() * 0.6;
            }
            let mut l_ref = [0.0f32; 4];
            let mut r_ref = [0.0f32; 4];
            for k in 0..4 {
                l_ref[k] = flt_scalar.step_left(l_in[k]);
                r_ref[k] = flt_scalar.step_right(r_in[k]);
            }
            let (mut l_got, mut r_got) = ([0.0f32; 4], [0.0f32; 4]);
            unsafe {
                use core::arch::x86_64::{_mm_set_ps, _mm_storeu_ps};
                let sl4 = _mm_set_ps(l_in[3], l_in[2], l_in[1], l_in[0]);
                let sr4 = _mm_set_ps(r_in[3], r_in[2], r_in[1], r_in[0]);
                let (ol, or) = apply_lpf4_stereo_avx2(sl4, sr4, &mut flt_simd);
                _mm_storeu_ps(l_got.as_mut_ptr(), ol);
                _mm_storeu_ps(r_got.as_mut_ptr(), or);
            }
            for k in 0..4 {
                assert!((l_ref[k] - l_got[k]).abs() < 1.0e-5, "L mismatch blk={blk} k={k}");
                assert!((r_ref[k] - r_got[k]).abs() < 1.0e-5, "R mismatch blk={blk} k={k}");
            }
        }
        assert!((flt_scalar.l_z1 - flt_simd.l_z1).abs() < 1.0e-5);
        assert!((flt_scalar.l_z2 - flt_simd.l_z2).abs() < 1.0e-5);
        assert!((flt_scalar.r_z1 - flt_simd.r_z1).abs() < 1.0e-5);
        assert!((flt_scalar.r_z2 - flt_simd.r_z2).abs() < 1.0e-5);
    }
}
