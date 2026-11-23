//! Cross-platform real-time audio output (`cpal`) with lock-free MIDI ingestion
//! directly inside the audio render callback and headless fallback support.
//!
//! - Linux: ALSA (routes transparently through PipeWire-ALSA or native ALSA).
//! - Windows / Wine: WASAPI via `cpal`.
//! - Zero heap allocations, zero mutexes, zero logging inside the audio callback.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Sample, SampleFormat, Stream, StreamConfig};

use crate::backend::{SynthBackend, SynthConfig, MAX_BUFFER_FRAMES, MIN_BUFFER_FRAMES};
use crate::ringbuf::{
    EventKind, LockFreeRingBuffer, MidiEvent, SysExPacket, TelemetryCounters,
    DEFAULT_SYSEX_RING_CAPACITY,
};

/// Optional env var to force headless consumer mode (useful for deterministic CI/benchmarks).
pub const NO_AUDIO_ENV: &str = "KDMAPIV3_NO_AUDIO";

/// Check the drain wall-clock budget once every 64 batches (16,384 events; power of two)
/// to avoid timer overhead on normal songs while guaranteeing extreme bursts never
/// starve DSP rendering into XRUNs. Remaining queued events stay untouched in the ring
/// buffer for the next callback (strictly zero note skipping).
const DRAIN_BUDGET_CHECK_MASK: usize = 64 - 1;

/// Maximum share of an audio callback period allocated to draining the MIDI ring buffer
/// before yielding to SIMD voice synthesis (35% drain / 65% DSP headroom).
const MAX_DRAIN_BUDGET_FRACTION: f32 = 0.35;

/// Heap-allocated lock-free shared state observed by both producers and the real-time audio callback.
///
/// Owned via `Arc<RtEngineShared>` so both the global FFI `OnceLock` runtime and owned in-process
/// Rust engine instances (`DirectKdmapiEngine`) share the exact same zero-lock, zero-leak hot path.
pub struct RtEngineShared {
    pub short_ring: LockFreeRingBuffer<MidiEvent>,
    pub sysex_ring: LockFreeRingBuffer<SysExPacket>,
    pub telemetry: TelemetryCounters,
    pub running: AtomicBool,
    pub stop_requested: AtomicBool,
    pub reset_pending: AtomicBool,
    pub voice_count: AtomicU64,
    pub rendering_time_bits: AtomicU32,
}

impl RtEngineShared {
    pub fn new(short_ring_capacity: usize) -> Self {
        Self {
            short_ring: LockFreeRingBuffer::with_capacity(short_ring_capacity),
            sysex_ring: LockFreeRingBuffer::with_capacity(DEFAULT_SYSEX_RING_CAPACITY),
            telemetry: TelemetryCounters::new(),
            running: AtomicBool::new(false),
            stop_requested: AtomicBool::new(false),
            reset_pending: AtomicBool::new(false),
            voice_count: AtomicU64::new(0),
            rendering_time_bits: AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

/// Clonable lock-free handle bundle passed into the real-time audio callback (cloned on cold path only).
#[derive(Clone)]
pub struct RtEngineHooks {
    pub shared: Arc<RtEngineShared>,
}

impl RtEngineHooks {
    #[inline]
    pub fn new(shared: Arc<RtEngineShared>) -> Self {
        Self { shared }
    }
}

impl Deref for RtEngineHooks {
    type Target = RtEngineShared;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

/// Active audio output driver — either a live `cpal` real-time audio stream owned by a
/// dedicated COM/thread-affine owner thread (safe across multi-threaded init/terminate
/// on Windows/Wine WASAPI and Linux ALSA), or a headless fallback consumer thread.
pub enum AudioDriver {
    CpalOwned {
        stop_tx: mpsc::SyncSender<()>,
        owner_handle: JoinHandle<Option<Box<dyn SynthBackend>>>,
    },
    HeadlessThread {
        handle: JoinHandle<Box<dyn SynthBackend>>,
    },
}

/// Lock-free single-slot ownership handoff for returning `Box<dyn SynthBackend>` from
/// `BackendCallbackOwner::drop` without ever touching a `Mutex` on an audio thread.
struct LockFreeBackendSlot {
    ptr: AtomicPtr<Box<dyn SynthBackend>>,
}

impl LockFreeBackendSlot {
    fn new() -> Self {
        Self {
            ptr: AtomicPtr::new(core::ptr::null_mut()),
        }
    }

    /// Wait-free store invoked from `BackendCallbackOwner::drop` (zero locks, zero allocs).
    #[inline]
    fn store_raw(&self, raw: *mut Box<dyn SynthBackend>) {
        if raw.is_null() {
            return;
        }
        let prev = self.ptr.swap(raw, Ordering::AcqRel);
        if !prev.is_null() {
            // SAFETY: `prev` was created by `Box::into_raw(Box::new(backend))` and uniquely claimed by `swap`.
            let _ = unsafe { Box::from_raw(prev) };
        }
    }

    /// Cold-path reclaim of the returned `SynthBackend`.
    fn take(&self) -> Option<Box<dyn SynthBackend>> {
        let raw = self.ptr.swap(core::ptr::null_mut(), Ordering::AcqRel);
        if raw.is_null() {
            None
        } else {
            // SAFETY: `raw` was created by `Box::into_raw(Box::new(backend))` and uniquely claimed by `swap`.
            Some(unsafe { *Box::from_raw(raw) })
        }
    }
}

impl Drop for LockFreeBackendSlot {
    fn drop(&mut self) {
        let _ = self.take();
    }
}

/// Move-only RAII guard held exclusively inside the `cpal` audio callback closure.
///
/// During audio rendering, the callback has direct `&mut dyn SynthBackend` ownership
/// with **zero locks**. When `cpal::Stream` is dropped on `TerminateKDMAPIStream` (or
/// device disconnect on an audio thread), `Drop` returns ownership via a wait-free
/// atomic pointer swap (`LockFreeBackendSlot::store_raw`) with zero locks and zero allocs.
struct BackendCallbackOwner {
    backend_ptr: *mut Box<dyn SynthBackend>,
    reclaim_slot: Arc<LockFreeBackendSlot>,
    scratch: Box<[f32]>,
    batch: [MidiEvent; 256],
}

// SAFETY: `backend_ptr` is uniquely owned by `BackendCallbackOwner` until dropped,
// and `Box<dyn SynthBackend>` is `Send`.
unsafe impl Send for BackendCallbackOwner {}

impl BackendCallbackOwner {
    fn new(
        backend: Box<dyn SynthBackend>,
        reclaim_slot: Arc<LockFreeBackendSlot>,
        scratch_frames: usize,
    ) -> Self {
        Self {
            backend_ptr: Box::into_raw(Box::new(backend)),
            reclaim_slot,
            scratch: vec![0.0f32; scratch_frames * 2].into_boxed_slice(),
            batch: [MidiEvent::EMPTY; 256],
        }
    }

    #[inline]
    fn split_mut(&mut self) -> Option<(&mut dyn SynthBackend, &mut [f32], &mut [MidiEvent; 256])> {
        if self.backend_ptr.is_null() {
            None
        } else {
            // SAFETY: `self` has exclusive `&mut` ownership of `backend_ptr` until `Drop`,
            // and `backend_ptr`, `scratch`, and `batch` are disjoint fields.
            let backend = unsafe { &mut **self.backend_ptr };
            Some((backend, &mut self.scratch, &mut self.batch))
        }
    }

    #[cfg(test)]
    #[inline]
    fn backend_mut(&mut self) -> Option<&mut dyn SynthBackend> {
        if self.backend_ptr.is_null() {
            None
        } else {
            // SAFETY: `self` has exclusive `&mut` ownership of `backend_ptr` until `Drop`.
            Some(unsafe { &mut **self.backend_ptr })
        }
    }
}

impl Drop for BackendCallbackOwner {
    fn drop(&mut self) {
        let raw = self.backend_ptr;
        self.backend_ptr = core::ptr::null_mut();
        self.reclaim_slot.store_raw(raw);
    }
}

impl AudioDriver {
    /// Start real-time `cpal` audio output honoring `buffer_frames`, or gracefully fall back
    /// to a dedicated real-time consumer thread if no audio output device is available.
    ///
    /// Returns `(driver, negotiated_sample_rate)`.
    pub fn start(
        mut backend: Box<dyn SynthBackend>,
        synth_cfg: SynthConfig,
        hooks: RtEngineHooks,
    ) -> Result<(Self, u32), Box<dyn SynthBackend>> {
        let force_headless = std::env::var_os(NO_AUDIO_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        if !force_headless {
            let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<u32, Box<dyn SynthBackend>>>(1);
            let (stop_tx, stop_rx) = mpsc::sync_channel::<()>(1);
            let cpal_hooks = hooks.clone();
            let mut thread_cfg = synth_cfg;

            match thread::Builder::new()
                .name("kdmapiv3-audio-owner".to_string())
                .spawn(move || -> Option<Box<dyn SynthBackend>> {
                    let mut b = Some(backend);
                    match try_start_cpal_stream(&mut b, &mut thread_cfg, &cpal_hooks) {
                        Some((stream, reclaim_slot, sr)) => {
                            let _ = ready_tx.send(Ok(sr));
                            // Keep `cpal::Stream` alive on this thread until stop is requested.
                            let _ = stop_rx.recv();
                            drop(stream);
                            reclaim_slot.take()
                        }
                        None => {
                            if let Some(returned) = b {
                                let _ = ready_tx.send(Err(returned));
                            }
                            None
                        }
                    }
                }) {
                Ok(owner_handle) => match ready_rx.recv() {
                    Ok(Ok(sr)) => {
                        return Ok((
                            Self::CpalOwned {
                                stop_tx,
                                owner_handle,
                            },
                            sr,
                        ));
                    }
                    Ok(Err(returned_backend)) => {
                        let _ = owner_handle.join();
                        backend = returned_backend;
                    }
                    Err(_) => {
                        let _ = owner_handle.join();
                        backend = Box::new(crate::synth::SoundFontSynth::new());
                    }
                },
                Err(_) => {
                    backend = Box::new(crate::synth::SoundFontSynth::new());
                }
            }
        }

        // Headless fallback (e.g., CI environment without sound card or KDMAPIV3_NO_AUDIO=1).
        if backend.init(synth_cfg).is_err() {
            return Err(backend);
        }
        let sr = synth_cfg.sample_rate;
        let spawn = thread::Builder::new()
            .name("kdmapiv3-consumer".to_string())
            .spawn(move || headless_consumer_loop(backend, hooks));

        match spawn {
            Ok(handle) => Ok((Self::HeadlessThread { handle }, sr)),
            Err(_) => Err(Box::new(crate::synth::SoundFontSynth::new())),
        }
    }

    /// Stop audio output cleanly and reclaim the `SynthBackend` instance.
    pub fn stop(self) -> Option<Box<dyn SynthBackend>> {
        match self {
            Self::CpalOwned {
                stop_tx,
                owner_handle,
            } => {
                let _ = stop_tx.send(());
                owner_handle.join().ok().flatten()
            }
            Self::HeadlessThread { handle } => handle.join().ok(),
        }
    }
}

fn try_start_cpal_stream(
    backend_slot: &mut Option<Box<dyn SynthBackend>>,
    synth_cfg: &mut SynthConfig,
    hooks: &RtEngineHooks,
) -> Option<(Stream, Arc<LockFreeBackendSlot>, u32)> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;
    let supported = device.default_output_config().ok()?;

    let sample_format = supported.sample_format();
    let channels = supported.channels().max(1) as usize;
    let device_sr = supported.sample_rate().0.max(8_000);
    synth_cfg.sample_rate = device_sr;

    // Initialize the backend with the exact hardware sample rate before opening the stream.
    if backend_slot.as_mut()?.init(*synth_cfg).is_err() {
        return None;
    }

    // Move backend out without allocating any dummy placeholder; restored on failure below.
    let live_backend = backend_slot.take()?;

    let reclaim_slot = Arc::new(LockFreeBackendSlot::new());
    let scratch_frames = (synth_cfg.buffer_frames as usize)
        .clamp(MIN_BUFFER_FRAMES as usize, MAX_BUFFER_FRAMES as usize);

    let mut owner = BackendCallbackOwner::new(
        live_backend,
        Arc::clone(&reclaim_slot),
        scratch_frames,
    );

    let fixed_cfg = StreamConfig {
        channels: supported.channels(),
        sample_rate: supported.sample_rate(),
        buffer_size: BufferSize::Fixed(synth_cfg.buffer_frames),
    };
    let default_cfg = StreamConfig {
        channels: supported.channels(),
        sample_rate: supported.sample_rate(),
        buffer_size: BufferSize::Default,
    };

    let cb_hooks = hooks.clone();

    let err_hooks = hooks.clone();
    let err_fn = move |_err: cpal::StreamError| {
        // Real-time safe: strictly atomic xrun counter increment (zero logging/allocs).
        err_hooks.telemetry.record_xrun();
    };

    // Try fixed buffer_frames first; if rejected by ALSA/WASAPI device, retry with Default.
    let stream_res = build_typed_stream(
        &device,
        sample_format,
        &fixed_cfg,
        channels,
        device_sr,
        owner,
        cb_hooks,
        err_fn,
    );

    let stream = match stream_res {
        Ok(s) => s,
        Err(()) => {
            // Reclaim backend if fixed buffer creation failed, and retry with default buffer size.
            let reclaimed = reclaim_slot.take()?;
            owner = BackendCallbackOwner::new(
                reclaimed,
                Arc::clone(&reclaim_slot),
                scratch_frames,
            );
            let retry_hooks = hooks.clone();
            let retry_err_hooks = hooks.clone();
            let retry_err_fn = move |_err: cpal::StreamError| {
                retry_err_hooks.telemetry.record_xrun();
            };
            match build_typed_stream(
                &device,
                sample_format,
                &default_cfg,
                channels,
                device_sr,
                owner,
                retry_hooks,
                retry_err_fn,
            ) {
                Ok(s) => s,
                Err(()) => {
                    if let Some(b) = reclaim_slot.take() {
                        *backend_slot = Some(b);
                    }
                    return None;
                }
            }
        }
    };

    if stream.play().is_err() {
        drop(stream);
        if let Some(b) = reclaim_slot.take() {
            *backend_slot = Some(b);
        }
        return None;
    }

    Some((stream, reclaim_slot, device_sr))
}

#[allow(clippy::too_many_arguments)]
fn build_typed_stream<E>(
    device: &cpal::Device,
    sample_format: SampleFormat,
    config: &StreamConfig,
    channels: usize,
    sample_rate: u32,
    mut owner: BackendCallbackOwner,
    hooks: RtEngineHooks,
    err_fn: E,
) -> Result<Stream, ()>
where
    E: FnMut(cpal::StreamError) + Send + 'static,
{
    match sample_format {
        SampleFormat::F32 => device
            .build_output_stream(
                config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    render_audio_callback(data, channels, sample_rate, &mut owner, &hooks);
                },
                err_fn,
                None,
            )
            .map_err(|_| ()),
        SampleFormat::I16 => device
            .build_output_stream(
                config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    render_audio_callback(data, channels, sample_rate, &mut owner, &hooks);
                },
                err_fn,
                None,
            )
            .map_err(|_| ()),
        SampleFormat::U16 => device
            .build_output_stream(
                config,
                move |data: &mut [u16], _: &cpal::OutputCallbackInfo| {
                    render_audio_callback(data, channels, sample_rate, &mut owner, &hooks);
                },
                err_fn,
                None,
            )
            .map_err(|_| ()),
        _ => Err(()),
    }
}

/// Hard real-time audio callback:
/// - Zero heap allocations
/// - Zero mutexes or blocking locks
/// - Zero logging or file I/O
/// - Drains MIDI ring buffers directly and renders stereo audio into `output`.
#[inline]
fn render_audio_callback<T: cpal::FromSample<f32> + Sample>(
    output: &mut [T],
    out_channels: usize,
    sample_rate: u32,
    owner: &mut BackendCallbackOwner,
    hooks: &RtEngineHooks,
) {
    let t0 = Instant::now();
    let total_frames = output.len() / out_channels.max(1);

    let Some((backend, scratch, batch)) = owner.split_mut() else {
        for s in output.iter_mut() {
            *s = T::from_sample(0.0f32);
        }
        return;
    };

    let budget_secs = (total_frames.max(1) as f32) / (sample_rate.max(8_000) as f32);
    let max_drain_dur = Duration::from_secs_f32((budget_secs * MAX_DRAIN_BUDGET_FRACTION).max(0.0005));

    // 1. Drain MIDI events up to the current real-time playhead without allocating.
    drain_midi_events(backend, batch, hooks, Some((t0, max_drain_dur)));

    // 2. Render in chunks up to pre-allocated `scratch` size (never allocates even if
    //    host audio backend requests a larger callback period than `buffer_frames`).
    let chunk_cap_frames = (scratch.len() / 2).max(1);
    let mut rendered_frames = 0usize;

    while rendered_frames < total_frames {
        let frames_now = (total_frames - rendered_frames).min(chunk_cap_frames);
        let stereo_slice = &mut scratch[..frames_now * 2];
        backend.render_audio(stereo_slice);

        let dst_start = rendered_frames * out_channels;
        for f in 0..frames_now {
            let raw_l = stereo_slice[f * 2];
            let raw_r = stereo_slice[f * 2 + 1];
            let l = if raw_l.is_finite() {
                raw_l.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let r = if raw_r.is_finite() {
                raw_r.clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let base = dst_start + f * out_channels;

            if out_channels == 2 {
                if let Some(frame_dst) = output.get_mut(base..base + 2) {
                    frame_dst[0] = T::from_sample(l);
                    frame_dst[1] = T::from_sample(r);
                }
            } else if out_channels == 1 {
                if let Some(dst) = output.get_mut(base) {
                    *dst = T::from_sample(0.5 * (l + r));
                }
            } else {
                for ch in 0..out_channels {
                    if let Some(dst) = output.get_mut(base + ch) {
                        let v = match ch {
                            0 => l,
                            1 => r,
                            _ => 0.0,
                        };
                        *dst = T::from_sample(v);
                    }
                }
            }
        }

        rendered_frames += frames_now;
    }

    // 3. Update lock-free telemetry (active voices, render load %, XRUN overrun detection).
    let stats = backend.stats();
    hooks
        .voice_count
        .store(stats.active_voices, Ordering::Relaxed);

    let elapsed_secs = t0.elapsed().as_secs_f32();
    let render_pct = ((elapsed_secs / budget_secs) * 100.0).min(999.9);
    hooks
        .rendering_time_bits
        .store(render_pct.to_bits(), Ordering::Relaxed);

    if elapsed_secs > budget_secs || stats.xrun_detected {
        hooks.telemetry.record_xrun();
    }
}

#[inline]
fn dispatch_midi_event(backend: &mut dyn SynthBackend, ev: &MidiEvent) {
    match ev.kind {
        EventKind::Short | EventKind::ShortNoBuf => {
            backend.process_short_event(ev.data1);
        }
        EventKind::Custom => {
            backend.process_custom_event(ev.data1, ev.data2, ev.data3);
        }
        EventKind::Reset => {
            backend.reset();
        }
        EventKind::Empty => {}
    }
}

/// Lock-free, allocation-free MIDI ring-buffer drain shared by `cpal` callback and headless loop.
///
/// Strictly adheres to Black MIDI real-time integrity rules (zero note skipping): snapshots
/// the pending queue depth at entry and dispatches 100% of every popped MIDI event faithfully
/// in FIFO order to `SynthBackend` without skipping, striding, or pre-filtering Note-Ons.
/// If `drain_deadline` expires, remaining events stay intact in the ring buffer for the next
/// callback so DSP rendering retains guaranteed headroom.
#[inline]
fn drain_midi_events(
    backend: &mut dyn SynthBackend,
    batch: &mut [MidiEvent; 256],
    hooks: &RtEngineHooks,
    drain_deadline: Option<(Instant, Duration)>,
) -> bool {
    let mut did_work = false;

    if hooks.reset_pending.swap(false, Ordering::AcqRel) {
        backend.reset();
        did_work = true;
    }

    let ring_cap = hooks.short_ring.capacity();
    let snapshot_pending = hooks.short_ring.len_approx();
    let mut remaining_target = snapshot_pending.max(batch.len()).min(ring_cap);
    let mut batches_drained = 0usize;

    while remaining_target > 0 {
        let want = remaining_target.min(batch.len());
        let count = hooks.short_ring.pop_batch(&mut batch[..want]);
        if count == 0 {
            break;
        }
        did_work = true;
        remaining_target = remaining_target.saturating_sub(count);
        batches_drained += 1;

        for ev in &batch[..count] {
            dispatch_midi_event(backend, ev);
        }

        hooks.telemetry.record_events_out(count as u64);

        if count < want {
            break;
        }

        if let Some((start_t, max_dur)) = drain_deadline {
            if (batches_drained & DRAIN_BUDGET_CHECK_MASK) == 0 && start_t.elapsed() >= max_dur {
                break;
            }
        }
    }

    let mut sysex_count = 0u64;
    while let Some(pkt) = hooks.sysex_ring.try_pop() {
        did_work = true;
        backend.process_sysex(pkt.as_slice());
        sysex_count += 1;
    }
    if sysex_count > 0 {
        hooks.telemetry.record_events_out(sysex_count);
    }

    did_work
}

/// Fallback real-time consumer loop used when no audio device is present or `KDMAPIV3_NO_AUDIO=1`.
fn headless_consumer_loop(
    mut backend: Box<dyn SynthBackend>,
    hooks: RtEngineHooks,
) -> Box<dyn SynthBackend> {
    let _ = crate::backend::platform::configure_realtime_thread();

    let mut batch = [MidiEvent::EMPTY; 256];
    let mut idle_spins: u32 = 0;

    while !hooks.stop_requested.load(Ordering::Acquire) {
        let did_work = drain_midi_events(backend.as_mut(), &mut batch, &hooks, None);
        if did_work {
            idle_spins = 0;
            let stats = backend.stats();
            hooks
                .voice_count
                .store(stats.active_voices, Ordering::Relaxed);
            hooks
                .rendering_time_bits
                .store(stats.rendering_time_pct.to_bits(), Ordering::Relaxed);
            if stats.xrun_detected {
                hooks.telemetry.record_xrun();
            }
        } else if idle_spins < 64 {
            core::hint::spin_loop();
            idle_spins += 1;
        } else if idle_spins < 128 {
            thread::yield_now();
            idle_spins += 1;
        } else {
            thread::sleep(Duration::from_micros(250));
        }
    }

    // Drain any final events before shutdown.
    while drain_midi_events(backend.as_mut(), &mut batch, &hooks, None) {}
    backend
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendError, SynthStats};
    use std::path::Path;

    #[derive(Default)]
    struct CountingMockSynth {
        note_ons: usize,
        note_offs: usize,
        ccs: usize,
        prog_changes: usize,
        active_voices: u64,
        last_note_on_key: u8,
    }

    impl SynthBackend for CountingMockSynth {
        fn name(&self) -> &'static str {
            "CountingMockSynth"
        }
        fn init(&mut self, _config: SynthConfig) -> Result<(), BackendError> {
            Ok(())
        }
        fn shutdown(&mut self) {}
        fn reset(&mut self) {}
        fn process_short_event(&mut self, dw_msg: u32) {
            let status = (dw_msg & 0xF0) as u8;
            let d1 = ((dw_msg >> 8) & 0x7F) as u8;
            let d2 = ((dw_msg >> 16) & 0x7F) as u8;
            match status {
                0x90 if d2 > 0 => {
                    self.note_ons += 1;
                    self.active_voices = self.active_voices.saturating_add(1);
                    self.last_note_on_key = d1;
                }
                0x80 | 0x90 => {
                    self.note_offs += 1;
                }
                0xB0 => {
                    self.ccs += 1;
                }
                0xC0 => {
                    self.prog_changes += 1;
                }
                _ => {}
            }
        }
        fn process_sysex(&mut self, _data: &[u8]) {}
        fn process_custom_event(&mut self, _event_type: u32, _chan: u32, _param: u32) {}
        fn render_audio(&mut self, output: &mut [f32]) {
            output.fill(0.0);
        }
        fn load_soundfont_list(&mut self, _path: &Path) -> Result<(), BackendError> {
            Ok(())
        }
        fn stats(&self) -> SynthStats {
            SynthStats {
                active_voices: self.active_voices,
                ..SynthStats::default()
            }
        }
    }

    fn make_test_hooks(cap: usize) -> RtEngineHooks {
        RtEngineHooks::new(Arc::new(RtEngineShared::new(cap)))
    }

    #[test]
    fn burst_drain_dispatches_100_percent_events_with_zero_note_skipping() {
        let hooks = make_test_hooks(262_144);

        // Queue 100,000 mixed events (Note-On, Note-Off, CC64, Program Change).
        // Per Black MIDI real-time integrity rules, 100% of every event must be
        // delivered faithfully in FIFO order with ZERO Note-On skipping.
        let total_events = 100_000usize;
        let mut expected_note_ons = 0usize;
        let mut expected_note_offs = 0usize;
        let mut expected_ccs = 0usize;
        let mut expected_progs = 0usize;

        for i in 0..total_events {
            let msg = if i + 1 == total_events {
                expected_note_ons += 1;
                0x007F_4890 // Final Note-On key 72
            } else {
                match i % 10 {
                    0 => {
                        expected_note_offs += 1;
                        0x0000_3C80 // Note-Off
                    }
                    1 => {
                        expected_note_offs += 1;
                        0x0000_3C90 // Note-On vel=0 (Note-Off)
                    }
                    2 => {
                        expected_ccs += 1;
                        0x007F_40B0 // CC64 Sustain On
                    }
                    3 => {
                        expected_progs += 1;
                        0x0000_05C0 // Program Change
                    }
                    _ => {
                        expected_note_ons += 1;
                        0x007F_3C90 // Note-On (must NEVER be skipped)
                    }
                }
            };
            assert!(hooks.short_ring.try_push(MidiEvent::short_msg(msg)));
        }

        let mut synth = CountingMockSynth::default();
        let mut batch = [MidiEvent::EMPTY; 256];

        let did_work = drain_midi_events(
            &mut synth,
            &mut batch,
            &hooks,
            Some((Instant::now(), Duration::from_secs(1))),
        );
        assert!(did_work);
        assert_eq!(
            hooks.short_ring.len_approx(),
            0,
            "entire burst must drain in one callback"
        );
        assert_eq!(hooks.telemetry.events_out(), total_events as u64);
        assert_eq!(synth.note_ons, expected_note_ons);
        assert_eq!(synth.note_offs, expected_note_offs);
        assert_eq!(synth.ccs, expected_ccs);
        assert_eq!(synth.prog_changes, expected_progs);
        assert_eq!(synth.last_note_on_key, 72);
    }

    #[test]
    fn note_on_wall_followed_by_note_off_wall_delivers_all_notes_faithfully() {
        let hooks = make_test_hooks(262_144);

        // Simulate Galaxy Collapse Black MIDI wall inside a single callback:
        // 100,000 Note-Ons followed immediately by 100,000 Note-Offs.
        for i in 0..100_000usize {
            let key = (21 + (i % 88)) as u32;
            assert!(hooks
                .short_ring
                .try_push(MidiEvent::short_msg(0x0064_0090 | (key << 8))));
        }
        for i in 0..100_000usize {
            let key = (21 + (i % 88)) as u32;
            assert!(hooks
                .short_ring
                .try_push(MidiEvent::short_msg(0x0000_0080 | (key << 8))));
        }

        let mut synth = CountingMockSynth::default();
        let mut batch = [MidiEvent::EMPTY; 256];

        let did_work = drain_midi_events(
            &mut synth,
            &mut batch,
            &hooks,
            Some((Instant::now(), Duration::from_secs(1))),
        );
        assert!(did_work);
        assert_eq!(hooks.short_ring.len_approx(), 0);
        assert_eq!(synth.note_ons, 100_000, "100% of Note-Ons must be delivered");
        assert_eq!(synth.note_offs, 100_000, "100% of Note-Offs must be delivered");
    }

    #[test]
    fn drain_deadline_yields_cleanly_and_preserves_remaining_events_for_next_callback() {
        let hooks = make_test_hooks(262_144);

        // Push 100,000 alternating Note-On / Note-Off events.
        for i in 0..100_000usize {
            let msg = if (i & 1) == 0 {
                0x0064_3C90
            } else {
                0x0000_3C80
            };
            assert!(hooks.short_ring.try_push(MidiEvent::short_msg(msg)));
        }

        let mut synth = CountingMockSynth::default();
        let mut batch = [MidiEvent::EMPTY; 256];

        // Callback 1: deadline expires at batch 64 (16,384 events) -> yields cleanly with zero skipping.
        let did_work = drain_midi_events(
            &mut synth,
            &mut batch,
            &hooks,
            Some((Instant::now(), Duration::ZERO)),
        );
        assert!(did_work);
        assert_eq!(hooks.telemetry.events_out(), 64 * 256);
        assert_eq!(synth.note_ons, 8_192);
        assert_eq!(synth.note_offs, 8_192);
        assert_eq!(hooks.short_ring.len_approx(), 100_000 - 16_384);

        // Callback 2: finishes draining all remaining events faithfully with zero loss.
        let did_work2 = drain_midi_events(
            &mut synth,
            &mut batch,
            &hooks,
            Some((Instant::now(), Duration::from_secs(1))),
        );
        assert!(did_work2);
        assert_eq!(hooks.short_ring.len_approx(), 0);
        assert_eq!(synth.note_ons, 50_000);
        assert_eq!(synth.note_offs, 50_000);
    }

    #[test]
    fn lock_free_backend_slot_handoff_on_callback_owner_drop() {
        let slot = Arc::new(LockFreeBackendSlot::new());
        assert!(slot.take().is_none());

        {
            let backend: Box<dyn SynthBackend> = Box::new(CountingMockSynth::default());
            let mut owner = BackendCallbackOwner::new(backend, Arc::clone(&slot), 256);
            owner
                .backend_mut()
                .expect("backend present")
                .process_short_event(0x007F_3C90);
            assert_eq!(owner.backend_mut().unwrap().stats().active_voices, 1);
        }

        let reclaimed = slot.take().expect("backend must be returned lock-free on drop");
        assert_eq!(reclaimed.stats().active_voices, 1);
        assert!(slot.take().is_none());
    }
}
