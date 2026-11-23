//! Pre-allocated, lock-free bounded ring buffers and atomic telemetry counters.
//!
//! 100% OS-agnostic. Zero heap allocations, zero locks, and zero panics on all
//! push/pop/telemetry operations after initial construction.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Default short MIDI event ring buffer capacity (power of two).
/// 1,048,576 slots x 24 bytes (~24 MiB) absorbs extreme Black MIDI bursts without stalling.
pub const DEFAULT_SHORT_RING_CAPACITY: usize = 1_048_576;

/// Default SysEx/long message ring buffer capacity (power of two).
pub const DEFAULT_SYSEX_RING_CAPACITY: usize = 256;

/// Maximum inline SysEx payload size copied without heap allocation.
pub const MAX_SYSEX_LEN: usize = 512;

/// Cache-line padded wrapper to eliminate false sharing between hot atomic fields.
#[repr(align(64))]
#[derive(Debug)]
pub struct CachePadded<T>(pub T);

impl<T> CachePadded<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T> Deref for CachePadded<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for CachePadded<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Discriminator for fixed-size events stored in the primary lock-free ring buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventKind {
    Empty = 0,
    Short = 1,
    ShortNoBuf = 2,
    Custom = 3,
    Reset = 4,
}

/// Fixed-size, `Copy` MIDI event representation for the lock-free hot path.
/// Never allocates and never owns heap memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct MidiEvent {
    pub kind: EventKind,
    pub _pad: [u8; 3],
    pub data1: u32,
    pub data2: u32,
    pub data3: u32,
}

impl MidiEvent {
    pub const EMPTY: Self = Self {
        kind: EventKind::Empty,
        _pad: [0; 3],
        data1: 0,
        data2: 0,
        data3: 0,
    };

    #[inline]
    pub const fn short_msg(dw_msg: u32) -> Self {
        Self {
            kind: EventKind::Short,
            _pad: [0; 3],
            data1: dw_msg,
            data2: 0,
            data3: 0,
        }
    }

    #[inline]
    pub const fn short_msg_no_buf(dw_msg: u32) -> Self {
        Self {
            kind: EventKind::ShortNoBuf,
            _pad: [0; 3],
            data1: dw_msg,
            data2: 0,
            data3: 0,
        }
    }

    #[inline]
    pub const fn custom(event_type: u32, chan: u32, param: u32) -> Self {
        Self {
            kind: EventKind::Custom,
            _pad: [0; 3],
            data1: event_type,
            data2: chan,
            data3: param,
        }
    }

    #[inline]
    pub const fn reset() -> Self {
        Self {
            kind: EventKind::Reset,
            _pad: [0; 3],
            data1: 0,
            data2: 0,
            data3: 0,
        }
    }
}

/// Fixed-size, `Copy` SysEx payload packet for allocation-free `SendDirectLongData`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct SysExPacket {
    pub len: u16,
    pub data: [u8; MAX_SYSEX_LEN],
}

impl SysExPacket {
    pub const EMPTY: Self = Self {
        len: 0,
        data: [0u8; MAX_SYSEX_LEN],
    };

    /// Copy up to `MAX_SYSEX_LEN` bytes from `src` into a stack-allocated packet.
    /// Truncates gracefully if `src.len() > MAX_SYSEX_LEN` without panicking.
    #[inline]
    pub fn from_slice_clamped(src: &[u8]) -> Self {
        let copy_len = if src.len() > MAX_SYSEX_LEN {
            MAX_SYSEX_LEN
        } else {
            src.len()
        };
        let mut pkt = Self::EMPTY;
        pkt.len = copy_len as u16;
        pkt.data[..copy_len].copy_from_slice(&src[..copy_len]);
        pkt
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        let n = (self.len as usize).min(MAX_SYSEX_LEN);
        &self.data[..n]
    }
}

/// Lock-free atomic telemetry counters for non-blocking observation.
/// Each counter lives on its own cache line to prevent false sharing.
pub struct TelemetryCounters {
    events_in: CachePadded<AtomicU64>,
    events_out: CachePadded<AtomicU64>,
    events_dropped: CachePadded<AtomicU64>,
    xrun_count: CachePadded<AtomicU64>,
}

/// Plain snapshot of atomic telemetry counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    pub events_in: u64,
    pub events_out: u64,
    pub events_dropped: u64,
    pub xrun_count: u64,
}

impl TelemetryCounters {
    #[inline]
    pub const fn new() -> Self {
        Self {
            events_in: CachePadded::new(AtomicU64::new(0)),
            events_out: CachePadded::new(AtomicU64::new(0)),
            events_dropped: CachePadded::new(AtomicU64::new(0)),
            xrun_count: CachePadded::new(AtomicU64::new(0)),
        }
    }

    #[inline]
    pub fn record_event_in(&self) {
        self.events_in.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_events_out(&self, count: u64) {
        self.events_out.fetch_add(count, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_event_dropped(&self) {
        self.events_dropped.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_xrun(&self) {
        self.xrun_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn events_in(&self) -> u64 {
        self.events_in.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn events_out(&self) -> u64 {
        self.events_out.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn events_dropped(&self) -> u64 {
        self.events_dropped.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn xrun_count(&self) -> u64 {
        self.xrun_count.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            events_in: self.events_in(),
            events_out: self.events_out(),
            events_dropped: self.events_dropped(),
            xrun_count: self.xrun_count(),
        }
    }

    #[inline]
    pub fn reset(&self) {
        self.events_in.store(0, Ordering::Relaxed);
        self.events_out.store(0, Ordering::Relaxed);
        self.events_dropped.store(0, Ordering::Relaxed);
        self.xrun_count.store(0, Ordering::Relaxed);
    }
}

impl Default for TelemetryCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Single ring slot with sequence number for Vyukov's bounded lock-free queue.
struct Slot<T> {
    seq: AtomicUsize,
    value: UnsafeCell<MaybeUninit<T>>,
}

/// Pre-allocated, lock-free bounded MPMC/MPSC ring buffer (Vyukov algorithm).
///
/// - Fixed power-of-two capacity allocated exactly once in `with_capacity`.
/// - Zero heap allocations, zero blocking locks, and zero panics in `try_push`,
///   `try_pop`, `pop_batch`, and `clear`.
/// - Multi-producer and multi-consumer safe via per-slot atomic sequence numbers
///   with `Acquire`/`Release` synchronization.
pub struct LockFreeRingBuffer<T> {
    buffer: Box<[Slot<T>]>,
    mask: usize,
    enqueue_pos: CachePadded<AtomicUsize>,
    dequeue_pos: CachePadded<AtomicUsize>,
}

// SAFETY: Access to each slot's `UnsafeCell` is strictly serialized by the per-slot
// `seq` atomic with Acquire/Release ordering, so `T: Send` is sufficient for `Sync`.
unsafe impl<T: Send> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Send> Sync for LockFreeRingBuffer<T> {}

impl<T> LockFreeRingBuffer<T> {
    /// Allocate a fixed-size ring buffer.
    ///
    /// `requested_capacity` is rounded up to the next power of two (minimum 2).
    /// This is the ONLY method that allocates heap memory.
    pub fn with_capacity(requested_capacity: usize) -> Self {
        let cap = requested_capacity.max(2).checked_next_power_of_two().unwrap_or(2);
        let mut vec = Vec::with_capacity(cap);
        for i in 0..cap {
            vec.push(Slot {
                seq: AtomicUsize::new(i),
                value: UnsafeCell::new(MaybeUninit::uninit()),
            });
        }
        Self {
            buffer: vec.into_boxed_slice(),
            mask: cap - 1,
            enqueue_pos: CachePadded::new(AtomicUsize::new(0)),
            dequeue_pos: CachePadded::new(AtomicUsize::new(0)),
        }
    }

    /// Fixed capacity (always a power of two >= 2).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Lock-free, allocation-free, panic-free push.
    /// Returns `true` if enqueued, or `false` if the ring buffer is full (graceful overflow).
    #[inline]
    pub fn try_push(&self, value: T) -> bool {
        let mut pos = self.enqueue_pos.load(Ordering::Relaxed);
        loop {
            // SAFETY: `self.buffer.len()` is a power of two and `self.mask == len - 1`,
            // so `pos & self.mask` is strictly in-bounds.
            let slot = unsafe { self.buffer.get_unchecked(pos & self.mask) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(pos) as isize;

            if diff == 0 {
                match self.enqueue_pos.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: Exclusive write access to `slot.value` is guaranteed because
                        // `seq == pos` and our CAS uniquely advanced `enqueue_pos` past `pos`.
                        unsafe {
                            (*slot.value.get()).write(value);
                        }
                        slot.seq.store(pos.wrapping_add(1), Ordering::Release);
                        return true;
                    }
                    Err(actual) => {
                        pos = actual;
                    }
                }
            } else if diff < 0 {
                // Full: caller must drop/coalesce cleanly without stalling.
                return false;
            } else {
                // Another producer won this slot; reload current head.
                pos = self.enqueue_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Lock-free, allocation-free, panic-free batched push for `Copy` events.
    ///
    /// Claims up to `values.len()` contiguous free slots with a single `enqueue_pos` CAS,
    /// returning the number of items enqueued (may be `< values.len()` if the ring fills).
    #[inline]
    pub fn try_push_batch(&self, values: &[T]) -> usize
    where
        T: Copy,
    {
        let cap = self.mask + 1;
        let max_want = values.len().min(cap);
        if max_want == 0 {
            return 0;
        }

        let mut pos = self.enqueue_pos.load(Ordering::Relaxed);
        loop {
            let mut avail = 0usize;
            let mut stop_diff: isize = 0;
            while avail < max_want {
                let cur = pos.wrapping_add(avail);
                // SAFETY: `cur & self.mask < self.buffer.len()` by power-of-two construction.
                let slot = unsafe { self.buffer.get_unchecked(cur & self.mask) };
                let seq = slot.seq.load(Ordering::Acquire);
                let diff = seq.wrapping_sub(cur) as isize;
                if diff == 0 {
                    avail += 1;
                } else {
                    stop_diff = diff;
                    break;
                }
            }

            if avail == 0 {
                if stop_diff < 0 {
                    return 0;
                }
                pos = self.enqueue_pos.load(Ordering::Relaxed);
                continue;
            }

            match self.enqueue_pos.compare_exchange(
                pos,
                pos.wrapping_add(avail),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    for (i, &val) in values.iter().take(avail).enumerate() {
                        let cur = pos.wrapping_add(i);
                        // SAFETY:
                        // 1. We observed `slot.seq == cur` with `Acquire`, so consumers finished reading.
                        // 2. Our CAS uniquely claimed `[pos, pos + avail)` from `enqueue_pos`.
                        let slot = unsafe { self.buffer.get_unchecked(cur & self.mask) };
                        unsafe {
                            (*slot.value.get()).write(val);
                        }
                        slot.seq.store(cur.wrapping_add(1), Ordering::Release);
                    }
                    return avail;
                }
                Err(actual) => {
                    pos = actual;
                }
            }
        }
    }

    /// Lock-free, allocation-free, panic-free pop.
    /// Returns `None` if the ring buffer is empty.
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        let mut pos = self.dequeue_pos.load(Ordering::Relaxed);
        loop {
            // SAFETY: `pos & self.mask < self.buffer.len()` by construction.
            let slot = unsafe { self.buffer.get_unchecked(pos & self.mask) };
            let seq = slot.seq.load(Ordering::Acquire);
            let diff = seq.wrapping_sub(pos.wrapping_add(1)) as isize;

            if diff == 0 {
                match self.dequeue_pos.compare_exchange_weak(
                    pos,
                    pos.wrapping_add(1),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // SAFETY: `seq == pos + 1` (observed with Acquire) establishes happens-before
                        // with the producer's write, and our CAS grants exclusive read ownership.
                        let val = unsafe { (*slot.value.get()).assume_init_read() };
                        slot.seq
                            .store(pos.wrapping_add(self.mask).wrapping_add(1), Ordering::Release);
                        return Some(val);
                    }
                    Err(actual) => {
                        pos = actual;
                    }
                }
            } else if diff < 0 {
                return None;
            } else {
                pos = self.dequeue_pos.load(Ordering::Relaxed);
            }
        }
    }

    /// Drain up to `out.len()` items into a caller-provided buffer without allocating.
    ///
    /// Uses a lock-free batched claim (a single `dequeue_pos` CAS per contiguous ready run)
    /// while remaining 100% MPMC-safe via per-slot Acquire/Release sequence numbers.
    #[inline]
    pub fn pop_batch(&self, out: &mut [T]) -> usize {
        let cap = self.mask + 1;
        let max_want = out.len().min(cap);
        if max_want == 0 {
            return 0;
        }

        let mut pos = self.dequeue_pos.load(Ordering::Relaxed);
        loop {
            // Probe contiguous ready slots starting at `pos` with Acquire ordering so every
            // producer write in `[pos, pos + avail)` happens-before our reads below.
            let mut avail = 0usize;
            let mut stop_diff: isize = 0;
            while avail < max_want {
                let cur = pos.wrapping_add(avail);
                // SAFETY: `cur & self.mask < self.buffer.len()` by power-of-two construction.
                let slot = unsafe { self.buffer.get_unchecked(cur & self.mask) };
                let seq = slot.seq.load(Ordering::Acquire);
                let diff = seq.wrapping_sub(cur.wrapping_add(1)) as isize;
                if diff == 0 {
                    avail += 1;
                } else {
                    stop_diff = diff;
                    break;
                }
            }

            if avail == 0 {
                if stop_diff < 0 {
                    return 0;
                }
                // Another consumer advanced `dequeue_pos`; reload and retry.
                pos = self.dequeue_pos.load(Ordering::Relaxed);
                continue;
            }

            match self.dequeue_pos.compare_exchange(
                pos,
                pos.wrapping_add(avail),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    for (i, dst) in out.iter_mut().take(avail).enumerate() {
                        let cur = pos.wrapping_add(i);
                        // SAFETY:
                        // 1. We observed `slot.seq == cur + 1` with `Ordering::Acquire` above,
                        //    establishing happens-before with the producer's write.
                        // 2. Our CAS uniquely claimed `[pos, pos + avail)` from `dequeue_pos`,
                        //    so no other consumer can read these slots and no producer can reuse
                        //    them until we publish `cur + cap` with `Ordering::Release`.
                        let slot = unsafe { self.buffer.get_unchecked(cur & self.mask) };
                        *dst = unsafe { (*slot.value.get()).assume_init_read() };
                        slot.seq.store(cur.wrapping_add(cap), Ordering::Release);
                    }
                    return avail;
                }
                Err(actual) => {
                    pos = actual;
                }
            }
        }
    }

    /// Approximate number of queued items (non-linearizable snapshot for telemetry/tests).
    #[inline]
    pub fn len_approx(&self) -> usize {
        let enq = self.enqueue_pos.load(Ordering::Relaxed);
        let deq = self.dequeue_pos.load(Ordering::Relaxed);
        enq.wrapping_sub(deq).min(self.capacity())
    }

    /// Returns `true` if the buffer appears empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len_approx() == 0
    }

    /// Drain all currently queued elements without reallocating.
    #[inline]
    pub fn clear(&self) {
        while self.try_pop().is_some() {}
    }
}

impl<T> Drop for LockFreeRingBuffer<T> {
    fn drop(&mut self) {
        // Ensure any remaining initialized `T` values are dropped cleanly.
        while self.try_pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ringbuf_fifo_and_graceful_overflow() {
        let rb = LockFreeRingBuffer::<MidiEvent>::with_capacity(4);
        assert_eq!(rb.capacity(), 4);

        for i in 0..4u32 {
            assert!(rb.try_push(MidiEvent::short_msg(0x0040_3C90 + i)));
        }
        // 5th push must fail gracefully without panicking or blocking.
        assert!(!rb.try_push(MidiEvent::short_msg(0xDEAD_BEEF)));

        let mut batch = [MidiEvent::EMPTY; 8];
        let n = rb.pop_batch(&mut batch);
        assert_eq!(n, 4);
        for (i, ev) in batch.iter().take(4).enumerate() {
            assert_eq!(ev.data1, 0x0040_3C90 + i as u32);
        }
        assert!(rb.try_pop().is_none());

        // Batched push: 6 events into capacity-4 ring enqueues 4 cleanly and drops overflow.
        let src: [MidiEvent; 6] = core::array::from_fn(|i| MidiEvent::short_msg(0x0050_3C90 + i as u32));
        assert_eq!(rb.try_push_batch(&src), 4);
        let n2 = rb.pop_batch(&mut batch);
        assert_eq!(n2, 4);
        for (i, ev) in batch.iter().take(4).enumerate() {
            assert_eq!(ev.data1, 0x0050_3C90 + i as u32);
        }
        assert!(rb.is_empty());
    }

    #[test]
    fn telemetry_counters_track_and_reset() {
        let t = TelemetryCounters::new();
        t.record_event_in();
        t.record_event_in();
        t.record_events_out(2);
        t.record_event_dropped();
        t.record_xrun();

        let snap = t.snapshot();
        assert_eq!(snap.events_in, 2);
        assert_eq!(snap.events_out, 2);
        assert_eq!(snap.events_dropped, 1);
        assert_eq!(snap.xrun_count, 1);

        t.reset();
        assert_eq!(t.snapshot(), TelemetrySnapshot::default());
    }

    #[test]
    fn pop_batch_mpmc_concurrent_no_loss_or_duplication() {
        use std::sync::Arc;
        use std::thread;

        let rb = Arc::new(LockFreeRingBuffer::<u64>::with_capacity(1024));
        let producers = 4usize;
        let per_producer = 50_000u64;
        let total = producers as u64 * per_producer;

        let mut prod_handles = Vec::new();
        for p in 0..producers {
            let rb = Arc::clone(&rb);
            prod_handles.push(thread::spawn(move || {
                let base = (p as u64) * per_producer;
                for i in 0..per_producer {
                    let val = base + i;
                    while !rb.try_push(val) {
                        thread::yield_now();
                    }
                }
            }));
        }

        let consumers = 2usize;
        let consumed = Arc::new(AtomicU64::new(0));
        let sum = Arc::new(AtomicU64::new(0));
        let mut cons_handles = Vec::new();
        for _ in 0..consumers {
            let rb = Arc::clone(&rb);
            let consumed = Arc::clone(&consumed);
            let sum = Arc::clone(&sum);
            cons_handles.push(thread::spawn(move || {
                let mut buf = [0u64; 256];
                loop {
                    let n = rb.pop_batch(&mut buf);
                    if n > 0 {
                        let mut local_sum = 0u64;
                        for &v in &buf[..n] {
                            local_sum = local_sum.wrapping_add(v);
                        }
                        sum.fetch_add(local_sum, Ordering::Relaxed);
                        if consumed.fetch_add(n as u64, Ordering::AcqRel) + (n as u64) >= total {
                            break;
                        }
                    } else if consumed.load(Ordering::Acquire) >= total {
                        break;
                    } else {
                        thread::yield_now();
                    }
                }
            }));
        }

        for h in prod_handles {
            h.join().unwrap();
        }
        for h in cons_handles {
            h.join().unwrap();
        }

        assert_eq!(consumed.load(Ordering::Acquire), total);
        let expected_sum = (total - 1) * total / 2;
        assert_eq!(sum.load(Ordering::Acquire), expected_sum);
        assert!(rb.is_empty());
    }
}
