//! Release-mode high-rate `SendDirectData` throughput & zero-allocation benchmark.
//!
//! Run with:
//!   cargo run --release --example stress_bench

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use OmniMIDI::{
    pending_short_events, reset_telemetry, short_ring_capacity, telemetry_snapshot,
    InitializeKDMAPIStream, ResetKDMAPIStream, SendDirectData, SendDirectDataNoBuf,
    TerminateKDMAPIStream,
};

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

fn main() {
    std::env::set_var(OmniMIDI::NO_AUDIO_ENV, "1");
    assert_eq!(InitializeKDMAPIStream(), 1, "InitializeKDMAPIStream failed");
    reset_telemetry();

    // Warm up consumer thread.
    SendDirectData(0x0064_3C90);
    while pending_short_events() > 0 {
        std::thread::yield_now();
    }
    reset_telemetry();

    const TOTAL_EVENTS: u64 = 5_000_000;

    let allocs_before = ALLOC_COUNT.load(Ordering::SeqCst);
    let start = Instant::now();

    for i in 0..TOTAL_EVENTS {
        let note = ((i & 0x7F) as u32) << 8;
        let ch = (i & 0x0F) as u32;
        let vel = if (i & 1) == 0 { 0x007F_0000 } else { 0 };
        let msg = vel | note | 0x90 | ch;

        if (i & 0x1F) == 0 {
            SendDirectDataNoBuf(msg);
        } else {
            SendDirectData(msg);
        }
    }

    let elapsed = start.elapsed();
    let allocs_after = ALLOC_COUNT.load(Ordering::SeqCst);
    let hot_allocs = allocs_after.saturating_sub(allocs_before);

    ResetKDMAPIStream();
    while pending_short_events() > 0 {
        std::thread::yield_now();
    }

    let snap = telemetry_snapshot();
    let secs = elapsed.as_secs_f64();
    let mev_per_sec = (TOTAL_EVENTS as f64 / secs) / 1_000_000.0;
    let ns_per_event = (elapsed.as_nanos() as f64) / (TOTAL_EVENTS as f64);

    println!("=== kdmapiv3 Stage 2 Hot-Path Stress Benchmark ===");
    println!("Ring capacity    : {}", short_ring_capacity());
    println!("Events sent      : {}", TOTAL_EVENTS);
    println!("Elapsed          : {:.3} ms", secs * 1000.0);
    println!("Throughput       : {:.2} M events/sec", mev_per_sec);
    println!("Avg call latency : {:.2} ns/event", ns_per_event);
    println!("events_in        : {}", snap.events_in);
    println!("events_out       : {}", snap.events_out);
    println!("events_dropped   : {}", snap.events_dropped);
    println!("xrun_count       : {}", snap.xrun_count);
    println!("Hot-path allocs  : {}", hot_allocs);

    assert_eq!(hot_allocs, 0, "Hot path allocated on heap!");
    assert_eq!(TerminateKDMAPIStream(), 1, "TerminateKDMAPIStream failed");
}
