//! Non-blocking 2Hz tail reader for `kdmapiv3.log` telemetry summaries.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, Instant};

const POLL_INTERVAL: Duration = Duration::from_millis(500); // 2Hz
const TAIL_BYTES: i64 = 4_096;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TelemetryReading {
    pub log_found: bool,
    pub events_in: u64,
    pub events_in_per_sec: u64,
    pub events_out: u64,
    pub events_out_per_sec: u64,
    pub events_dropped: u64,
    pub xrun_count: u64,
    pub active_voices: u64,
    pub render_pct: f32,
}

pub struct TelemetryPoller {
    last_poll: Option<Instant>,
    reading: TelemetryReading,
}

impl Default for TelemetryPoller {
    fn default() -> Self {
        Self {
            last_poll: None,
            reading: TelemetryReading::default(),
        }
    }
}

impl TelemetryPoller {
    /// Poll `log_path` at most once every 500ms (2Hz), reading only the tail chunk
    /// so UI rendering is never stalled even on long-running sessions.
    pub fn poll_if_due(&mut self, log_path: &Path) -> &TelemetryReading {
        let due = match self.last_poll {
            None => true,
            Some(prev) => prev.elapsed() >= POLL_INTERVAL,
        };
        if due {
            self.last_poll = Some(Instant::now());
            self.reading = read_latest_telemetry(log_path);
        }
        &self.reading
    }

    pub fn force_refresh(&mut self, log_path: &Path) {
        self.last_poll = Some(Instant::now());
        self.reading = read_latest_telemetry(log_path);
    }
}

fn read_latest_telemetry(log_path: &Path) -> TelemetryReading {
    let mut file = match File::open(log_path) {
        Ok(f) => f,
        Err(_) => return TelemetryReading::default(),
    };

    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return TelemetryReading {
            log_found: true,
            ..TelemetryReading::default()
        };
    }

    if len > TAIL_BYTES as u64 {
        let _ = file.seek(SeekFrom::End(-TAIL_BYTES));
    }

    let mut buf = Vec::with_capacity(TAIL_BYTES as usize);
    if file.read_to_end(&mut buf).is_err() {
        return TelemetryReading::default();
    }

    let text = String::from_utf8_lossy(&buf);
    for line in text.lines().rev() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(parsed) = parse_telemetry_line(trimmed) {
            return parsed;
        }
    }

    TelemetryReading {
        log_found: true,
        ..TelemetryReading::default()
    }
}

/// Parse a `kdmapiv3.log` line formatted as:
/// `[ts=...] events_in=5000 (5000/s) events_out=5000 (5000/s) events_dropped=0 (+0) xrun_count=0 voices=0 render_pct=0.00`
pub fn parse_telemetry_line(line: &str) -> Option<TelemetryReading> {
    if !line.contains("events_in=") {
        return None;
    }

    let mut reading = TelemetryReading {
        log_found: true,
        ..TelemetryReading::default()
    };

    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        if let Some(v) = tok.strip_prefix("events_in=") {
            reading.events_in = v.parse().unwrap_or(0);
            if let Some(rate_tok) = tokens.get(i + 1) {
                if let Some(rate) = parse_rate_paren(rate_tok) {
                    reading.events_in_per_sec = rate;
                    i += 1;
                }
            }
        } else if let Some(v) = tok.strip_prefix("events_out=") {
            reading.events_out = v.parse().unwrap_or(0);
            if let Some(rate_tok) = tokens.get(i + 1) {
                if let Some(rate) = parse_rate_paren(rate_tok) {
                    reading.events_out_per_sec = rate;
                    i += 1;
                }
            }
        } else if let Some(v) = tok.strip_prefix("events_dropped=") {
            reading.events_dropped = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("xrun_count=") {
            reading.xrun_count = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("voices=") {
            reading.active_voices = v.parse().unwrap_or(0);
        } else if let Some(v) = tok.strip_prefix("render_pct=") {
            reading.render_pct = v.parse().unwrap_or(0.0);
        }
        i += 1;
    }

    Some(reading)
}

fn parse_rate_paren(tok: &str) -> Option<u64> {
    let inner = tok.strip_prefix('(')?.strip_suffix("/s)")?;
    inner.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_telemetry_summary_line() {
        let line = "[ts=1732292000] events_in=1500000 (750000/s) events_out=1499900 (749950/s) events_dropped=100 (+10) xrun_count=2 voices=4096 render_pct=18.50";
        let r = parse_telemetry_line(line).expect("should parse");
        assert!(r.log_found);
        assert_eq!(r.events_in, 1_500_000);
        assert_eq!(r.events_in_per_sec, 750_000);
        assert_eq!(r.events_out, 1_499_900);
        assert_eq!(r.events_out_per_sec, 749_950);
        assert_eq!(r.events_dropped, 100);
        assert_eq!(r.xrun_count, 2);
        assert_eq!(r.active_voices, 4096);
        assert!((r.render_pct - 18.5).abs() < 0.01);
    }
}
