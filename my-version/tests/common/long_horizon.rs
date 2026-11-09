//! #931: shared fixtures of the long-horizon label tests: hourly bars, real
//! label columns from the label builder, target spec files, stores and tables.
#![allow(dead_code)]

use rcsr::labels::{self, LongLabelSpec, LongLabels};
use rcsr::ohlcv::Bar;

pub const HOUR: i64 = 3600;
/// 2021-01-04T00:00:00Z, a Monday.
pub const ANCHOR: i64 = 1_609_718_400;
/// The default purge of the protocol windows table (1 h entry + 60 h label).
pub const DEFAULT_PURGE: f64 = 219_600.0;
/// Horizon in hourly bars and the required purge `entry_offset + horizon` in seconds.
pub const HORIZONS: [(usize, i64); 3] = [(72, 262_800), (168, 608_400), (336, 1_213_200)];

/// Hourly bars from `start` with a distinct open and close at every bar.
pub fn bars(start: i64, n: usize, phase: f64) -> Vec<Bar> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            let o = 100.0 + 3.0 * (x * 0.37 + phase).sin() + 0.01 * x;
            let c = 100.0 + 3.0 * (x * 0.37 + phase + 0.4).sin() + 0.01 * x;
            Bar::new(start + i as i64 * HOUR, o, o.max(c) + 0.3, o.min(c) - 0.3, c, 10.0 + (i % 7) as f64)
        })
        .collect()
}

pub fn spec(src: &str) -> LongLabelSpec {
    LongLabelSpec::parse(src, HOUR).unwrap()
}

pub fn build(bars: &[Bar], src: &str) -> LongLabels {
    labels::build(bars, &spec(src), None).unwrap()
}

/// `long:<h>,name=h<h>,fee=10` plus `extra`.
pub fn src(h: usize, extra: &str) -> String {
    format!("long:{h},name=h{h},fee=10{extra}")
}

/// An `rcsr-target-spec/1` file with the `y` entry of `spec`.
pub fn spec_file(spec: &LongLabelSpec) -> String {
    let ts = spec.target_spec(&spec.column("y"), &spec.column("fwd"));
    format!("{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{{}}}]}}", ts.file_json())
}

/// A symbol-major `.rcsb` store of raw bars (`ts,symbol,open,high,low,close,volume`).
pub fn write_store(path: &str, series: &[(&str, &[Bar])]) {
    let mut csv = String::from("ts,symbol,open,high,low,close,volume\n");
    for (sym, bars) in series {
        for b in bars.iter() {
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", b.time, sym, b.open, b.high, b.low, b.close, b.volume));
        }
    }
    let t = rcsr::data::parse_table_with_time(&csv, false, Some("ts")).unwrap();
    rcsr::store::write_store_with_time(path, &t, 1024, Some(0)).unwrap();
}

/// Index of the row stamped `time` in a contiguous hourly series.
pub fn row_of(bars: &[Bar], time: f64) -> usize {
    ((time as i64 - bars[0].time) / HOUR) as usize
}

pub fn fmt(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "NaN".to_string()
    }
}
