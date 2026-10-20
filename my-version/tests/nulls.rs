//! Block-permutation null for autocorrelated targets (issue #114).
//!
//! The i.i.d. row shuffle destroys autocorrelation, so a real pipeline looks
//! better than a null that no longer resembles the data. These tests pin the
//! permutation mechanics and show, on synthetic AR(1) series, that the i.i.d.
//! null over-rejects while the block null is close to the nominal level.

use rcsr::diagnostics::{block_count, block_permute, NullControl};
use rcsr::rng::Mt;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, Output};

/// A stationary-ish AR(1) series; innovations are a sum of three uniforms.
fn ar1(n: usize, phi: f64, seed: u32) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut prev = 0.0;
    for i in 0..n {
        let e = (rng.next_f64() + rng.next_f64() + rng.next_f64() - 1.5) * 2.0;
        let v = if i == 0 { e } else { phi * prev + e };
        x.push(v);
        prev = v;
    }
    x
}

/// Lag-1 autocorrelation as the test statistic.
fn lag1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let ma = x[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
    let mb = x[1..].iter().sum::<f64>() / (n - 1) as f64;
    let mut num = 0.0;
    let mut a = 0.0;
    let mut b = 0.0;
    for i in 0..n - 1 {
        let u = x[i] - ma;
        let v = x[i + 1] - mb;
        num += u * v;
        a += u * u;
        b += v * v;
    }
    if a <= 0.0 || b <= 0.0 {
        0.0
    } else {
        num / (a.sqrt() * b.sqrt())
    }
}

/// One-sided permutation p-value for the statistic `-lag1` (lower = more
/// autocorrelated), using the same p-value formula as the engine.
fn p_value(y: &[f64], block_len: usize, reps: usize, seed: u32) -> f64 {
    let observed = -lag1(y);
    let mut rng = Mt::new(seed);
    let mut nulls = Vec::with_capacity(reps);
    for _ in 0..reps {
        let perm = block_permute(y, block_len, &mut rng);
        nulls.push(-lag1(&perm));
    }
    NullControl::summarise(observed, nulls).p_value
}

#[test]
fn block_permutation_is_deterministic_and_keeps_every_row() {
    let y = ar1(100, 0.8, 7);
    let mut a = Mt::new(123);
    let mut b = Mt::new(123);
    let pa = block_permute(&y, 24, &mut a);
    let pb = block_permute(&y, 24, &mut b);
    assert_eq!(pa, pb, "the same seed must reproduce the null exactly");
    let mut sorted = pa.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut original = y.clone();
    original.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(sorted, original, "permutation must keep every value once");
    assert_eq!(block_count(100, 24), 5);
    assert_eq!(block_count(100, 1), 100);
}

#[test]
fn block_null_is_calibrated_on_exchangeable_data_and_does_not_over_reject_ar1() {
    const SERIES: usize = 200;
    const N: usize = 512;
    const REPS: usize = 99;

    // 1. On exchangeable data the block null must have a size close to the
    // nominal 5%: permuting blocks is just another random rearrangement.
    let mut iid_data_rejections = 0usize;
    let mut rng = Mt::new(99);
    for s in 0..SERIES {
        let y: Vec<f64> = (0..N).map(|_| rng.next_f64() - 0.5).collect();
        if p_value(&y, 64, REPS, 30_000 + s as u32) <= 0.05 {
            iid_data_rejections += 1;
        }
    }
    let iid_data_rate = iid_data_rejections as f64 / SERIES as f64;
    eprintln!("iid data, block null rejection rate: {iid_data_rate:.3}");
    assert!(
        iid_data_rate <= 0.12,
        "on exchangeable data the block null should be near nominal 5%, rate={iid_data_rate}"
    );

    // 2. On AR(1) data the i.i.d. shuffle is anti-conservative: it destroys the
    // serial dependence, so the real autocorrelation looks impossible. Blocks
    // of half the series keep the dependence and do not over-reject.
    let mut iid_rejections = 0usize;
    let mut block_rejections = 0usize;
    for s in 0..SERIES {
        let y = ar1(N, 0.9, s as u32 + 1);
        if p_value(&y, 1, REPS, 10_000 + s as u32) <= 0.05 {
            iid_rejections += 1;
        }
        if p_value(&y, N / 2, REPS, 20_000 + s as u32) <= 0.05 {
            block_rejections += 1;
        }
    }
    let iid_rate = iid_rejections as f64 / SERIES as f64;
    let block_rate = block_rejections as f64 / SERIES as f64;
    eprintln!("AR(1): iid rejection rate {iid_rate:.3}, block rejection rate {block_rate:.3}");
    assert!(
        iid_rate > 0.5,
        "the i.i.d. null should over-reject on AR(1) data, rate={iid_rate}"
    );
    assert!(
        block_rate <= 0.12,
        "the block null should stay near the nominal 5%, rate={block_rate}"
    );
    assert!(
        block_rate < iid_rate,
        "block permutation must be less anti-conservative: block={block_rate} iid={iid_rate}"
    );
}

#[test]
fn the_report_records_the_null_method_and_block_count() {
    let c = NullControl::summarise_block(0.1, vec![0.3, 0.2, 0.4], 24, 6);
    assert_eq!(c.method, "block");
    assert_eq!(c.block_len, 24);
    assert_eq!(c.blocks, 6);
    assert_eq!(c.observed, 0.1);
    assert!(c.p_value > 0.0 && c.p_value <= 1.0);
    let iid = NullControl::summarise(0.1, vec![0.3, 0.2]);
    assert_eq!(iid.method, "iid");
    assert_eq!(iid.block_len, 0);
}

struct TempCsv(PathBuf);

impl TempCsv {
    fn new(contents: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rcsr-null-block-{}-{}.csv",
            std::process::id(),
            n
        ));
        std::fs::write(&path, contents).expect("write temp csv");
        TempCsv(path)
    }
    fn as_str(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for TempCsv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn null_block_24_runs_on_a_time_ordered_sample_and_reports_p_and_blocks() {
    let mut text = String::from("t,x,y\n");
    for i in 0..200 {
        let x = (i as f64 * 0.017).sin();
        let y = 1.5 * x + 0.01 * (i as f64 * 0.37).cos();
        let _ = writeln!(text, "{},{:.6},{:.6}", i, x, y);
    }
    let csv = TempCsv::new(&text);
    let out = run(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--split",
        "temporal",
        "--time",
        "t",
        "--null-block",
        "24",
        "--null-runs",
        "3",
        "--generations",
        "2",
        "--pop",
        "20",
        "--format",
        "json",
        "--quiet",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"method\":\"block\""), "json: {stdout}");
    assert!(stdout.contains("\"block_len\":24"), "json: {stdout}");
    assert!(stdout.contains("\"blocks\":"), "json: {stdout}");
    assert!(stdout.contains("\"p_value\":"), "json: {stdout}");
    assert!(stdout.contains("\"observed\":"), "json: {stdout}");
}
