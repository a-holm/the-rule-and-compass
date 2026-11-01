//! Archive-level trajectory pin for #87.
//!
//! The initial-SSE reuse was removed because it could flip the LM acceptance
//! test and change the search trajectory even when the metric is not `squared`.
//! This test pins the complete `--out` front for the documented baseline
//! command on a fixed table, so any future trajectory shift fails loudly. The
//! pinned front's FNV-1a 64 hash is `3e063cac13d4138a`. #143 replaced the
//! simplifier with the sound normal form and canonical `Add`/`Mul` ordering, so
//! the documented front was re-measured on the same command. #117 moved the
//! default `absolute` metric onto L1 fitting, so this documented baseline was
//! re-measured; `squared` remains bit-identical and is pinned separately in
//! `fit.rs`.

use std::process::Command;

const FRONT: &str = "rank\tcomplexity\ttrain_error\ttest_error\texpression\n\
1\t1\t2.97670764\t3.66924796\t(0.5726 * (0.1305 + x))\n\
2\t2\t2.17852613\t2.42006255\texp((0.5726 * (0.1305 + x)))\n\
3\t3\t1.61548477\t1.58574957\t(3.9826 ^ (0.5726 * (0.1305 + x)))\n\
4\t4\t1.12695731\t1.05986720\tsinh((1.4645 + (0.5726 * (0.1305 + x))))\n\
5\t5\t0.00000045\t0.00000067\t(2.7389 + (2 * (0.1305 + x)))\n";

#[test]
fn documented_baseline_front_is_pinned() {
    pinned_front(None);
}

/// #463: the serial path and a many-worker run give the same front.
#[test]
fn documented_baseline_front_is_pinned_at_one_and_eight_threads() {
    pinned_front(Some("1"));
    pinned_front(Some("8"));
}

fn pinned_front(threads: Option<&str>) {
    let out = std::env::temp_dir().join(format!(
        "rcsr-trajectory-{}.tsv",
        std::process::id()
    ));
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            data,
            "--target",
            "y",
            "--vars",
            "x",
            "--seed",
            "3",
            "--islands",
            "1",
            "--pop",
            "60",
            "--generations",
            "20",
            // #76: this pin is the pre-front-feedback trajectory, so the
            // feature is explicitly off here.
            "--no-front-parents",
            "--no-prune",
            // #141: the pin is a CPU f64 trajectory. A CUDA build now prefers
            // the GPU, whose f32 fitness is not bit-identical, so the
            // reproducibility escape hatch is explicit here.
            "--cpu",
            "--out",
            out.to_str().expect("utf-8 path"),
            "--quiet",
        ])
        .args(threads.map(|t| vec!["--threads", t]).unwrap_or_default())
        .output()
        .expect("run rcsr");
    assert!(
        output.status.success(),
        "rcsr failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let got = std::fs::read_to_string(&out).expect("read the front");
    let _ = std::fs::remove_file(&out);

    assert_eq!(
        got, FRONT,
        "the search trajectory changed; do not adjust this pin without a new \
         baseline measurement"
    );

    // A compact hash makes the pin quotable in review. The initial-SSE reuse
    // was reverted because it changed the LM acceptance test on the reported
    // baseline table; this fixture is the fixed table available in the repo.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in got.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    assert_eq!(
        format!("{hash:016x}"),
        "3e063cac13d4138a",
        "the front hash changed"
    );
}
