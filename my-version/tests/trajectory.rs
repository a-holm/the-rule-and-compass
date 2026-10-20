//! Archive-level trajectory pin for #87.
//!
//! The initial-SSE reuse was removed because it could flip the LM acceptance
//! test and change the search trajectory even when the metric is not `squared`.
//! This test pins the complete `--out` front for the documented baseline
//! command on a fixed table, so any future trajectory shift fails loudly. The
//! pinned SHA-256 of the file is
//!
//! `e4140058d01882540e877259014b07559bf1b9d1805b4d4e7c776220d18c2992`
//!
//! (the FNV-1a 64 hash is `e73a628e49150567`).

use std::process::Command;

const FRONT: &str = "rank\tcomplexity\ttrain_error\ttest_error\texpression\n\
1\t1\t2.97670764\t3.66924796\t(0.5726 * (0.1305 + x))\n\
2\t2\t2.17852613\t2.42006255\texp((0.5726 * (0.1305 + x)))\n\
3\t3\t1.61569133\t1.58832256\t(3.9597 ^ (0.5726 * (0.1305 + x)))\n\
4\t4\t1.14507499\t1.12247276\t(8.6931 ^ sin((0.5726 * (0.1305 + x))))\n\
5\t5\t0.00000048\t0.00000067\t(3.4928 * (0.7842 + (0.5726 * (0.1305 + x))))\n";

#[test]
fn documented_baseline_front_is_pinned() {
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
            // #141: the pin is a CPU f64 trajectory. A CUDA build now prefers
            // the GPU, whose f32 fitness is not bit-identical, so the
            // reproducibility escape hatch is explicit here.
            "--cpu",
            "--out",
            out.to_str().expect("utf-8 path"),
            "--quiet",
        ])
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
        "e73a628e49150567",
        "the front hash changed"
    );
}
