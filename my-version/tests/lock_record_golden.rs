//! #1130: without a lock record every sidecar is byte-identical to the one the
//! build wrote before the record existed. The hashes pin the label and feature
//! reports of `enrich`, the archive snapshot, the base and panel manifests and
//! the panel schema of a small fixture, with the directory and file times
//! normalised.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/lock_record.rs"]
mod lr;
#[path = "common/long_horizon.rs"]
mod lh;
#[path = "common/scratch.rs"]
mod scratch;

use lr::{run, s, Pipeline};
use std::path::Path;

fn normalise(text: &str, dir: &Path) -> String {
    let text = text.replace(&s(dir), "<dir>");
    let mut out = String::new();
    let mut rest = text.as_str();
    while let Some(at) = rest.find("\"mtime\":") {
        let end = at + "\"mtime\":".len();
        out.push_str(&rest[..end]);
        out.push('0');
        rest = rest[end..].trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
}

fn sha(text: &str) -> String {
    rcsr::manifest::sha256_hex(text.as_bytes())
}

/// The `features` and `long_labels` members of an enrich report (the rest holds timings).
fn enrich_report(dir: &Path) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let data = lh::bars(lr::T0, 1000, 0.0);
    let other = lh::bars(lr::T0, 1000, 1.0);
    let src = s(&dir.join("in.rcsb"));
    lh::write_store(&src, &[("AAA", &data), ("BBB", &other)]);
    let out = s(&dir.join("out.rcsb"));
    let (code, _, err) = run(&["enrich", "--data", &src, "--out", &out, "--cpu", "--features", "ret1", "--label", &lh::src(1, "")]);
    assert_eq!(code, 0, "{err}");
    let text = std::fs::read_to_string(format!("{out}.enrich.json")).unwrap();
    let from = text.find("\"features\":[").unwrap();
    let to = text.find(",\n  \"plan\":").unwrap();
    text[from..to].to_string()
}

fn observed() -> Vec<(String, String)> {
    let sc = scratch::Scratch::new("lock-record-golden");
    let dir = sc.path();
    let p = Pipeline::new(&dir.join("pipe"), false);
    p.build(None);
    let mut rows = vec![("enrich.report".to_string(), sha(&normalise(&enrich_report(&dir.join("enrich")), dir)))];
    for (name, f) in ["snapshot", "base.manifest", "panel.manifest", "panel.schema"].iter().zip(p.sidecars()) {
        rows.push((name.to_string(), sha(&normalise(&std::fs::read_to_string(&f).unwrap(), dir))));
    }
    rows
}

#[test]
fn sidecars_without_a_record_equal_the_hashes_taken_before_the_record_existed() {
    let golden = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lock_record/golden.sha256")).unwrap_or_default();
    let want: Vec<(String, String)> = golden.lines().filter_map(|l| l.split_once(' ')).map(|(a, b)| (a.to_string(), b.to_string())).collect();
    let got = observed();
    let text: String = got.iter().map(|(a, b)| format!("{a} {b}\n")).collect();
    assert_eq!(got, want, "observed hashes:\n{text}");
}
