//! #284: checks shared by the enrich, walk-forward and ladder phase-ledger tests.
#![allow(dead_code)]

const STAGES: [&str; 9] = [
    "vary", "eval", "fit", "select", "archive", "features", "pnl", "io", "host_other",
];
const EPS: f64 = 1e-9;

fn number(body: &str, key: &str) -> f64 {
    let tag = format!("\"{key}\":");
    let at = body.find(&tag).unwrap_or_else(|| panic!("ledger key {key} missing in {body}")) + tag.len();
    let rest = &body[at..];
    let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].parse().unwrap_or_else(|_| panic!("ledger key {key} is not a number: {}", &rest[..end]))
}

/// Parsed run-clock numbers of a command's `ledger` object.
pub struct Ledger {
    pub wall: f64,
    pub busy: f64,
    pub stages: Vec<(&'static str, f64)>,
}

impl Ledger {
    pub fn stage(&self, name: &str) -> f64 {
        self.stages.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap()
    }
}

/// Parse the `ledger` object of `json` and assert the closure rules: every
/// stage is named, the stages sum to no more than the wall, device-busy is
/// within the wall, and `positive` stages carry time.
pub fn check(json: &str, command: &str, positive: &[&str]) -> Ledger {
    let at = json.find("\"ledger\":{").unwrap_or_else(|| panic!("no ledger object in {json}"));
    let body = &json[at..];
    assert!(body.contains(&format!("\"command\":\"{command}\"")), "{body}");
    let stages_at = body.find("\"stages\":{").unwrap();
    let stages_body = &body[stages_at..stages_at + body[stages_at..].find('}').unwrap()];
    let stages: Vec<(&'static str, f64)> =
        STAGES.iter().map(|s| (*s, number(stages_body, &format!("{s}_s")))).collect();
    let wall = number(body, "wall_s");
    let sum = number(body, "stage_sum_s");
    let busy = number(body, "device_busy_s");
    assert!(wall > 0.0, "wall {wall}");
    let parts: f64 = stages.iter().map(|(_, v)| v).sum();
    assert!((parts - sum).abs() <= EPS, "stage sum {parts} vs stage_sum_s {sum}");
    assert!(sum <= wall + EPS, "stages {sum} exceed wall {wall}");
    assert!(busy >= 0.0 && busy <= wall + EPS, "device busy {busy} outside wall {wall}");
    for (name, v) in &stages {
        assert!(*v >= 0.0, "{name} = {v}");
    }
    for name in positive {
        let v = stages.iter().find(|(n, _)| n == name).unwrap().1;
        assert!(v > 0.0, "stage {name} must carry time: {body}");
    }
    Ledger { wall, busy, stages }
}
