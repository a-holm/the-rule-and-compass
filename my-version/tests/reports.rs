//! Reports (issue #29): summary, analytic sensitivity and repeated-search
//! benchmark.
//!
//! The CLI tests run the real binary with a tiny search budget. The sensitivity
//! and statistics tests call the library directly so they check the numerics,
//! not just the Markdown scaffolding.

use rcsr::expr::{self, constant, input, node, Op};
use rcsr::{data, diagnostics, fit};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, Output};

/// A temporary CSV that removes itself when the test ends.
struct TempCsv {
    path: PathBuf,
}

impl TempCsv {
    fn new(contents: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rcsr-reports-{}-{}.csv",
            std::process::id(),
            n
        ));
        std::fs::write(&path, contents).expect("write temp csv");
        TempCsv { path }
    }

    fn as_str(&self) -> &str {
        self.path.to_str().expect("utf-8 path")
    }
}

impl Drop for TempCsv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn line_csv(n: usize) -> String {
    let mut text = String::from("x,y\n");
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        let _ = writeln!(text, "{:.6},{:.6}", x, 2.0 * x + 1.0);
    }
    text
}

fn obj_get<'a>(items: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

fn report(path: &str, kind: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data",
        path,
        "--target",
        "y",
        "--vars",
        "x",
        "--report",
        kind,
        "--seed",
        "3",
        "--generations",
        "6",
        "--pop",
        "30",
        "--max-complexity",
        "8",
        "--metric",
        "squared",
    ];
    args.extend_from_slice(extra);
    run(&args)
}

#[test]
fn each_report_is_markdown_with_a_law_errors_and_a_table() {
    let csv = TempCsv::new(&line_csv(40));
    for kind in ["summary", "sensitivity"] {
        let out = report(csv.as_str(), kind, &[]);
        assert!(
            out.status.success(),
            "{kind} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(!text.trim().is_empty(), "{kind} produced no report");
        assert!(text.contains("## Chosen law") || text.contains("# Variable sensitivity"));
        assert!(text.contains("y = "), "{kind} is missing the law:\n{text}");
        assert!(text.contains("train"), "{kind} is missing error numbers");
        assert!(text.contains('|'), "{kind} is missing a table header");
    }

    // Benchmark asks for a different budget and repeat count.
    let out = run(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--report",
        "benchmark",
        "--report-repeats",
        "2",
        "--generations",
        "2",
        "--pop",
        "12",
        "--max-complexity",
        "8",
        "--seed",
        "3",
        "--metric",
        "squared",
    ]);
    assert!(
        out.status.success(),
        "benchmark failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("# Search benchmark"), "{text}");
    assert!(text.contains("| complexity |"), "{text}");
    assert!(text.contains("train median"), "{text}");
    assert!(text.contains("test median"), "{text}");
    assert!(text.contains("y = "), "{text}");
    assert!(text.contains("Wall clock"), "{text}");
}

#[test]
fn report_out_writes_a_file_and_keeps_stdout_clean() {
    let csv = TempCsv::new(&line_csv(40));
    let out_path = std::env::temp_dir().join(format!(
        "rcsr-report-{}-summary.md",
        std::process::id()
    ));
    let out_path_str = out_path.to_str().expect("utf-8");
    let out = report(csv.as_str(), "summary", &["--report-out", out_path_str]);
    assert!(
        out.status.success(),
        "summary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("## Chosen law"),
        "the report should go to --report-out, not stdout"
    );
    let text = std::fs::read_to_string(&out_path).expect("report file");
    assert!(text.contains("## Chosen law"), "{text}");
    assert!(text.contains("| rank | complexity |"), "{text}");
    assert!(text.contains("y = "), "{text}");
    let _ = std::fs::remove_file(&out_path);
}

/// #134: the JSON timing block is complete, non-overlapping and its phases
/// account for the independently measured wall total within 1%.
#[test]
fn json_timing_phases_sum_to_the_measured_wall() {
    let csv = TempCsv::new(&line_csv(1200));
    let out = run(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "3",
        "--generations",
        "4",
        "--pop",
        "24",
        "--max-complexity",
        "9",
        "--baseline-samples",
        "0",
        "--json",
    ]);
    assert!(
        out.status.success(),
        "json run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let parsed = parse_json(String::from_utf8_lossy(&out.stdout).trim())
        .expect("stdout must be valid JSON");
    let Json::Obj(items) = parsed else {
        panic!("top-level JSON must be an object")
    };
    let timing = match obj_get(&items, "timing") {
        Some(Json::Obj(t)) => t,
        _ => panic!("JSON is missing the timing object"),
    };
    let get = |key: &str| match obj_get(timing, key) {
        Some(Json::Num(v)) => *v,
        other => panic!("timing.{key} missing or not a number: {other:?}"),
    };
    let prep = get("prep_seconds");
    let search = get("search_seconds");
    let baselines = get("baselines_seconds");
    let diagnostics = get("diagnostics_seconds");
    let report = get("report_seconds");
    let total = get("total_seconds");
    assert!(
        prep >= 0.0 && search >= 0.0 && baselines >= 0.0 && diagnostics >= 0.0 && report >= 0.0,
        "phases must be non-negative"
    );
    assert!(total > 0.0, "total wall must be positive");
    assert!(
        (prep + search + baselines + diagnostics + report - total).abs() <= 0.01 * total,
        "phases must sum to total wall within 1%: {prep}+{search}+{baselines}+{diagnostics}+{report} vs {total}"
    );
    // The uncapped 1200-row k-NN baseline is real work, not a zero placeholder.
    assert!(
        baselines > 0.0,
        "an uncapped baseline must report a positive baselines_seconds, got {baselines}"
    );
    // Compatibility: `elapsed_seconds` is the search phase, and it is explicit.
    match obj_get(&items, "elapsed_seconds") {
        Some(Json::Num(v)) => assert_eq!(*v, search),
        other => panic!("elapsed_seconds missing or not a number: {other:?}"),
    }
    }

/// #497: the CLI's default baseline is exact up to the row threshold, an
/// explicit N samples for every metric, and 0 asks for every row. The JSON
/// carries the sampling fields.
#[test]
fn cli_baseline_default_is_exact_below_the_threshold_and_explicit_samples_any_metric() {
    let csv = TempCsv::new(&line_csv(2000));
    let base = [
        "--data", csv.as_str(), "--target", "y", "--vars", "x",
        "--seed", "3", "--generations", "1", "--pop", "8", "--json",
    ];
    let parse = |out: &Output| {
        let parsed = parse_json(String::from_utf8_lossy(&out.stdout).trim())
            .expect("valid JSON");
        let Json::Obj(items) = parsed else { panic!("object") };
        let baselines = match obj_get(&items, "baselines") {
            Some(Json::Obj(v)) => v,
            _ => panic!("baselines missing"),
        };
        let knn = match obj_get(baselines, "knn") {
            Some(Json::Obj(v)) => v,
            _ => panic!("knn missing"),
        };
        let num = |k: &str| match obj_get(knn, k) {
            Some(Json::Num(v)) => *v,
            other => panic!("knn.{k} missing: {other:?}"),
        };
        let capped = match obj_get(knn, "capped") {
            Some(Json::Bool(v)) => *v,
            other => panic!("knn.capped missing: {other:?}"),
        };
        (num("reference_rows"), num("train_query_rows"), capped, obj_get(knn, "seconds").is_some())
    };
    let out = run(&base);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (reference, queries, capped, has_seconds) = parse(&out);
    assert_eq!(reference, 1500.0, "default test split leaves 1500 training rows");
    assert_eq!(queries, 1500.0, "the default is exact below the threshold");
    assert!(!capped);
    assert!(has_seconds, "JSON must report the baseline seconds");

    for extra in [vec!["--baseline-samples", "512"], vec!["--baseline-samples", "64", "--metric", "median"]] {
        let mut args = base.to_vec();
        args.extend(extra.iter().copied());
        let out = run(&args);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let (_, queries, capped, _) = parse(&out);
        let want: f64 = extra[1].parse().unwrap();
        assert_eq!(queries, want, "an explicit N must sample any metric");
        assert!(capped);
    }

    let mut args = base.to_vec();
    args.extend(["--baseline-samples", "0"]);
    let out = run(&args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (_, queries, capped, _) = parse(&out);
    assert_eq!(queries, 1500.0, "0 must mean every row");
    assert!(!capped);
}

#[test]
fn json_stays_valid_when_a_report_is_requested() {
    let csv = TempCsv::new(&line_csv(40));
    let out = report(csv.as_str(), "summary", &["--json"]);
    assert!(
        out.status.success(),
        "json run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed = parse_json(stdout.trim()).expect("stdout must be valid JSON");
    match parsed {
        Json::Obj(items) => {
            assert!(
                matches!(obj_get(&items, "tool"), Some(Json::Str(s)) if s == "rcsr"),
                "JSON is missing the tool field"
            );
            match obj_get(&items, "pareto") {
                Some(Json::Arr(rows)) => assert!(!rows.is_empty(), "empty Pareto front"),
                _ => panic!("JSON is missing the Pareto array"),
            }
            assert!(
                matches!(obj_get(&items, "elapsed_seconds"), Some(Json::Num(v)) if *v >= 0.0),
                "JSON is missing elapsed_seconds"
            );
            match obj_get(&items, "converged_early") {
                Some(Json::Bool(v)) => assert!(*v || !*v, "bool field parsed"),
                _ => panic!("JSON is missing converged_early"),
            }
        }
        _ => panic!("top-level JSON must be an object"),
    }
    // The Markdown went to stderr, where it cannot corrupt the stream.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("## Chosen law"),
        "the report should be on stderr in JSON mode:\n{stderr}"
    );
}

#[test]
fn unknown_report_type_is_a_usage_error() {
    let out = run(&["--report", "bogus"]);
    assert_eq!(
        out.status.code(),
        Some(2),
        "unknown --report must exit 2, got {:?}: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn sensitivity_puts_the_driving_input_first() {
    // y = 3*x1, with x2 varying just as widely but absent from the law.
    let tree = node(Op::Mul, vec![constant(3.0), input(0)]);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let x1 = -2.0 + 4.0 * i as f64 / 49.0;
        let x2 = ((i * 37 % 50) as f64 - 25.0) / 12.5;
        x.push(vec![x1, x2]);
        y.push(3.0 * x1);
    }
    let report = diagnostics::sensitivity(
        &tree,
        &data::Sample {
            x,
            y,
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        },
        &[1.0, 1.0],
    );
    assert_eq!(report.rows[0].input, 0);
    assert!(
        report.rows[0].mean_abs_elasticity > 100.0 * report.rows[1].mean_abs_elasticity,
        "{:?}",
        report.rows
    );
    assert_eq!(report.rows[1].mean_abs_elasticity, 0.0);
}

#[test]
fn analytic_derivative_matches_central_difference() {
    // f(x1, x2) = sin(x1) + x1 * x2^2, so neither derivative is constant.
    let tree = node(
        Op::Add,
        vec![
            node(Op::Sin, vec![input(0)]),
            node(
                Op::Mul,
                vec![input(0), node(Op::Mul, vec![input(1), input(1)])],
            ),
        ],
    );
    let x = vec![0.7, -1.3];
    let analytic = fit::input_gradients(&tree, &x);
    assert_eq!(analytic.len(), 2);
    let h = 1e-6;
    for j in 0..2 {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[j] += h;
        xm[j] -= h;
        let fd = (expr::eval(&tree, &xp) - expr::eval(&tree, &xm)) / (2.0 * h);
        assert!(
            (analytic[j] - fd).abs() < 1e-6,
            "input {j}: analytic {} vs finite difference {}",
            analytic[j],
            fd
        );
    }
}

#[test]
fn median_and_iqr_are_exact_on_hand_made_numbers() {
    assert_eq!(diagnostics::median(&[3.0, 1.0, 2.0]), 2.0);
    assert_eq!(diagnostics::median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    assert!((diagnostics::iqr(&[1.0, 2.0, 3.0, 4.0]) - 1.5).abs() < 1e-12);

    let repeats = vec![
        vec![diagnostics::BenchmarkPoint {
            complexity: 3,
            train_error: 2.0,
            test_error: 4.0,
        }],
        vec![diagnostics::BenchmarkPoint {
            complexity: 3,
            train_error: 4.0,
            test_error: 8.0,
        }],
        vec![diagnostics::BenchmarkPoint {
            complexity: 3,
            train_error: 6.0,
            test_error: 12.0,
        }],
    ];
    let rows = diagnostics::benchmark_by_complexity(&repeats);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].runs, 3);
    assert!((rows[0].train_median - 4.0).abs() < 1e-12);
    assert!((rows[0].train_iqr - 2.0).abs() < 1e-12);
    assert!((rows[0].test_median - 8.0).abs() < 1e-12);
    assert!((rows[0].test_iqr - 4.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// A deliberately small JSON parser, so the test proves the stream parses
// rather than merely looking like JSON. It is not a general-purpose library.
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\n' | b'\r' | b'\t') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", c as char, self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.b.get(self.i).copied() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(c) => Err(format!("unexpected byte '{}' at {}", c as char, self.i)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn literal(&mut self, s: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(())
        } else {
            Err(format!("expected {} at byte {}", s, self.i))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(items));
        }
        loop {
            let k = self.string()?;
            self.eat(b':')?;
            let v = self.value()?;
            items.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(items));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = *self
                .b
                .get(self.i)
                .ok_or_else(|| "unterminated string".to_string())?;
            self.i += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = *self
                        .b
                        .get(self.i)
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            out.push(char::from_u32(hi).unwrap_or('\u{fffd}'));
                        }
                        other => {
                            return Err(format!("bad escape '\\{}'", other as char));
                        }
                    }
                }
                c if c < 0x20 => return Err("control character in string".to_string()),
                _ => {
                    // All non-ASCII in these documents is UTF-8; collect bytes
                    // until the next ASCII byte and decode them as a unit.
                    let start = self.i - 1;
                    let mut end = self.i;
                    while end < self.b.len() && self.b[end] & 0xc0 == 0x80 {
                        end += 1;
                    }
                    if end > self.i {
                        self.i = end;
                    }
                    let piece = std::str::from_utf8(&self.b[start..end])
                        .map_err(|_| "invalid UTF-8 in string".to_string())?;
                    out.push_str(piece);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = *self
                .b
                .get(self.i)
                .ok_or_else(|| "short \\u escape".to_string())?;
            self.i += 1;
            v = v * 16
                + match c {
                    b'0'..=b'9' => (c - b'0') as u32,
                    b'a'..=b'f' => (c - b'a' + 10) as u32,
                    b'A'..=b'F' => (c - b'A' + 10) as u32,
                    _ => return Err("bad hex digit".to_string()),
                };
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i])
            .map_err(|_| "invalid number".to_string())?;
        let v = text
            .parse::<f64>()
            .map_err(|_| format!("invalid number '{}'", text))?;
        Ok(Json::Num(v))
    }
}

fn parse_json(s: &str) -> Result<Json, String> {
    let mut p = Parser {
        b: s.as_bytes(),
        i: 0,
    };
    let v = p.value()?;
    p.ws();
    if p.i == p.b.len() {
        Ok(v)
    } else {
        Err(format!("trailing input at byte {}", p.i))
    }
}
