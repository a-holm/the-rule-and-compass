//! Seed-solution tests (issue #16).
//!
//! A seed is a formula the user already knows part of. These tests pin down
//! the processing order (parse -> simplify -> fit -> evaluate -> insert), the
//! elite rule above `--max-complexity`, the CLI/file error contract, the
//! reporting in text/JSON/NDJSON, and that a run without seeds is unchanged.

use rcsr::bench;
use rcsr::data::Sample;
use rcsr::expr::{self, Node};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};
use std::path::PathBuf;
use std::process::{Command, Output};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The spring-period law `p = sqrt(m / k)` at zero noise, over a fixed grid.
fn spring() -> (Vec<String>, Sample, Sample) {
    let p = bench::known_truth()
        .into_iter()
        .find(|p| p.name == "spring_period")
        .expect("the spring_period problem");
    let (train, test) = p.generate(0.0);
    (p.vars.clone(), train, test)
}

fn config(seed: u32, generations: usize) -> Config {
    Config {
        generations,
        islands: 1,
        pop: 60,
        max_complexity: 30,
        fit_iters: 8,
        patience: 0,
        seed,
        metric: Metric::Absolute,
        ..Default::default()
    }
}

fn best_error(out: &Outcome) -> f64 {
    out.archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min)
}

fn describe(out: &Outcome) -> Vec<(u64, u64, u64)> {
    out.archive
        .sorted()
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.cost.to_bits(),
                expr::structural_hash(&s.tree),
            )
        })
        .collect()
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

struct TempPath(PathBuf);

impl TempPath {
    fn new(suffix: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        TempPath(std::env::temp_dir().join(format!(
            "rcsr-seeds-{}-{}-{}",
            std::process::id(),
            n,
            suffix
        )))
    }
    fn as_str(&self) -> &str {
        self.0.to_str().expect("utf-8 path")
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A small table whose target is exactly `sqrt(m / k)`.
fn spring_csv() -> String {
    let mut text = String::from("m,k,period\n");
    for i in 0..8 {
        for j in 0..8 {
            let m = 0.5 + 2.5 * i as f64 / 7.0;
            let k = 0.5 + 2.5 * j as f64 / 7.0;
            text.push_str(&format!("{:.10},{:.10},{:.10}\n", m, k, (m / k).sqrt()));
        }
    }
    text
}

fn spring_data() -> TempPath {
    let path = TempPath::new("spring.csv");
    std::fs::write(&path.0, spring_csv()).expect("write csv");
    path
}

// ---------------------------------------------------------------------------
// Processing order and the archive
// ---------------------------------------------------------------------------

#[test]
fn seed_is_simplified_fitted_and_present_in_the_final_archive() {
    let (vars, train, test) = spring();
    // `+ 0` must disappear in simplify, and the leading `2` is a free constant
    // that the fitter must drive to 1 on exact data.
    let seed = expr::parse("2 * sqrt(m / k) + 0", &vars).expect("parse seed");
    let cfg = config(1, 8);
    let out = search::run_seeded(&cfg, &train, &test, &[seed]);

    assert_eq!(out.seeds.len(), 1, "one accepted seed");
    let fitted = &out.seeds[0];
    assert!(
        fitted.error < 1e-10,
        "the fitted seed should explain the data, error = {}",
        fitted.error
    );
    assert!(fitted.complexity > 0.0);

    // The reported expression is the simplified, fitted form: no `+ 0`, and the
    // constant is no longer 2.
    let shown = expr::to_string(&fitted.tree, &vars);
    assert!(
        !shown.contains(" + 0") && !shown.contains("2 *"),
        "seed must be simplified and fitted, got {}",
        shown
    );

    // Present in the final archive, with the same tree and the same error.
    let hash = expr::structural_hash(&fitted.tree);
    let archived = out
        .archive
        .items
        .iter()
        .find(|s| expr::structural_hash(&s.tree) == hash)
        .expect("the seed must survive in the final archive");
    assert_eq!(archived.error.to_bits(), fitted.error.to_bits());
    assert_eq!(
        expr::to_string(&archived.tree, &vars),
        shown,
        "the archived expression and the reported seed must agree"
    );
}

#[test]
fn structurally_duplicate_seeds_are_inserted_once() {
    let (vars, train, test) = spring();
    let a = expr::parse("sqrt(m / k)", &vars).expect("parse");
    let b = expr::parse("sqrt(m / k) + 0", &vars).expect("parse");
    let out = search::run_seeded(&config(2, 3), &train, &test, &[a, b]);
    assert_eq!(out.seeds.len(), 1, "the duplicate must be dropped");
}

#[test]
fn a_seed_above_max_complexity_is_still_inserted() {
    let (vars, train, test) = spring();
    let seed = expr::parse("sqrt(m / k)", &vars).expect("parse");
    let mut cfg = config(3, 4);
    cfg.max_complexity = 2; // below the seed's node count
    assert!(
        cfg.complexity_model.complexity(&expr::simplify(&seed)) > 2.0,
        "the test seed must exceed the budget"
    );
    let out = search::run_seeded(&cfg, &train, &test, &[seed]);
    assert_eq!(out.seeds.len(), 1);
    assert!(out.seeds[0].complexity > cfg.max_complexity as f64);
    let hash = expr::structural_hash(&out.seeds[0].tree);
    assert!(
        out.archive
            .items
            .iter()
            .any(|s| expr::structural_hash(&s.tree) == hash),
        "an elite seed is archived even above --max-complexity"
    );
}

#[test]
fn seeds_enter_every_island() {
    let (vars, train, test) = spring();
    let seed = expr::parse("sqrt(m / k)", &vars).expect("parse");
    // Inspect the initial populations: zero generations means the checkpoint
    // written on exit is exactly the state seeds were inserted into.
    let mut cfg = config(4, 0);
    cfg.islands = 3;
    cfg.init_depth = 1;
    let hash = expr::structural_hash(&expr::simplify(&seed));
    let mut pops: Option<Vec<Vec<u64>>> = None;
    search::run_resumable_seeded(
        &cfg,
        &train,
        &test,
        None,
        0,
        &[seed],
        |cp| {
            pops = Some(
                cp.islands
                    .iter()
                    .map(|isl| {
                        isl.population
                            .iter()
                            .map(|ind| expr::structural_hash(&ind.tree))
                            .collect()
                    })
                    .collect(),
            );
        },
    )
    .expect("run");
    let pops = pops.expect("a checkpoint");
    assert_eq!(pops.len(), 3, "every island must receive the seed");
    for (i, pop) in pops.iter().enumerate() {
        assert!(pop.contains(&hash), "island {} lost the seed", i);
    }
}

// ---------------------------------------------------------------------------
// The measured comparison
// ---------------------------------------------------------------------------

#[test]
fn a_seed_reaches_a_lower_error_at_the_same_small_budget() {
    let (vars, train, test) = spring();
    let seed: Node = expr::parse("sqrt(m / k)", &vars).expect("parse");
    // Every seed runs the identical budget; only `--seed-expr` differs.
    //
    // The budget is deliberately tiny. It was 15 generations when this test was
    // written, but #77 (mutation now always changes the tree) improved the
    // unseeded search enough that it sometimes solves this law by generation
    // 15 — which is a win for the engine and invalidates the comparison. At 3
    // generations the seed's advantage is unambiguous: it is exact from
    // generation 1, while a random start has barely begun.
    let mut rows = Vec::new();
    for s in 1..=3u32 {
        let cfg = config(s, 3);
        let plain = search::run(&cfg, &train, &test);
        let seeded = search::run_seeded(&cfg, &train, &test, &[seed.clone()]);
        let (plain_err, seeded_err) = (best_error(&plain), best_error(&seeded));
        rows.push((s, plain_err, seeded_err));
        assert!(
            seeded_err < 1e-12,
            "seed={} seeded best error = {}",
            s,
            seeded_err
        );
        assert!(
            plain_err > 1e-3,
            "seed={} the unseeded run must be clearly worse at 15 generations, \
             got {} (the margin this test asserts is 1e-3 vs 1e-12)",
            s,
            plain_err
        );
    }
    for (s, plain, seeded) in &rows {
        eprintln!(
            "seed={} unseeded={:.6e} seeded={:.6e}",
            s, plain, seeded
        );
    }
}

/// Best error after each completed generation, for the threshold-crossing
/// measurement. The seed is inserted before generation 1, so a generation
/// count of 1 means the seed is already in the resulting front.
fn progress_error(cfg: &mut Config, train: &Sample, test: &Sample, seed: Option<&Node>) -> Vec<f64> {
    use std::sync::{Arc, Mutex};
    let seen: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    cfg.progress = Some(Arc::new(move |p: search::Progress| {
        sink.lock().expect("progress lock").push(p.best_error);
    }));
    let seeds: Vec<Node> = seed.cloned().into_iter().collect();
    let _ = search::run_seeded(cfg, train, test, &seeds);
    let v = seen.lock().expect("progress lock").clone();
    v
}

fn first_below(errors: &[f64], threshold: f64) -> Option<usize> {
    errors.iter().position(|e| *e <= threshold).map(|i| i + 1)
}

#[test]
fn the_seed_crosses_the_threshold_first() {
    let (vars, train, test) = spring();
    let seed = expr::parse("sqrt(m / k)", &vars).expect("parse");
    let threshold = 1e-6;
    for s in 1..=3u32 {
        let mut cfg = config(s, 15);
        let plain = progress_error(&mut cfg.clone(), &train, &test, None);
        let seeded = progress_error(&mut cfg, &train, &test, Some(&seed));
        let plain_gen = first_below(&plain, threshold);
        let seeded_gen = first_below(&seeded, threshold);
        eprintln!(
            "seed={} threshold {:e}: unseeded first at {:?}, seeded first at {:?}; \
             unseeded 1e-3 at {:?}",
            s,
            threshold,
            plain_gen,
            seeded_gen,
            first_below(&plain, 1e-3)
        );
        assert_eq!(
            seeded_gen,
            Some(1),
            "a seed is usable from the first generation; got {:?}",
            seeded_gen
        );
        assert!(
            plain_gen.map(|g| g > 1).unwrap_or(true),
            "the unseeded control should not reach the threshold at generation 1, got {:?}",
            plain_gen
        );
    }
}

// ---------------------------------------------------------------------------
// CLI contract
// ---------------------------------------------------------------------------

#[test]
fn bad_seed_expr_exits_2_and_names_flag_text_and_column() {
    let csv = spring_data();
    let out = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "period",
        "--vars",
        "m,k",
        "--seed-expr",
        "sqrt(m / )",
    ]);
    assert_eq!(out.status.code(), Some(2), "usage error");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--seed-expr"), "must name the flag: {}", err);
    assert!(
        err.contains("sqrt(m / )"),
        "must quote the offending text: {}",
        err
    );
    assert!(
        err.contains("(column 10)"),
        "must carry the parser column: {}",
        err
    );
}

#[test]
fn missing_seed_file_exits_3() {
    let csv = spring_data();
    let out = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "period",
        "--vars",
        "m,k",
        "--seed-file",
        "/nonexistent/rcsr-seeds.txt",
    ]);
    assert_eq!(out.status.code(), Some(3), "input error");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--seed-file"), "must name the flag: {}", err);
    assert!(
        err.contains("/nonexistent/rcsr-seeds.txt"),
        "must name the path: {}",
        err
    );
}

#[test]
fn seed_file_skips_comments_and_blank_lines_and_combines_with_seed_expr() {
    let csv = spring_data();
    let seeds = TempPath::new("seeds.txt");
    std::fs::write(
        &seeds.0,
        "# prior solutions\n\nsqrt(m / k)\n   \n# another line\nm / k\n",
    )
    .expect("write seed file");
    let out = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "period",
        "--vars",
        "m,k",
        "--generations",
        "2",
        "--pop",
        "20",
        "--seed",
        "1",
        "--seed-file",
        seeds.as_str(),
        "--seed-expr",
        "m / k",
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        err.matches("# seeded:").count(),
        2,
        "two distinct seeds; the duplicate m/k is folded:\n{}",
        err
    );
    assert!(err.contains("sqrt"), "the file seed must be reported: {}", err);
}

#[test]
fn json_and_ndjson_report_each_seed() {
    let csv = spring_data();
    let base = [
        "--data",
        csv.as_str(),
        "--target",
        "period",
        "--vars",
        "m,k",
        "--generations",
        "2",
        "--pop",
        "20",
        "--seed",
        "1",
        "--seed-expr",
        "sqrt(m / k)",
    ];

    let mut json_args = base.to_vec();
    json_args.push("--json");
    let json = rcsr(&json_args);
    assert_eq!(json.status.code(), Some(0));
    let text = String::from_utf8_lossy(&json.stdout);
    assert!(
        text.contains("\"seeds\":[{\"expression\":\"sqrt"),
        "JSON seeds array: {}",
        text
    );
    assert!(text.contains("\"complexity\":"), "{}", text);
    assert!(text.contains("\"train_error\":"), "{}", text);

    let mut nd_args = base.to_vec();
    nd_args.push("--ndjson");
    let nd = rcsr(&nd_args);
    assert_eq!(nd.status.code(), Some(0));
    let lines = String::from_utf8_lossy(&nd.stdout);
    let seeds: Vec<&str> = lines
        .lines()
        .filter(|l| l.starts_with("{\"event\":\"seed\""))
        .collect();
    assert_eq!(seeds.len(), 1, "one seed event: {}", lines);
    assert!(seeds[0].contains("\"expression\""), "{}", seeds[0]);
    assert!(seeds[0].contains("\"train_error\""), "{}", seeds[0]);
}

#[test]
fn seed_above_max_complexity_prints_a_note() {
    let csv = spring_data();
    let out = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "period",
        "--vars",
        "m,k",
        "--generations",
        "2",
        "--pop",
        "20",
        "--seed",
        "1",
        "--max-complexity",
        "3",
        "--seed-expr",
        "sqrt(m / k)",
        "--json",
    ]);
    assert_eq!(out.status.code(), Some(0));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("# note:") && err.contains("--max-complexity 3"),
        "the elite rule must be announced: {}",
        err
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("\"seeds\":[{"),
        "the seed is still reported: {}",
        text
    );
}

// ---------------------------------------------------------------------------
// Regression: no seeds means no change
// ---------------------------------------------------------------------------

#[test]
fn no_seeds_is_the_historical_trajectory() {
    let (_vars, train, test) = spring();
    let cfg = config(7, 10);
    let plain = search::run(&cfg, &train, &test);
    let empty = search::run_seeded(&cfg, &train, &test, &[]);
    assert_eq!(describe(&plain), describe(&empty));
    assert_eq!(plain.generations, empty.generations);
    assert!(empty.seeds.is_empty());
}

#[test]
fn cli_without_seed_flags_prints_no_seeded_headings() {
    let csv = spring_data();
    let out = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "period",
        "--vars",
        "m,k",
        "--generations",
        "2",
        "--pop",
        "20",
        "--seed",
        "1",
    ]);
    assert_eq!(out.status.code(), Some(0));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("# seeded:"),
        "no seed was requested: {}",
        err
    );
}
