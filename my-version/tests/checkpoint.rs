//! Save/resume tests (issue #17).
//!
//! The central guarantee is trajectory determinism: an uninterrupted run and
//! a run split at an arbitrary generation boundary must produce the same
//! archive, bit for bit. The tests also pin down exact tree/RNG round-trips and
//! the refusal of corrupt checkpoints and mismatched data.

use rcsr::checkpoint::{decode_tree, encode_tree, Checkpoint};
use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, node, Op};
use rcsr::rng::Mt;
use rcsr::search::{self, Config, Outcome};
use std::path::PathBuf;
use std::process::{Command, Output};

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b + 1.0 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new() }
}

fn config(seed: u32) -> Config {
    Config {
        generations: 8,
        islands: 2,
        pop: 30,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        ..Default::default()
    }
}

/// A comparable view of a front: exact error and cost bits, node count,
/// structural hash (which includes every constant bit) and printed form.
fn describe(out: &Outcome) -> Vec<(u64, u64, usize, u64, String)> {
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.cost.to_bits(),
                s.complexity,
                expr::structural_hash(&s.tree),
                expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]),
            )
        })
        .collect()
}

fn expect_err<T>(result: Result<T, String>, what: &str) -> String {
    match result {
        Ok(_) => panic!("{}: expected an error", what),
        Err(e) => e,
    }
}

fn checkpoint_text(cfg: &Config, train: &Sample, test: &Sample) -> String {
    let mut text: Option<String> = None;
    search::run_resumable(cfg, train, test, None, 0, |cp| {
        text = Some(cp.to_text());
    })
    .expect("fresh run");
    text.expect("a checkpoint callback")
}

#[test]
fn rng_state_round_trip_produces_the_identical_stream() {
    let mut original = Mt::new(0xDEAD_BEEF);
    // Advance across at least one twist and stop at an arbitrary index.
    for _ in 0..1_000 {
        original.next_u32();
    }
    let (words, idx) = original.state();
    let mut restored = Mt::from_state(*words, idx);
    for i in 0..10_000 {
        assert_eq!(
            original.next_u32(),
            restored.next_u32(),
            "RNG diverged at draw {}",
            i
        );
    }

    // The same check from the untwisted initial state (idx == 624).
    let mut fresh = Mt::new(7);
    let (words, idx) = fresh.state();
    let mut restored = Mt::from_state(*words, idx);
    for i in 0..2_000 {
        assert_eq!(fresh.next_u32(), restored.next_u32(), "diverged at {}", i);
    }
}

#[test]
fn trees_round_trip_exactly_including_awkward_constants() {
    let trees = vec![
        constant(1.234_567_890_123_456_7),
        constant(-0.0),
        constant(-987.654_321_098_765_4),
        node(
            Op::Neg,
            vec![node(
                Op::Pow,
                vec![input(0), constant(std::f64::consts::E)],
            )],
        ),
        node(
            Op::If,
            vec![
                node(Op::Greater, vec![input(0), constant(0.123_456_789_012_345)]),
                node(
                    Op::Mul,
                    vec![input(1), constant(-1.987_654_321_098_765)],
                ),
                node(
                    Op::Constrain,
                    vec![node(
                        Op::Sub,
                        vec![input(0), constant(std::f64::consts::PI)],
                    )],
                ),
            ],
        ),
        node(
            Op::Between,
            vec![
                input(0),
                constant(-2.000_000_000_000_001),
                constant(3.999_999_999_999_999),
            ],
        ),
        node(
            Op::Require,
            vec![node(
                Op::Mod,
                vec![input(0), constant(1.000_000_000_000_000_2)],
            )],
        ),
    ];

    for tree in trees {
        let text = encode_tree(&tree);
        let back = decode_tree(&text).expect("decode");
        assert!(
            expr::same(&tree, &back),
            "tree did not round-trip exactly:\n  {}\n  {}",
            text,
            encode_tree(&back)
        );
    }
}

#[test]
fn a_corrupt_tree_is_refused_not_guessed() {
    assert!(decode_tree("999 0 0").is_err(), "out-of-range op index");
    assert!(decode_tree("27 0 1 0 1.0 0").is_err(), "arity mismatch");
    assert!(decode_tree("0 1.5").is_err(), "truncated tree");
    assert!(decode_tree("0 1.5 0 0 2.0 0").is_err(), "trailing tokens");
    assert!(
        decode_tree("0 nope 0").is_err(),
        "a constant that is not a number"
    );
}

#[test]
fn the_checkpoint_round_trips_every_search_setting() {
    let train = sample(100);
    let test = sample(30);
    let mut cfg = Config {
        generations: 1,
        islands: 2,
        pop: 12,
        max_complexity: 9,
        metric: rcsr::metrics::Metric::Bic,
        seed: 41,
        ops: vec![Op::Add, Op::Mul, Op::Sin, Op::If, Op::Constrain],
        migration_interval: 3,
        restart_interval: 4,
        init_depth: 2,
        fit_iters: 0,
        dedup: false,
        time_budget: Some(12.5),
        patience: 7,
        predictor_samples: Some(40),
        ..Default::default()
    };
    cfg.complexity_model.apply_op_costs("pow=2.5,sin=1.5").unwrap();
    cfg.complexity_model.apply_weights("var=2,const=3,depth=0.5").unwrap();

    let text = checkpoint_text(&cfg, &train, &test);
    let cp = Checkpoint::from_text(&text).expect("parse");
    assert!(
        cfg.same_search_settings(&cp.config),
        "every trajectory setting must survive the round trip; differences: {:?}",
        cfg.search_setting_differences(&cp.config)
    );
    assert_eq!(cfg.generations, cp.config.generations);
    assert_eq!(cp.generations, 1);
    assert_eq!(cp.config.time_budget, Some(12.5));
    assert_eq!(cp.config.predictor_samples, Some(40));
    cp.check_data(&train, &test).expect("same data");
}

#[test]
fn an_uninterrupted_run_and_a_saved_then_resumed_run_match_exactly() {
    let train = sample(240);
    let test = sample(80);
    let full = Config {
        generations: 40,
        ..config(11)
    };
    let uninterrupted = search::run(&full, &train, &test);

    // First process: run 20 generations and leave a checkpoint behind.
    let first = Config {
        generations: 20,
        ..config(11)
    };
    let text = checkpoint_text(&first, &train, &test);
    let saved = Checkpoint::from_text(&text).expect("parse saved checkpoint");

    // Second process: continue with the remaining budget.
    let resumed = search::run_resumable(&full, &train, &test, Some(saved), 0, |_| {})
        .expect("resume");

    assert_eq!(
        uninterrupted.generations, resumed.generations,
        "the generation count must match"
    );
    assert_eq!(
        uninterrupted.converged_early, resumed.converged_early,
        "the stop reason must match"
    );
    assert_eq!(
        describe(&uninterrupted),
        describe(&resumed),
        "the resumed archive must be identical (errors compared by bit pattern)"
    );
    assert_eq!(
        uninterrupted
            .test_errors
            .iter()
            .map(|e| e.to_bits())
            .collect::<Vec<_>>(),
        resumed
            .test_errors
            .iter()
            .map(|e| e.to_bits())
            .collect::<Vec<_>>(),
        "test errors must be identical"
    );
}

#[test]
fn resuming_onto_different_training_data_is_refused() {
    let train = sample(120);
    let test = sample(40);
    let cfg = config(3);
    let text = checkpoint_text(&cfg, &train, &test);
    let saved = Checkpoint::from_text(&text).expect("parse");

    let mut changed = train.clone();
    changed.y[0] += 0.25;
    let err = expect_err(
        search::run_resumable(&cfg, &changed, &test, Some(saved), 0, |_| {}),
        "modified training data must be refused",
    );
    assert!(
        err.contains("training data mismatch") && err.contains("fingerprint"),
        "the error must explain the fingerprint mismatch: {}",
        err
    );
}

#[test]
fn resuming_onto_different_test_data_is_refused() {
    let train = sample(120);
    let test = sample(40);
    let cfg = config(4);
    let text = checkpoint_text(&cfg, &train, &test);
    let saved = Checkpoint::from_text(&text).expect("parse");

    let mut changed = test.clone();
    changed.x[0][0] += 1.0;
    let err = expect_err(
        search::run_resumable(&cfg, &train, &changed, Some(saved), 0, |_| {}),
        "modified test data must be refused",
    );
    assert!(
        err.contains("test data mismatch") && err.contains("fingerprint"),
        "the error must explain the fingerprint mismatch: {}",
        err
    );
}

#[test]
fn truncated_corrupt_and_wrong_version_checkpoints_are_clear_errors() {
    assert!(Checkpoint::from_text("").is_err(), "an empty file");
    let err = expect_err(
        Checkpoint::from_text("rcsr-checkpoint v2\n"),
        "unknown version",
    );
    assert!(err.contains("unsupported checkpoint version"), "{}", err);
    assert!(
        Checkpoint::from_text("hello\n").is_err(),
        "not a checkpoint"
    );

    let train = sample(60);
    let test = sample(20);
    let text = checkpoint_text(&config(9), &train, &test);
    let cut = text.len() / 2;
    assert!(
        Checkpoint::from_text(&text[..cut]).is_err(),
        "a truncated checkpoint must be refused"
    );

    let mut bad_op = text.clone();
    bad_op.push_str("ind 0 0 0 999 0 0\n");
    let err = expect_err(Checkpoint::from_text(&bad_op), "bad op index");
    assert!(err.contains("bad operator index"), "{}", err);

    let mut duplicate = text.clone();
    duplicate.push_str("generations 0\n");
    assert!(
        Checkpoint::from_text(&duplicate).is_err(),
        "a duplicate record must be refused"
    );
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
            "rcsr-checkpoint-{}-{}-{}",
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

fn line_csv(n: usize) -> String {
    let mut text = String::from("x,y\n");
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        text.push_str(&format!("{:.6},{:.6}\n", x, 2.0 * x + 1.0));
    }
    text
}

#[test]
fn save_alone_does_not_change_the_result() {
    let csv = TempPath::new("save.csv");
    std::fs::write(&csv.0, line_csv(60)).expect("write csv");
    let plain_out = TempPath::new("plain.tsv");
    let saved_out = TempPath::new("saved.tsv");
    let checkpoint = TempPath::new("run.rcsr");

    let base = [
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "5",
        "--generations",
        "8",
        "--pop",
        "30",
        "--max-complexity",
        "8",
        "--quiet",
    ];
    let mut plain_args: Vec<&str> = base.to_vec();
    plain_args.extend(["--out", plain_out.as_str()]);
    let plain = rcsr(&plain_args);
    assert!(plain.status.success(), "plain run failed: {:?}", plain);

    let mut saved_args: Vec<&str> = base.to_vec();
    saved_args.extend([
        "--save",
        checkpoint.as_str(),
        "--save-every",
        "3",
        "--out",
        saved_out.as_str(),
    ]);
    let saved = rcsr(&saved_args);
    assert!(saved.status.success(), "saved run failed: {:?}", saved);

    let a = std::fs::read_to_string(&plain_out.0).expect("plain tsv");
    let b = std::fs::read_to_string(&saved_out.0).expect("saved tsv");
    assert_eq!(a, b, "--save must not change the search result");

    let text = std::fs::read_to_string(&checkpoint.0).expect("checkpoint file");
    assert!(
        text.starts_with("rcsr-checkpoint v1\n"),
        "the checkpoint must carry the version line"
    );
}

#[test]
fn help_and_schema_document_checkpoints() {
    let help = rcsr(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    for flag in ["--save", "--save-every", "--resume"] {
        assert!(text.contains(flag), "--help is missing {flag}");
    }

    let schema = rcsr(&["schema"]);
    assert!(schema.status.success());
    let text = String::from_utf8_lossy(&schema.stdout);
    for flag in ["--save", "--save-every", "--resume"] {
        assert!(text.contains(flag), "schema is missing {flag}");
    }
}

#[test]
fn cli_resume_errors_use_the_documented_exit_codes() {
    let csv = TempPath::new("cli.csv");
    std::fs::write(&csv.0, line_csv(40)).expect("write csv");
    let checkpoint = TempPath::new("cli.rcsr");

    // Missing resume file: an input error (3).
    let missing = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--resume",
        "/definitely/not/here.rcsr",
    ]);
    assert_eq!(missing.status.code(), Some(3), "missing file is exit 3");

    // Create a real checkpoint, then contradict one of its search settings.
    let made = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "2",
        "--generations",
        "5",
        "--pop",
        "25",
        "--save",
        checkpoint.as_str(),
    ]);
    assert!(made.status.success(), "save run failed: {:?}", made);

    let conflict = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--resume",
        checkpoint.as_str(),
        "--pop",
        "26",
    ]);
    assert_eq!(
        conflict.status.code(),
        Some(2),
        "a contradictory override is a usage error"
    );
    let err = String::from_utf8_lossy(&conflict.stderr);
    assert!(err.contains("--resume"), "the error must name --resume: {}", err);

    // A corrupt checkpoint is an input error (3), not a panic.
    let corrupt = TempPath::new("corrupt.rcsr");
    std::fs::write(&corrupt.0, "rcsr-checkpoint v9\n").expect("write corrupt");
    let refused = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--resume",
        corrupt.as_str(),
    ]);
    assert_eq!(refused.status.code(), Some(3), "corrupt input is exit 3");
}
