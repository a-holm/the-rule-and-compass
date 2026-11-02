//! Save/resume tests (issue #17).
//!
//! The central guarantee is trajectory determinism: an uninterrupted run and
//! a run split at an arbitrary generation boundary must produce the same
//! archive, bit for bit. The tests also pin down exact tree/RNG round-trips and
//! the refusal of corrupt checkpoints and mismatched data.

use rcsr::atomic::Fault;
use rcsr::checkpoint::{config_hash, decode_tree, encode_tree, Backend, Checkpoint};
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
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
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
fn checkpoint_tree_depth_and_node_limits_are_enforced() {
    use rcsr::parse::Limits;
    // Depth 256 (255 Neg nodes plus the leaf) is accepted; one more is not.
    let mut tree = input(0);
    for _ in 0..(Limits::MAX_DEPTH - 1) {
        tree = node(Op::Neg, vec![tree]);
    }
    let text = encode_tree(&tree);
    assert!(decode_tree(&text).is_ok(), "depth 256 must round-trip");

    let too_deep = node(Op::Neg, vec![tree]);
    let e = expect_err(decode_tree(&encode_tree(&too_deep)), "depth 257");
    assert!(e.contains("depth") && e.contains("256"), "{e}");

    // An exactly-4096-node balanced tree is accepted; 4097 nodes is rejected.
    // 4095 nodes is a full-ish binary tree (2047 + 2047 + root); one Neg
    // wrapper makes exactly 4096.
    fn balanced(nodes: usize) -> expr::Node {
        if nodes == 1 {
            return input(0);
        }
        let left = (nodes - 1) / 2;
        let right = nodes - 1 - left;
        node(Op::Add, vec![balanced(left), balanced(right)])
    }
    let chain = node(Op::Neg, vec![balanced(Limits::MAX_NODES - 1)]);
    assert!(decode_tree(&encode_tree(&chain)).is_ok());
    let over = node(Op::Neg, vec![chain]);
    let e = expect_err(decode_tree(&encode_tree(&over)), "4097 nodes");
    assert!(e.contains("more than 4096 nodes"), "{e}");
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
fn checkpoint_text_is_a_fixed_point_of_serialise_then_parse() {
    let train = sample(40);
    let test = sample(10);
    // Non-default patience so `best_prev`/`no_improve` are live. A restart
    // every generation makes the per-island stagnation counters advance, so
    // the two islands end with different values rather than the default zeros.
    let cfg = Config {
        generations: 3,
        islands: 2,
        pop: 8,
        max_complexity: 8,
        patience: 1,
        restart_interval: 1,
        fit_iters: 2,
        seed: 2,
        // #76: the fixture needs a *stagnating* run so `stopped_early`,
        // `best_prev`, `no_improve` and the per-island counters are live. The
        // archive-parent feedback keeps this tiny run improving, which is the
        // feature working; the checkpoint round trip is independent of it.
        front_parents: false,
        gpu: false,
        ..Default::default()
    };
    let text = checkpoint_text(&cfg, &train, &test);
    let cp = Checkpoint::from_text(&text).expect("parse");

    // The fixture must exercise the fields, not just the serialiser: a
    // checkpoint whose counters are all zero would pass the round trip
    // vacuously.
    assert!(cp.stopped_early, "the fixture should stop early");
    assert!(cp.best_prev.is_finite() && cp.best_prev > 0.0);
    assert!(cp.no_improve > 0);
    assert_eq!(cp.stagnant.len(), 2, "one counter per island");
    assert_ne!(
        cp.stagnant[0], cp.stagnant[1],
        "the two islands should have different stagnation counters"
    );

    assert_eq!(
        cp.to_text(),
        text,
        "serialise -> parse -> serialise changed bytes"
    );

    // Name the bookkeeping fields explicitly: byte identity alone would still
    // pass if the writer and the reader both forgot a field.
    let cp2 = Checkpoint::from_text(&cp.to_text()).expect("reparse");
    assert_eq!(cp2.stopped_early, cp.stopped_early, "stopped_early");
    assert_eq!(
        cp2.best_prev.to_bits(),
        cp.best_prev.to_bits(),
        "best_prev"
    );
    assert_eq!(cp2.no_improve, cp.no_improve, "no_improve");
    assert_eq!(cp2.stagnant, cp.stagnant, "stagnant");
    assert_eq!(
        cp2.main_rng.state(),
        cp.main_rng.state(),
        "main_rng state"
    );
    assert_eq!(cp2.generations, cp.generations, "generations");
}

#[test]
fn fit_portfolio_settings_round_trip() {
    use rcsr::fit::FitStrategy;
    let train = sample(120);
    let test = sample(40);
    let mut cfg = config(5);
    cfg.fit_strategy = FitStrategy::VarPro;
    cfg.fit_rows = 17;
    let text = checkpoint_text(&cfg, &train, &test);
    let cp = Checkpoint::from_text(&text).expect("parse");
    assert_eq!(cp.config.fit_strategy, FitStrategy::VarPro);
    assert_eq!(cp.config.fit_rows, 17);
    assert_eq!(cp.to_text(), text, "round trip must preserve the new fields");
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
    let saved_candidates = saved.candidates_evaluated;

    // Second process: continue with the remaining budget.
    let resumed = search::run_resumable(&full, &train, &test, Some(saved), 0, |_| {})
        .expect("resume");

    // #482: the offspring count survives the checkpoint, so a resumed run
    // reports the same total as the uninterrupted one.
    assert!(saved_candidates > 0 && saved_candidates < resumed.candidates_evaluated);
    assert_eq!(uninterrupted.candidates_evaluated, resumed.candidates_evaluated);

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

/// #140: a GPU-backed checkpoint resumed on the same GPU must reproduce the
/// uninterrupted archive bit for bit, just like the CPU A/B test above. The
/// test is hardware-gated: without a usable device it is skipped, not failed.
#[cfg(feature = "cuda")]
#[test]
fn a_gpu_run_and_a_gpu_resume_match_exactly() {
    if !rcsr::backend::probe().available() {
        eprintln!("no usable GPU; skipping the GPU->GPU resume A/B test");
        return;
    }
    let train = sample(240);
    let test = sample(80);
    let mut full = Config {
        generations: 20,
        // #236: the device scores squared error only, so a GPU run must use a
        // metric the kernel can rank. The default `absolute` would keep this
        // test on the CPU and quietly stop testing the GPU resume path.
        metric: rcsr::metrics::Metric::Squared,
        ..config(17)
    };
    full.gpu = true;
    let uninterrupted = search::run(&full, &train, &test);
    assert_eq!(uninterrupted.backend, rcsr::backend::Kind::Gpu);

    let mut first = full.clone();
    first.generations = 10;
    let text = checkpoint_text(&first, &train, &test);
    assert!(
        text.contains("backend gpu\n"),
        "a GPU checkpoint must record its backend"
    );
    let saved = Checkpoint::from_text(&text).expect("parse saved GPU checkpoint");

    let resumed = search::run_resumable(&full, &train, &test, Some(saved), 0, |_| {})
        .expect("GPU resume");
    assert_eq!(resumed.backend, rcsr::backend::Kind::Gpu);
    assert_eq!(
        describe(&uninterrupted),
        describe(&resumed),
        "the GPU-resumed archive must be identical (errors compared by bit pattern)"
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
        "GPU test errors must be identical"
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
fn checkpoint_records_backend_and_canonical_config_hash() {
    let train = sample(40);
    let test = sample(10);
    let mut cfg = config(7);
    cfg.gpu = false;
    let text = checkpoint_text(&cfg, &train, &test);
    assert!(
        text.contains(&format!("backend {}\n", Backend::Cpu.as_str())),
        "a CPU checkpoint must record its backend"
    );
    assert!(
        text.contains(&format!("config_hash {:016x}\n", config_hash(&cfg))),
        "the canonical config hash must be present"
    );
    let cp = Checkpoint::from_text(&text).expect("parse");
    assert_eq!(cp.backend, Backend::Cpu);
    assert_eq!(cp.config_hash, config_hash(&cp.config));
    // Raising only the generation budget must not change the hash: it is the
    // one field that a resume may legitimately change.
    let mut raised = cp.config.clone();
    raised.generations += 20;
    assert_eq!(config_hash(&raised), cp.config_hash);

    // Editing a saved setting without updating the hash is refused.
    let edited = text.replacen("config seed 7", "config seed 8", 1);
    assert_ne!(edited, text, "fixture must contain the seed line");
    let err = expect_err(Checkpoint::from_text(&edited), "config edit must fail");
    assert!(err.contains("config_hash mismatch"), "{}", err);
}

#[test]
fn a_gpu_checkpoint_resumed_without_gpu_is_refused_unless_overridden() {
    let csv = TempPath::new("backend.csv");
    std::fs::write(&csv.0, line_csv(40)).expect("write csv");
    let checkpoint = TempPath::new("backend.rcsr");

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
        "3",
        "--pop",
        "20",
        "--save",
        checkpoint.as_str(),
        "--cpu",
        "--quiet",
    ]);
    assert!(made.status.success(), "save run failed: {:?}", made);

    // Pretend the checkpoint was produced by the GPU backend. The backend is
    // deliberately outside config_hash, so only the backend check fires.
    let text = std::fs::read_to_string(&checkpoint.0).expect("read checkpoint");
    assert!(text.contains("backend cpu\n"), "fixture must be a CPU run");
    let text = text.replacen("backend cpu\n", "backend gpu\n", 1);
    std::fs::write(&checkpoint.0, text).expect("write edited checkpoint");

    let refused = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--resume",
        checkpoint.as_str(),
        "--cpu",
    ]);
    assert_eq!(refused.status.code(), Some(3), "backend mismatch is exit 3");
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(
        err.contains("backend mismatch") && err.contains("gpu") && err.contains("cpu"),
        "the error must name both backends: {}",
        err
    );

    let allowed = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--resume",
        checkpoint.as_str(),
        "--allow-backend-change",
        "--cpu",
        "--generations",
        "4",
        "--quiet",
    ]);
    assert!(
        allowed.status.success(),
        "the override must let the run proceed: {}",
        String::from_utf8_lossy(&allowed.stderr)
    );
    let note = String::from_utf8_lossy(&allowed.stderr);
    assert!(
        note.contains("backend changed") && note.contains("--allow-backend-change"),
        "the override must be recorded on stderr: {}",
        note
    );
}

#[test]
fn a_partial_checkpoint_write_keeps_the_previous_good_generation() {
    let train = sample(40);
    let test = sample(10);
    let cfg = config(13);
    let text = checkpoint_text(&cfg, &train, &test);
    let cp = Checkpoint::from_text(&text).expect("parse");
    let path = TempPath::new("atomic.rcsr");
    cp.save(&path.0).expect("first save");
    let good = std::fs::read_to_string(&path.0).expect("first generation");

    let err = cp
        .save_with_fault(&path.0, Fault::Partial)
        .expect_err("the injected write must fail");
    assert!(err.contains("partial"), "{}", err);
    assert_eq!(
        std::fs::read_to_string(&path.0).expect("destination after fault"),
        good,
        "a killed write must leave the previous checkpoint intact"
    );
    assert!(
        !std::path::Path::new(&format!("{}.prev", path.as_str())).exists(),
        "a first save has no previous generation to retain"
    );

    cp.save(&path.0).expect("second save");
    let prev = format!("{}.prev", path.as_str());
    assert_eq!(
        std::fs::read_to_string(&prev).expect("retained previous generation"),
        good,
        "the previous good generation must be retained as <path>.prev"
    );
    let _ = std::fs::remove_file(prev);
    let _ = std::fs::remove_file(format!("{}.tmp", path.as_str()));
}

#[test]
fn help_and_schema_document_checkpoints() {
    let help = rcsr(&["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8_lossy(&help.stdout);
    for flag in [
        "--save",
        "--save-every",
        "--resume",
        "--allow-backend-change",
        "--run-manifest",
        "--json-out",
        "verify-manifest",
    ] {
        assert!(text.contains(flag), "--help is missing {flag}");
    }

    let schema = rcsr(&["schema"]);
    assert!(schema.status.success());
    let text = String::from_utf8_lossy(&schema.stdout);
    for flag in [
        "--save",
        "--save-every",
        "--resume",
        "--allow-backend-change",
        "--run-manifest",
        "--json-out",
    ] {
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

fn pnl_config(seed: u32, cost_bps: f64, train: &Sample) -> Config {
    let slice = rcsr::data::TradingSlice {
        rows: rcsr::trading::TradingRows {
            time: (0..train.len()).map(|i| 900.0 * i as f64).collect(),
            group: vec![0; train.len()],
            funding: Vec::new(),
        },
        ret: train.x.iter().map(|r| 0.002 * r[0]).collect(),
    };
    let spec = rcsr::trading::TradingSpec {
        cost_bps,
        min_trades: 0,
        ..Default::default()
    };
    let book = rcsr::trading::FitnessBook::new(&slice, &spec, 900.0).expect("book");
    Config {
        fitness: search::Fitness::LongPnl(std::sync::Arc::new(book)),
        ..config(seed)
    }
}

/// #480: the book hash is saved search state. The same book continues bit for
/// bit; a different cost is refused.
#[test]
fn a_long_pnl_run_resumes_with_the_same_book_and_refuses_another() {
    let train = sample(240);
    let test = sample(80);
    let full = Config {
        generations: 30,
        ..pnl_config(11, 5.0, &train)
    };
    let uninterrupted = search::run(&full, &train, &test);

    let first = Config {
        generations: 15,
        ..pnl_config(11, 5.0, &train)
    };
    let text = checkpoint_text(&first, &train, &test);
    assert!(text.contains("config fitness long_pnl:"), "the book hash is saved");
    let saved = Checkpoint::from_text(&text).expect("parse");
    assert_eq!(saved.config.fitness.key(), first.fitness.key());
    assert_ne!(config_hash(&first), config_hash(&Config { fitness: search::Fitness::Metric, ..first.clone() }));

    let resumed = search::run_resumable(&full, &train, &test, Some(saved), 0, |_| {}).expect("resume");
    assert_eq!(describe(&uninterrupted), describe(&resumed));
    assert_eq!(uninterrupted.candidates_evaluated, resumed.candidates_evaluated);

    let other = Config {
        generations: 30,
        ..pnl_config(11, 6.0, &train)
    };
    let saved = Checkpoint::from_text(&text).expect("parse");
    let err = expect_err(search::run_resumable(&other, &train, &test, Some(saved), 0, |_| {}), "other cost");
    assert!(err.contains("fitness"), "{err}");
}

/// #598: the pipeline depth is an execution setting. A resume takes the one it
/// is given, and the archive equals an uninterrupted run at that depth.
#[cfg(feature = "cuda")]
#[test]
fn a_resume_takes_the_requested_pipeline_depth() {
    if !rcsr::backend::probe().available() {
        eprintln!("no usable GPU; skipping the resume pipeline-depth test");
        return;
    }
    let csv = TempPath::new("depth.csv");
    std::fs::write(&csv.0, line_csv(200)).expect("write csv");
    let checkpoint = TempPath::new("depth.rcsr");
    let resumed_out = TempPath::new("depth-resumed.tsv");
    let resumed_deep_out = TempPath::new("depth-resumed-deep.tsv");
    let full_out = TempPath::new("depth-full.tsv");

    let common = |extra: &[&'static str]| -> Vec<String> {
        let mut args: Vec<String> = [
            "--data",
            csv.as_str(),
            "--target",
            "y",
            "--vars",
            "x",
            "--seed",
            "4",
            "--pop",
            "32",
            "--max-complexity",
            "8",
            "--metric",
            "squared",
            "--gpu",
            "--dispatch",
            "device",
            "--no-dev-log",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.extend(extra.iter().map(|s| s.to_string()));
        args
    };
    let run = |args: Vec<String>| -> Output {
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        rcsr(&refs)
    };

    let mut saved_args = common(&["--generations", "3", "--save"]);
    saved_args.push(checkpoint.as_str().to_string());
    let saved = run(saved_args);
    assert!(saved.status.success(), "save run failed: {:?}", saved);

    let resume = |depth: &'static str, generations: &'static str, out: &TempPath| -> Output {
        let mut args = common(&["--generations", generations, "--pipeline-depth", depth]);
        args.extend(["--resume", checkpoint.as_str(), "--out", out.as_str()].map(String::from));
        run(args)
    };

    let sync = resume("0", "6", &resumed_out);
    assert!(sync.status.success(), "resume failed: {:?}", sync);
    let err = String::from_utf8_lossy(&sync.stderr);
    assert!(
        err.contains("synchronous") && !err.contains("overlap depth"),
        "a resume with --pipeline-depth 0 must run synchronously: {}",
        err
    );

    let deep = resume("3", "6", &resumed_deep_out);
    assert!(deep.status.success(), "resume failed: {:?}", deep);
    let err = String::from_utf8_lossy(&deep.stderr);
    assert!(
        err.contains("overlap depth 3"),
        "a resume with --pipeline-depth 3 must overlap at depth 3: {}",
        err
    );

    let mut full_args = common(&["--generations", "6", "--pipeline-depth", "0", "--out"]);
    full_args.push(full_out.as_str().to_string());
    let full = run(full_args);
    assert!(full.status.success(), "uninterrupted run failed: {:?}", full);

    let a = std::fs::read_to_string(&full_out.0).expect("full tsv");
    let b = std::fs::read_to_string(&resumed_out.0).expect("resumed tsv");
    let c = std::fs::read_to_string(&resumed_deep_out.0).expect("deep tsv");
    assert_eq!(a, b, "the depth-0 resume must equal the uninterrupted depth-0 run");
    assert_eq!(a, c, "the depth-3 resume must give the same archive");
}
