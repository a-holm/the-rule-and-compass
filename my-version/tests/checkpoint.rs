//! Save/resume tests (issue #17).
//!
//! The central guarantee is trajectory determinism: an uninterrupted run and
//! a run split at an arbitrary generation boundary must produce the same
//! archive, bit for bit. The tests also pin down exact tree/RNG round-trips and
//! the refusal of corrupt checkpoints and mismatched data.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

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
        seed: 3,
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
        rcsr::backend::skip_without_device("no usable GPU; skipping the GPU->GPU resume A/B test");
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
    full.dispatch = rcsr::dispatch::DispatchMode::Device;
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
    rcsr_cmd::rcsr()
        .args(args)
        .output()
        .expect("run rcsr")
}

#[cfg(feature = "cuda")]
fn rcsr_on_device(args: &[&str]) -> Output {
    rcsr_cmd::rcsr_on_device()
        .args(args)
        .output()
        .expect("run rcsr")
}

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

struct TempPath(PathBuf, #[allow(dead_code)] Scratch);

impl TempPath {
    fn new(suffix: &str) -> Self {
        let scratch = Scratch::new("checkpoint");
        let path = scratch.join(suffix);
        TempPath(path, scratch)
    }
    fn as_str(&self) -> &str {
        self.0.to_str().expect("utf-8 path")
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

fn island_config(seed: u32, generations: usize) -> Config {
    Config {
        generations,
        islands: 3,
        pop: 24,
        max_complexity: 10,
        fit_iters: 2,
        migration_interval: 3,
        restart_interval: 4,
        restart_policy: search::RestartPolicy::Island,
        restart_isolation: 7,
        patience: 0,
        seed,
        ..Default::default()
    }
}

/// The exit checkpoint text and the restart count of one run.
fn island_run(
    cfg: &Config,
    resume: Option<Checkpoint>,
    train: &Sample,
    test: &Sample,
) -> (Outcome, String, u64) {
    let restarts = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sink = restarts.clone();
    let cfg = Config {
        dev_log: Some(std::sync::Arc::new(move |r: &rcsr::devlog::GenRecord| {
            sink.store(r.restarts_total, std::sync::atomic::Ordering::SeqCst);
        })),
        ..cfg.clone()
    };
    let mut text: Option<String> = None;
    let out = search::run_resumable(&cfg, train, test, resume, 0, |cp| text = Some(cp.to_text()))
        .expect("island run");
    (out, text.expect("exit checkpoint"), restarts.load(std::sync::atomic::Ordering::SeqCst))
}

#[test]
fn an_island_policy_run_and_its_resume_match_exactly() {
    let train = sample(240);
    let test = sample(80);
    let (uninterrupted, full_text, restarts) = island_run(&island_config(11, 40), None, &train, &test);
    assert!(restarts > 0, "the fixture restarts an island, else the resume proves nothing");
    assert!(full_text.contains("config restart_policy island\n"));
    assert!(full_text.contains("config restart_isolation 7\n"));
    assert_eq!(full_text.matches("\nisland_stall ").count(), 3, "one record per island");

    let (_, saved_text, _) = island_run(&island_config(11, 20), None, &train, &test);
    let saved = Checkpoint::from_text(&saved_text).expect("parse saved checkpoint");
    let (resumed, resumed_text, _) =
        island_run(&island_config(11, 40), Some(saved), &train, &test);

    assert_eq!(describe(&uninterrupted), describe(&resumed), "the archive");
    assert_eq!(
        full_text, resumed_text,
        "archive, populations, rng state and island_stall counters of the exit checkpoint"
    );
}

#[test]
fn an_archive_policy_checkpoint_keeps_its_bytes() {
    let text = checkpoint_text(&config(3), &sample(60), &sample(20));
    assert!(!text.contains("island_stall"));
    assert!(!text.contains("restart_policy"));
    assert!(!text.contains("restart_isolation"));
}

#[test]
fn an_island_checkpoint_resumes_only_under_its_own_policy() {
    let train = sample(100);
    let test = sample(30);
    let (_, text, _) = island_run(&island_config(5, 8), None, &train, &test);

    let archive_cfg = Config { restart_policy: search::RestartPolicy::Archive, ..island_config(5, 8) };
    let saved = Checkpoint::from_text(&text).expect("parse");
    let err = expect_err(
        search::run_resumable(&archive_cfg, &train, &test, Some(saved), 0, |_| {}),
        "resume under archive",
    );
    assert!(err.contains("restart_policy"), "{err}");

    let isolation_cfg = Config { restart_isolation: 9, ..island_config(5, 8) };
    let saved = Checkpoint::from_text(&text).expect("parse");
    let err = expect_err(
        search::run_resumable(&isolation_cfg, &train, &test, Some(saved), 0, |_| {}),
        "resume under another isolation",
    );
    assert!(err.contains("restart_isolation"), "{err}");

    let stripped: String = text
        .lines()
        .filter(|l| !l.starts_with("island_stall "))
        .map(|l| format!("{l}\n"))
        .collect();
    let err = expect_err(Checkpoint::from_text(&stripped), "island checkpoint without records");
    assert!(err.contains("island_stall"), "{err}");
}

#[test]
fn cli_resume_of_an_island_checkpoint_exits_2_naming_restart_policy() {
    let csv = TempPath::new("island.csv");
    std::fs::write(&csv.0, line_csv(40)).expect("write csv");
    let checkpoint = TempPath::new("island.rcsr");
    let base = ["--data", csv.as_str(), "--target", "y", "--vars", "x", "--seed", "2", "--pop", "25"];

    let mut made_args = base.to_vec();
    made_args.extend(["--generations", "5", "--restart-policy", "island", "--restart-isolation", "7", "--save", checkpoint.as_str()]);
    let made = rcsr(&made_args);
    assert!(made.status.success(), "save run failed: {:?}", made);
    let saved = std::fs::read_to_string(&checkpoint.0).expect("read checkpoint");
    assert!(saved.contains("config restart_policy island\n") && saved.contains("island_stall 0 "));

    let mut conflict_args = base.to_vec();
    conflict_args.extend(["--resume", checkpoint.as_str(), "--restart-policy", "archive"]);
    let conflict = rcsr(&conflict_args);
    assert_eq!(conflict.status.code(), Some(2), "a contradictory policy is a usage error");
    let err = String::from_utf8_lossy(&conflict.stderr);
    assert!(err.contains("restart_policy"), "the error must name restart_policy: {}", err);

    let mut isolation_args = base.to_vec();
    isolation_args.extend(["--resume", checkpoint.as_str(), "--restart-isolation", "9"]);
    let isolation = rcsr(&isolation_args);
    assert_eq!(isolation.status.code(), Some(2), "a contradictory isolation is a usage error");
    assert!(String::from_utf8_lossy(&isolation.stderr).contains("restart_isolation"));

    let mut resume_args = base.to_vec();
    resume_args.extend(["--resume", checkpoint.as_str()]);
    let resumed = rcsr(&resume_args);
    assert!(resumed.status.success(), "a resume that omits the flags keeps the saved policy: {:?}", resumed);
}

/// #1090: an `rcsr` run with the `island_panic` test fault (every island thread
/// panics at the start of its generation). The fault hook exists in a `cuda`
/// build only; the run itself stays on the CPU.
#[cfg(feature = "cuda")]
fn panicking_rcsr(args: &[&str]) -> Output {
    rcsr_cmd::rcsr()
        .args(args)
        .env("RCSR_FAULT", "island_panic")
        .output()
        .expect("run rcsr")
}

/// #1090: a panic in an island thread fails the run with the search-error exit
/// code, names every island that panicked, and writes no checkpoint for that
/// generation: the last good checkpoint stays on disk.
#[cfg(feature = "cuda")]
#[test]
fn an_island_panic_exits_non_zero_and_keeps_the_last_good_checkpoint() {
    let csv = TempPath::new("panic.csv");
    // No short formula fits these rows, so the run cannot stop early as solved.
    let rows: String = (0..60)
        .map(|i| {
            let x = -2.0 + 4.0 * i as f64 / 59.0;
            format!("{x:.6},{:.6}\n", (3.0 * x).sin() + 0.1 * ((i * 7919) % 13) as f64)
        })
        .collect();
    std::fs::write(&csv.0, format!("x,y\n{rows}")).expect("write csv");
    for (islands, named) in [("1", vec![0usize]), ("3", vec![0, 1, 2])] {
        let checkpoint = TempPath::new(&format!("panic-{islands}.rcsr"));
        let base = [
            "--data", csv.as_str(), "--target", "y", "--vars", "x", "--seed", "2", "--pop", "20", "--islands", islands,
            "--cpu", "--save", checkpoint.as_str(), "--save-every", "1",
        ];
        let mut first = base.to_vec();
        first.extend(["--generations", "3"]);
        let made = rcsr(&first);
        assert!(made.status.success(), "save run failed: {:?}", made);
        let good = std::fs::read_to_string(&checkpoint.0).expect("read checkpoint");
        assert_eq!(Checkpoint::from_text(&good).expect("parse").generations, 3);

        let mut resumed = base.to_vec();
        resumed.extend(["--generations", "6", "--resume", checkpoint.as_str()]);
        let failed = panicking_rcsr(&resumed);
        let err = String::from_utf8_lossy(&failed.stderr);
        assert_eq!(failed.status.code(), Some(3), "a search error exits 3: {err}");
        for i in &named {
            assert!(
                err.contains(&format!("island {i} panicked in generation 3: injected island panic")),
                "islands {islands}: the error names island {i}: {err}"
            );
        }
        assert!(!err.contains(&format!("island {islands} panicked")), "only real islands are named: {err}");
        assert_eq!(
            std::fs::read_to_string(&checkpoint.0).expect("read checkpoint"),
            good,
            "islands {islands}: the failing generation wrote no checkpoint"
        );
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
        rcsr::backend::skip_without_device("no usable GPU; skipping the resume pipeline-depth test");
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
        rcsr_on_device(&refs)
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

/// #860: the exit checkpoint text and the report of one run at `threads`
/// workers. `threads_explicit` makes the scheduler obey the count.
fn run_at_threads(cfg: &Config, threads: usize, resume: Option<Checkpoint>, train: &Sample, test: &Sample) -> (Outcome, String) {
    let cfg = Config { threads, threads_explicit: true, ..cfg.clone() };
    let mut text: Option<String> = None;
    let out = search::run_resumable(&cfg, train, test, resume, 0, |cp| text = Some(cp.to_text()))
        .expect("run");
    (out, text.expect("exit checkpoint"))
}

fn assert_thread_free(base: &Config, what: &str, train: &Sample, test: &Sample) {
    let (reference, reference_text) = run_at_threads(base, 1, None, train, test);
    for threads in [2usize, 3, 12] {
        let (out, text) = run_at_threads(base, threads, None, train, test);
        assert_eq!(describe(&reference), describe(&out), "{what}: archive at {threads} threads");
        let bits = |o: &Outcome| o.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&reference), bits(&out), "{what}: test errors at {threads} threads");
        assert_eq!(reference_text, text, "{what}: exit checkpoint bytes at {threads} threads");
    }
    // A resume under another count continues the same trajectory.
    let first = Config { generations: base.generations / 2, ..base.clone() };
    let (_, half) = run_at_threads(&first, 12, None, train, test);
    let saved = Checkpoint::from_text(&half).expect("parse");
    let (resumed, resumed_text) = run_at_threads(base, 1, Some(saved), train, test);
    assert_eq!(describe(&reference), describe(&resumed), "{what}: 12 threads then 1");
    assert_eq!(reference_text, resumed_text, "{what}: resumed checkpoint bytes");
}

/// #860: no stage of the node-path search reads the thread count into a result:
/// batch scoring, constant fits, the screened refit and the finalist rebuild
/// give the same archive, test errors and checkpoint bytes at every count, and
/// a resume under another count continues the same trajectory.
#[test]
fn search_results_do_not_depend_on_the_thread_count() {
    let train = sample(2600);
    let test = sample(300);
    let base = Config {
        generations: 6,
        islands: 3,
        pop: 40,
        max_complexity: 10,
        fit_iters: 3,
        metric: rcsr::metrics::Metric::Squared,
        ..config(11)
    };
    assert_thread_free(&base, "exact fit", &train, &test);
    let screened = Config { fit_samples: 700, ..base.clone() };
    assert_thread_free(&screened, "screened fit", &train, &test);
    assert_thread_free(&pnl_config(12, 5.0, &train), "pnl", &train, &test);
}

/// #860: the same on the device with the device fit arm, whose exact CPU
/// fallback pool and finalist refit read the thread count to size their workers.
#[cfg(feature = "cuda")]
#[test]
fn a_device_search_does_not_depend_on_the_thread_count() {
    if !rcsr::backend::probe().available() {
        rcsr::backend::skip_without_device("no usable GPU; skipping the device thread-count check");
        return;
    }
    let train = sample(2600);
    let test = sample(300);
    let mut base = Config {
        generations: 6,
        islands: 2,
        pop: 40,
        max_complexity: 10,
        fit_iters: 3,
        metric: rcsr::metrics::Metric::Squared,
        fit_strategy: rcsr::fit::FitStrategy::Gpu,
        ..config(11)
    };
    base.gpu = true;
    base.dispatch = rcsr::dispatch::DispatchMode::Device;
    assert_thread_free(&base, "device fit", &train, &test);
}

fn allowed_cpus() -> Vec<usize> {
    let status = std::fs::read_to_string("/proc/self/status").expect("proc status");
    let list = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .expect("Cpus_allowed_list")
        .trim();
    let mut cpus = Vec::new();
    for part in list.split(',') {
        match part.split_once('-') {
            Some((a, b)) => cpus.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => cpus.push(part.parse().unwrap()),
        }
    }
    cpus
}

/// #860: three generations under two CPUs, then a resume under one CPU: exit 0,
/// a notice on stderr and the same checkpoint as six uninterrupted generations.
#[test]
fn resume_under_another_cpu_count_is_allowed() {
    let cpus = allowed_cpus();
    assert!(cpus.len() >= 2, "the test needs two allowed CPUs, have {cpus:?}");
    let (two, one) = (format!("{},{}", cpus[0], cpus[1]), cpus[0].to_string());
    let scratch = Scratch::new("cpu-count");
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let (ck, whole) = (scratch.join("ck.rcsr"), scratch.join("whole.rcsr"));
    let program = rcsr_cmd::rcsr().get_program().to_owned();
    let run = |cpus: &str, generations: &str, save: &std::path::Path, resume: bool| -> Output {
        let mut cmd = Command::new("taskset");
        cmd.env("RCSR_GPU", "0")
            .args(["-c", cpus])
            .arg(&program)
            .args(["--data", csv.to_str().unwrap(), "--target", "y", "--vars", "x", "--seed", "5"])
            .args(["--pop", "30", "--max-complexity", "8", "--cpu", "--no-dev-log", "--quiet"])
            .args(["--generations", generations, "--save", save.to_str().unwrap()]);
        if resume {
            cmd.args(["--resume", ck.to_str().unwrap()]);
        }
        cmd.output().expect("run rcsr")
    };
    let first = run(&two, "3", &ck, false);
    assert!(first.status.success(), "{:?}", first);
    let resumed = run(&one, "6", &ck, true);
    assert!(resumed.status.success(), "{:?}", resumed);
    let err = String::from_utf8_lossy(&resumed.stderr);
    assert!(
        err.contains("# checkpoint: build identity changed (threads: saved 2 -> current 1)"),
        "{err}"
    );
    assert!(!err.contains("--allow-build-change"), "{err}");
    let uninterrupted = run(&one, "6", &whole, false);
    assert!(uninterrupted.status.success(), "{:?}", uninterrupted);
    let final_line = |p: &std::path::Path| {
        std::fs::read_to_string(p)
            .unwrap()
            .lines()
            .find(|l| l.starts_with("final "))
            .map(str::to_string)
            .expect("a final line")
    };
    assert_eq!(final_line(&ck), final_line(&whole));
    assert_eq!(std::fs::read(&ck).unwrap(), std::fs::read(&whole).unwrap());
}

/// #880, #966: a resume names the time budget it runs with, which is only the one on its own
/// command line; the budget saved in the checkpoint is not used.
#[test]
fn a_resume_prints_the_time_budget_it_uses() {
    let csv = TempPath::new("budget.csv");
    std::fs::write(&csv.0, line_csv(60)).expect("write csv");
    let ck = TempPath::new("budget.rcsr");
    let out = TempPath::new("budget.json");
    let run = |extra: &[&str]| -> Output {
        let mut args = vec![
            "--data", csv.as_str(), "--target", "y", "--vars", "x", "--seed", "5", "--pop", "30",
            "--max-complexity", "8", "--cpu", "--no-dev-log", "--generations", "6", "--save",
            ck.as_str(), "--json-out", out.as_str(),
        ];
        args.extend_from_slice(extra);
        rcsr(&args)
    };
    let budget_lines = |o: &Output| -> Vec<String> {
        String::from_utf8_lossy(&o.stderr)
            .lines()
            .filter(|l| l.starts_with("# checkpoint: time budget"))
            .map(str::to_string)
            .collect()
    };
    let stop = || {
        let json = std::fs::read_to_string(&out.0).unwrap();
        let reason = json.split("\"stop_reason\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        let gens = json.split("\"generations_run\":").nth(1).unwrap().split(',').next().unwrap().to_string();
        (reason, gens)
    };

    let first = run(&["--time-budget", "0"]);
    assert!(first.status.success(), "{:?}", first);
    assert!(budget_lines(&first).is_empty(), "a fresh run prints no resume line");

    let unlimited = run(&["--resume", ck.as_str()]);
    assert!(unlimited.status.success(), "{:?}", unlimited);
    assert!(budget_lines(&unlimited).is_empty(), "the saved budget is not inherited");
    assert_eq!(stop(), ("generations".to_string(), "6".to_string()));

    let first = run(&["--time-budget", "0"]);
    assert!(first.status.success(), "{:?}", first);
    let replaced = run(&["--resume", ck.as_str(), "--time-budget", "100000"]);
    assert!(replaced.status.success(), "{:?}", replaced);
    assert_eq!(budget_lines(&replaced), ["# checkpoint: time budget 100000 s (from --time-budget)"]);
    assert_eq!(stop(), ("generations".to_string(), "6".to_string()));

    let ck2 = TempPath::new("nobudget.rcsr");
    let plain = rcsr(&[
        "--data", csv.as_str(), "--target", "y", "--vars", "x", "--seed", "5", "--pop", "30",
        "--max-complexity", "8", "--cpu", "--no-dev-log", "--generations", "3", "--save", ck2.as_str(),
        "--json-out", out.as_str(),
    ]);
    assert!(plain.status.success());
    let resumed = rcsr(&[
        "--data", csv.as_str(), "--target", "y", "--vars", "x", "--cpu", "--no-dev-log", "--generations", "6",
        "--resume", ck2.as_str(), "--json-out", out.as_str(),
    ]);
    assert!(resumed.status.success(), "{:?}", resumed);
    assert!(budget_lines(&resumed).is_empty(), "no budget, no line");
}

/// #966: a checkpoint saved under `saved_budget`, resumed with `resume_extra`.
/// Returns `(exit success, stop_reason, generations_run)` of the resume.
fn resume_after_budget(tag: &str, saved_budget: &str, resume_extra: &[&str]) -> (bool, String, String) {
    let csv = TempPath::new(&format!("{tag}.csv"));
    std::fs::write(&csv.0, line_csv(60)).expect("write csv");
    let ck = TempPath::new(&format!("{tag}.rcsr"));
    let out = TempPath::new(&format!("{tag}.json"));
    let run = |extra: &[&str]| -> Output {
        let mut args = vec![
            "--data", csv.as_str(), "--target", "y", "--vars", "x", "--seed", "5", "--pop", "30",
            "--max-complexity", "8", "--cpu", "--no-dev-log", "--generations", "6", "--save",
            ck.as_str(), "--json-out", out.as_str(),
        ];
        args.extend_from_slice(extra);
        rcsr(&args)
    };
    let first = run(&["--time-budget", saved_budget]);
    assert!(first.status.success(), "{:?}", first);
    let saved = Checkpoint::load(std::path::Path::new(ck.as_str())).expect("saved checkpoint");
    assert!(saved.generations < 6, "the saved budget must stop the first run early");
    let mut extra = vec!["--resume", ck.as_str()];
    extra.extend_from_slice(resume_extra);
    let resumed = run(&extra);
    let json = std::fs::read_to_string(&out.0).unwrap();
    let reason = json.split("\"stop_reason\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    let gens = json.split("\"generations_run\":").nth(1).unwrap().split(',').next().unwrap().to_string();
    (resumed.status.success(), reason, gens)
}

/// #966: a resume without `--time-budget` has no budget, whatever was saved.
#[test]
fn resume_without_time_budget_flag_is_unlimited() {
    let (ok, reason, gens) = resume_after_budget("unlimited", "0.000000001", &[]);
    assert!(ok);
    assert_eq!((reason.as_str(), gens.as_str()), ("generations", "6"));
}

/// #966: `--time-budget` on the resume line still limits that process.
#[test]
fn resume_with_time_budget_flag_uses_it() {
    let (ok, reason, _) = resume_after_budget("flagged", "0", &["--time-budget", "0"]);
    assert!(ok);
    assert_eq!(reason, "time_budget");
}

/// #966: a checkpoint saved under budget 0 resumes to `--generations` with exit code 0.
#[test]
fn resume_of_zero_budget_checkpoint_runs_to_generations() {
    let (ok, reason, gens) = resume_after_budget("zero", "0", &[]);
    assert!(ok);
    assert_eq!((reason.as_str(), gens.as_str()), ("generations", "6"));
}

/// #855: `chunks` one-generation chunks (a zero `--time-budget` stops after one
/// generation), each resumed from the previous exit checkpoint, then a last
/// chunk to `base.generations`. Returns the last chunk's outcome and the text of
/// its exit checkpoint.
fn chunked(base: &Config, interim: bool, chunks: usize, train: &Sample, test: &Sample) -> (Outcome, String) {
    let mut saved: Option<Checkpoint> = None;
    for k in 1..=chunks {
        let cfg = Config {
            time_budget: Some(0.0),
            interim_chunk: interim,
            ..base.clone()
        };
        let mut text: Option<String> = None;
        let out = search::run_resumable(&cfg, train, test, saved.take(), 0, |cp| {
            text = Some(cp.to_text());
        })
        .expect("chunk");
        assert_eq!(out.stop_reason, "time_budget");
        assert_eq!(out.generations, k);
        assert_eq!(out.interim, interim);
        let cp = Checkpoint::from_text(&text.expect("exit checkpoint")).expect("parse");
        if interim {
            assert!(out.test_errors.is_empty(), "an interim chunk scores no test rows");
            assert!(cp.final_archive.is_none(), "an interim checkpoint holds no reported archive");
        } else {
            assert_eq!(out.test_errors.len(), out.archive.items.len());
            assert!(cp.final_archive.is_some());
        }
        saved = Some(cp);
    }
    let mut text: Option<String> = None;
    let last = search::run_resumable(base, train, test, saved, 0, |cp| text = Some(cp.to_text()))
        .expect("last chunk");
    assert!(!last.interim);
    (last, text.expect("exit checkpoint"))
}

fn assert_same_report(a: &Outcome, b: &Outcome, what: &str) {
    assert_eq!(describe(a), describe(b), "{what}: archive");
    let bits = |o: &Outcome| o.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(a), bits(b), "{what}: test errors");
    assert_eq!(a.archive.hash(), b.archive.hash(), "{what}: archive hash");
    assert_eq!(a.candidates_evaluated, b.candidates_evaluated, "{what}: candidates");
    assert_eq!(a.generations, b.generations, "{what}: generations");
}

/// #855: three interim chunks and a last chunk give the same last-chunk report
/// and the same exit checkpoint, byte for byte, as the same chunks without the
/// flag and as one uninterrupted run.
#[test]
fn interim_chunks_keep_the_final_hash() {
    let train = sample(240);
    let test = sample(80);
    for seed in [11u32, 12] {
        let bases = [
            ("squared", Config { metric: rcsr::metrics::Metric::Squared, ..config(seed) }),
            ("pnl", pnl_config(seed, 5.0, &train)),
        ];
        for (name, base) in bases {
            let what = format!("{name} seed {seed}");
            let uninterrupted = search::run(&base, &train, &test);
            let uninterrupted_text = checkpoint_text(&base, &train, &test);
            let (plain, plain_text) = chunked(&base, false, 3, &train, &test);
            let (interim, interim_text) = chunked(&base, true, 3, &train, &test);
            assert_same_report(&plain, &interim, &format!("{what}: interim against plain chunks"));
            assert_same_report(&uninterrupted, &interim, &format!("{what}: interim against uninterrupted"));
            assert_eq!(plain_text, interim_text, "{what}: exit checkpoint bytes");
            assert_eq!(uninterrupted_text, interim_text, "{what}: exit checkpoint against uninterrupted");
        }
    }
}

/// #855: a chunk that stops on the generation budget is the last one in fact
/// and reports in full.
#[test]
fn interim_stop_on_generations_reports() {
    let train = sample(240);
    let test = sample(80);
    let cfg = Config {
        time_budget: Some(1.0e9),
        interim_chunk: true,
        ..config(11)
    };
    let mut text: Option<String> = None;
    let out = search::run_resumable(&cfg, &train, &test, None, 0, |cp| text = Some(cp.to_text()))
        .expect("run");
    assert_eq!(out.stop_reason, "generations");
    assert!(!out.interim);
    assert_eq!(out.test_errors.len(), out.archive.items.len());
    let cp = Checkpoint::from_text(&text.expect("checkpoint")).expect("parse");
    assert_eq!(cp.final_archive.expect("reported archive").hash(), out.archive.hash());
}

fn rcsr_in(dir: &std::path::Path, args: &[&str]) -> Output {
    rcsr_cmd::rcsr()
        .current_dir(dir)
        .args(args)
        .output()
        .expect("run rcsr")
}

/// The CLI arguments shared by every chunk of the #855 chains. Paths are
/// relative to the chunk's own directory so two chains write equal text.
fn chunk_args(csv: &str, extra: &[&'static str]) -> Vec<String> {
    let mut args: Vec<String> = [
        "--data", csv, "--target", "y", "--vars", "x", "--seed", "5", "--pop", "30",
        "--max-complexity", "8", "--cpu", "--no-dev-log", "--quiet", "--generations", "8",
        "--save", "ck.rcsr", "--json-out", "run.json",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.extend(extra.iter().map(|s| s.to_string()));
    args
}

fn run_chunk(dir: &std::path::Path, csv: &str, extra: &[&'static str]) -> Output {
    let args = chunk_args(csv, extra);
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    rcsr_in(dir, &refs)
}

/// `chunks` one-generation chunks (`--time-budget 0`), with or without
/// `--interim-chunk`, then one last chunk with `last`.
fn cli_chain(dir: &std::path::Path, csv: &str, interim: bool, chunks: usize, last: &[&'static str]) {
    for k in 0..chunks {
        let mut extra: Vec<&'static str> = vec!["--time-budget", "0"];
        if interim {
            extra.push("--interim-chunk");
        }
        if k > 0 {
            extra.extend(["--resume", "ck.rcsr"]);
        }
        let out = run_chunk(dir, csv, &extra);
        assert!(out.status.success(), "chunk {k} failed: {:?}", out);
    }
    let mut extra: Vec<&'static str> = vec!["--resume", "ck.rcsr"];
    extra.extend(last);
    let out = run_chunk(dir, csv, &extra);
    assert!(out.status.success(), "last chunk failed: {:?}", out);
}

/// `run.json` with every wall-clock number replaced, so two runs compare equal
/// exactly when only timing differs.
fn without_timing(json: &str) -> String {
    let bytes = json.as_bytes();
    let mut out = String::with_capacity(json.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            if let Some(len) = json[i + 1..].find('"') {
                let key = &json[i + 1..i + 1 + len];
                let after = i + 2 + len;
                let timed = key == "seconds" || key.ends_with("_seconds") || key.ends_with("_s");
                if timed && bytes.get(after) == Some(&b':') {
                    out.push_str(&json[i..=after]);
                    let mut j = after + 1;
                    while j < bytes.len() && !matches!(bytes[j], b',' | b'}' | b']') {
                        j += 1;
                    }
                    out.push('0');
                    i = j;
                    continue;
                }
                out.push_str(&json[i..after]);
                i = after;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

#[test]
fn without_timing_masks_only_timing_keys() {
    let json = r#"{"a":1,"elapsed_seconds":0.5,"timing":{"x_s":2,"y":3},"seconds":7}"#;
    assert_eq!(
        without_timing(json),
        r#"{"a":1,"elapsed_seconds":0,"timing":{"x_s":0,"y":3},"seconds":0}"#
    );
}

/// #855: through the CLI, the last chunk after interim chunks writes the same
/// checkpoint bytes and the same `run.json`, apart from timing, as a chain
/// without the flag.
#[test]
fn interim_chain_cli_matches_a_plain_chain() {
    let scratch = Scratch::new("interim-chain");
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let csv = csv.to_str().expect("utf-8").to_string();
    let (a, b) = (scratch.join("interim"), scratch.join("plain"));
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    for last in [&[][..], &["--time-budget", "0"][..]] {
        cli_chain(&a, &csv, true, 3, last);
        cli_chain(&b, &csv, false, 3, last);
        let ck_a = std::fs::read(a.join("ck.rcsr")).expect("interim checkpoint");
        let ck_b = std::fs::read(b.join("ck.rcsr")).expect("plain checkpoint");
        assert_eq!(ck_a, ck_b, "exit checkpoint bytes, last chunk {last:?}");
        let run_a = std::fs::read_to_string(a.join("run.json")).expect("interim run.json");
        let run_b = std::fs::read_to_string(b.join("run.json")).expect("plain run.json");
        assert!(run_a.contains("\"pareto\":["), "the last chunk reports in full");
        assert!(!run_a.contains("\"interim\""));
        assert_eq!(without_timing(&run_a), without_timing(&run_b), "run.json, last chunk {last:?}");
        for dir in [&a, &b] {
            std::fs::remove_file(dir.join("ck.rcsr")).unwrap();
        }
    }
}

/// #855: an interim chunk writes a short run.json, and the readers of a result
/// refuse it by name.
#[test]
fn interim_run_json_is_refused_by_evaluators() {
    let scratch = Scratch::new("interim-refuse");
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let csv = csv.to_str().expect("utf-8").to_string();
    let dir = scratch.join("run");
    std::fs::create_dir(&dir).unwrap();
    let out = run_chunk(&dir, &csv, &["--time-budget", "0", "--interim-chunk", "--run-manifest", "m.json"]);
    assert!(out.status.success(), "{:?}", out);
    let run = std::fs::read_to_string(dir.join("run.json")).expect("run.json");
    for key in ["\"interim\":true", "\"stop_reason\":\"time_budget\"", "\"generations_run\":1", "\"backend\":\"cpu\"", "\"threads\":", "\"checkpoint\":{"] {
        assert!(run.contains(key), "short run.json lacks {key}: {run}");
    }
    assert!(!run.contains("\"pareto\""), "an interim run.json holds no front");
    let ck = std::fs::read_to_string(dir.join("ck.rcsr")).expect("checkpoint");
    assert!(!ck.contains("\nfinal "), "an interim checkpoint holds no reported archive");

    let wf = rcsr_in(&dir, &["walkforward", "--data", &csv, "--from-run", "run.json"]);
    assert_eq!(wf.status.code(), Some(2), "{:?}", wf);
    assert!(String::from_utf8_lossy(&wf.stderr).contains("--interim-chunk"), "{:?}", wf);

    let vm = rcsr_in(&dir, &["verify-manifest", "m.json"]);
    assert_eq!(vm.status.code(), Some(2), "{:?}", vm);
    assert!(String::from_utf8_lossy(&vm.stderr).contains("--interim-chunk"), "{:?}", vm);

    let again = run_chunk(&dir, &csv, &["--resume", "ck.rcsr", "--run-manifest", "m.json"]);
    assert!(again.status.success(), "{:?}", again);
    let vm = rcsr_in(&dir, &["verify-manifest", "m.json"]);
    assert_eq!(vm.status.code(), Some(0), "the last chunk's manifest verifies: {:?}", vm);
}

/// #855: the flag needs a checkpoint to write and a budget to stop on.
#[test]
fn interim_chunk_needs_save_and_time_budget() {
    let scratch = Scratch::new("interim-usage");
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let csv = csv.to_str().expect("utf-8");
    for args in [
        vec!["--data", csv, "--target", "y", "--vars", "x", "--cpu", "--interim-chunk", "--time-budget", "1"],
        vec!["--data", csv, "--target", "y", "--vars", "x", "--cpu", "--interim-chunk", "--save", "ck.rcsr"],
    ] {
        let out = rcsr_in(scratch.path(), &args);
        assert_eq!(out.status.code(), Some(2), "{:?}", out);
        assert!(String::from_utf8_lossy(&out.stderr).contains("--interim-chunk requires"));
    }
}

/// #855: a process killed inside an interim chunk leaves a checkpoint that
/// resumes to the report of an uninterrupted run.
#[test]
fn a_kill_during_an_interim_chunk_resumes_to_the_uninterrupted_report() {
    let scratch = Scratch::new("interim-kill");
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let csv = csv.to_str().expect("utf-8").to_string();
    let (killed, whole) = (scratch.join("killed"), scratch.join("whole"));
    std::fs::create_dir(&killed).unwrap();
    std::fs::create_dir(&whole).unwrap();

    let mut args = chunk_args(&csv, &["--time-budget", "3600", "--interim-chunk", "--save-every", "1", "--patience", "0"]);
    let at = args.iter().position(|a| a == "--generations").expect("generations flag") + 1;
    args[at] = "1000000".to_string();
    let mut child = rcsr_cmd::rcsr()
        .current_dir(&killed)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn");
    let ck = killed.join("ck.rcsr");
    while !ck.exists() {
        assert!(child.try_wait().expect("poll").is_none(), "the chunk ended before it saved");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    child.kill().expect("kill the chunk");
    child.wait().expect("reap the chunk");

    let saved = Checkpoint::load(&ck).expect("a complete checkpoint survives the kill");
    assert!(saved.final_archive.is_none());
    // One more generation after the kill gives the report. A fresh run to that
    // generation is the reference.
    let run = |dir: &std::path::Path, generations: String, extra: &[&'static str]| {
        let mut args = chunk_args(&csv, &["--patience", "0"]);
        let at = args.iter().position(|a| a == "--generations").expect("generations flag") + 1;
        args[at] = generations;
        args.extend(extra.iter().map(|s| s.to_string()));
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let out = rcsr_in(dir, &refs);
        assert!(out.status.success(), "{:?}", out);
    };
    run(&killed, "1000000".to_string(), &["--resume", "ck.rcsr", "--time-budget", "0"]);
    run(&whole, (saved.generations + 1).to_string(), &[]);
    let final_line = |dir: &std::path::Path| {
        std::fs::read_to_string(dir.join("ck.rcsr"))
            .unwrap()
            .lines()
            .find(|l| l.starts_with("final "))
            .map(str::to_string)
            .expect("a final line")
    };
    assert_eq!(final_line(&killed), final_line(&whole));
    let front = |dir: &std::path::Path| {
        let text = std::fs::read_to_string(dir.join("run.json")).unwrap();
        text[text.find("\"pareto\":[").expect("a front")..].to_string()
    };
    assert_eq!(front(&killed), front(&whole));
}

/// #855: the same chain on the device, node path and flat path alike: the
/// last chunk after interim chunks equals the last chunk after plain chunks.
#[cfg(feature = "cuda")]
#[test]
fn interim_chunks_keep_the_final_hash_on_the_device() {
    if !rcsr::backend::probe().available() {
        rcsr::backend::skip_without_device("no usable GPU; skipping the device interim chain");
        return;
    }
    let train = sample(240);
    let test = sample(80);
    let mut base = Config {
        generations: 6,
        metric: rcsr::metrics::Metric::Squared,
        ..config(17)
    };
    base.gpu = true;
    base.dispatch = rcsr::dispatch::DispatchMode::Device;
    let (plain, plain_text) = chunked(&base, false, 2, &train, &test);
    let (interim, interim_text) = chunked(&base, true, 2, &train, &test);
    assert_eq!(interim.backend, rcsr::backend::Kind::Gpu);
    assert_same_report(&plain, &interim, "device interim against plain chunks");
    assert_eq!(plain_text, interim_text, "device exit checkpoint bytes");
}

/// An interim chunk's `run.json` in a fresh directory.
fn interim_run_dir(scratch: &Scratch) -> (PathBuf, String) {
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let csv = csv.to_str().expect("utf-8").to_string();
    let dir = scratch.join("run");
    std::fs::create_dir(&dir).unwrap();
    let out = run_chunk(&dir, &csv, &["--time-budget", "0", "--interim-chunk"]);
    assert!(out.status.success(), "{:?}", out);
    (dir, csv)
}

/// #855: `ablation score --run` names the interim flag instead of a missing `ablation` field.
#[test]
fn interim_run_json_is_refused_by_ablation_score() {
    let scratch = Scratch::new("interim-ablation");
    let (dir, csv) = interim_run_dir(&scratch);
    for flag in ["--run", "--from-run"] {
        let out = rcsr_in(&dir, &["ablation", "score", flag, "run.json", "--data", &csv, "--target", "y", "--time", "t"]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{flag}: {err}");
        assert!(err.contains("--interim-chunk"), "{flag}: {err}");
        assert!(!err.contains("no 'ablation' object"), "{flag}: {err}");
    }
}

/// #855: `ladder --sr-run` refuses the interim document by name.
#[test]
fn interim_run_json_is_refused_by_the_ladder() {
    let scratch = Scratch::new("interim-ladder");
    let (dir, csv) = interim_run_dir(&scratch);
    let out = rcsr_in(
        &dir,
        &["ladder", "--data", &csv, "--target", "y", "--time", "t", "--windows", "w.csv", "--sr-run", "run.json"],
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("--interim-chunk"), "{err}");
}

/// One chunk of the #923 chains: `--dev-log` replaces `--no-dev-log`, so each
/// chunk leaves its own `run_end`. `chunk` 0 starts fresh, the others resume.
fn cost_chunk(dir: &std::path::Path, csv: &str, chunk: usize, last: bool) -> Output {
    let mut args = chunk_args(csv, &[]);
    let at = args.iter().position(|a| a == "--no-dev-log").expect("dev log flag");
    args[at] = "--dev-log".to_string();
    args.insert(at + 1, format!("dev{chunk}.jsonl"));
    if !last {
        args.extend(["--time-budget", "0", "--interim-chunk"].map(String::from));
    }
    if chunk > 0 {
        args.extend(["--resume", "ck.rcsr"].map(String::from));
    }
    let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let out = rcsr_in(dir, &refs);
    assert!(out.status.success(), "chunk {chunk} failed: {:?}", out);
    out
}

fn json_file(path: std::path::PathBuf) -> rcsr::devlog::Json {
    rcsr::devlog::Json::parse(std::fs::read_to_string(path).expect("json file").trim()).expect("valid json")
}

fn run_end_of(dir: &std::path::Path, chunk: usize) -> rcsr::devlog::Json {
    let text = std::fs::read_to_string(dir.join(format!("dev{chunk}.jsonl"))).expect("dev log");
    let line = text.lines().rev().find(|l| l.contains("\"event\":\"run_end\"")).expect("run_end line");
    rcsr::devlog::Json::parse(line).expect("run_end json")
}

fn cost_scratch(tag: &str) -> (Scratch, std::path::PathBuf, String) {
    let scratch = Scratch::new(tag);
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, line_csv(60)).expect("csv");
    let dir = scratch.join("run");
    std::fs::create_dir(&dir).unwrap();
    (scratch, dir, csv.to_str().expect("utf-8").to_string())
}

/// #923: `search.cumulative` adds the `elapsed_seconds` of every chunk (the same value
/// as the chunk's run_end `search`, one clock), while
/// `elapsed_seconds` stays the last process alone.
#[test]
fn chunked_run_reports_cumulative_search_seconds() {
    let (_scratch, dir, csv) = cost_scratch("cumulative-cost");
    let mut elapsed_of_chunk = Vec::new();
    for chunk in 0..3 {
        cost_chunk(&dir, &csv, chunk, chunk == 2);
        let own = json_file(dir.join("run.json"));
        elapsed_of_chunk.push(own.get("elapsed_seconds").and_then(|v| v.as_f64()).expect("elapsed_seconds"));
    }
    let run = json_file(dir.join("run.json"));
    let cumulative = run.get("search").and_then(|s| s.get("cumulative")).expect("search.cumulative");
    assert!(matches!(cumulative.get("complete"), Some(rcsr::devlog::Json::Bool(true))), "{cumulative:?}");
    assert_eq!(cumulative.get("chunks").and_then(|v| v.as_usize()), Some(3));
    let total = cumulative.get("elapsed_seconds").and_then(|v| v.as_f64()).expect("elapsed_seconds");
    let sum: f64 = elapsed_of_chunk.iter().sum();
    assert!((total - sum).abs() <= 1e-9, "cumulative {total} against the chunks' own elapsed_seconds {elapsed_of_chunk:?}");
    let elapsed = run.get("elapsed_seconds").and_then(|v| v.as_f64()).expect("elapsed_seconds");
    assert_eq!(elapsed, elapsed_of_chunk[2], "the last chunk's elapsed_seconds is one term of the sum");
    let timing = run.get("timing").and_then(|t| t.get("search_seconds")).and_then(|v| v.as_f64()).expect("timing");
    assert_eq!(elapsed, timing, "elapsed_seconds keeps its per-process meaning");
    for (k, own) in elapsed_of_chunk.iter().enumerate() {
        let end = run_end_of(&dir, k);
        assert_eq!(end.get("search").and_then(|v| v.as_f64()), Some(*own), "run_end search is the chunk's elapsed_seconds, chunk {k}");
    }
    let end_total = run_end_of(&dir, 2).get("search_cumulative").and_then(|v| v.as_f64()).expect("run_end search_cumulative");
    assert!((end_total - total).abs() <= 1e-9, "{end_total} against {total}");
    assert!(dir.join("ck.rcsr.cost").exists(), "the sidecar sits next to the checkpoint");
}

/// #923: a run without `--resume` is its own total.
#[test]
fn fresh_run_cumulative_equals_its_own_elapsed_seconds() {
    let (_scratch, dir, csv) = cost_scratch("fresh-cost");
    cost_chunk(&dir, &csv, 0, true);
    let run = json_file(dir.join("run.json"));
    let cumulative = run.get("search").and_then(|s| s.get("cumulative")).expect("search.cumulative");
    assert_eq!(cumulative.get("chunks").and_then(|v| v.as_usize()), Some(1));
    assert_eq!(
        cumulative.get("elapsed_seconds").and_then(|v| v.as_f64()),
        run.get("elapsed_seconds").and_then(|v| v.as_f64())
    );
}

/// #923: a checkpoint from before the sidecar existed resumes, and the report
/// says the total is unknown.
#[test]
fn resume_without_sidecar_marks_cumulative_incomplete() {
    let (_scratch, dir, csv) = cost_scratch("no-sidecar-cost");
    cost_chunk(&dir, &csv, 0, false);
    std::fs::remove_file(dir.join("ck.rcsr.cost")).expect("sidecar written by the first chunk");
    cost_chunk(&dir, &csv, 1, true);
    let run = json_file(dir.join("run.json"));
    let cumulative = run.get("search").and_then(|s| s.get("cumulative")).expect("search.cumulative");
    assert!(matches!(cumulative.get("complete"), Some(rcsr::devlog::Json::Bool(false))), "{cumulative:?}");
    assert!(cumulative.get("elapsed_seconds").is_none());
    assert!(run_end_of(&dir, 1).get("search_cumulative").map_or(false, |v| v.as_f64().is_none()));
}

/// #923: a sidecar left by another generation (a crash between the two writes)
/// is not used.
#[test]
fn resume_ignores_sidecar_from_other_generation() {
    let (_scratch, dir, csv) = cost_scratch("other-generation-cost");
    cost_chunk(&dir, &csv, 0, false);
    let first = std::fs::read_to_string(dir.join("ck.rcsr.cost")).expect("sidecar");
    cost_chunk(&dir, &csv, 1, false);
    assert_ne!(std::fs::read_to_string(dir.join("ck.rcsr.cost")).unwrap(), first);
    std::fs::write(dir.join("ck.rcsr.cost"), first).unwrap();
    cost_chunk(&dir, &csv, 2, true);
    let run = json_file(dir.join("run.json"));
    let cumulative = run.get("search").and_then(|s| s.get("cumulative")).expect("search.cumulative");
    assert!(matches!(cumulative.get("complete"), Some(rcsr::devlog::Json::Bool(false))), "{cumulative:?}");
}

/// #1087: a saved run, then a resume with explicit flags. `extra` goes on the
/// save run, `resume` on the resume.
fn resume_with(tag: &str, extra: &[&str], resume: &[&str]) -> Output {
    let csv = TempPath::new("flags.csv");
    std::fs::write(&csv.0, line_csv(40)).expect("write csv");
    let checkpoint = TempPath::new(&format!("{tag}.rcsr"));
    let base = ["--data", csv.as_str(), "--target", "y", "--vars", "x", "--seed", "2", "--pop", "25"];
    let mut made_args = base.to_vec();
    made_args.extend(["--generations", "2", "--save", checkpoint.as_str()]);
    made_args.extend(extra);
    let made = rcsr(&made_args);
    assert!(made.status.success(), "save run failed: {}", String::from_utf8_lossy(&made.stderr));
    let mut resume_args = base.to_vec();
    resume_args.extend(["--generations", "3", "--resume", checkpoint.as_str()]);
    resume_args.extend(resume);
    rcsr(&resume_args)
}

fn assert_conflict(out: &Output, line: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{err}");
    assert!(err.contains("conflicts with the saved search settings:"), "{err}");
    assert!(err.lines().any(|l| l == format!("  - {line}")), "missing '{line}' in:\n{err}");
}

fn assert_resumed(out: &Output) {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn run_resumable_refuses_another_variation_mode() {
    let (train, test) = (sample(60), sample(20));
    let saved_cfg = Config { variation: search::VariationMode::Node, ..config(3) };
    let text = checkpoint_text(&saved_cfg, &train, &test);
    assert!(text.contains("config variation node\n"));
    let other = Config { variation: search::VariationMode::Auto, ..config(3) };
    let saved = Checkpoint::from_text(&text).expect("parse");
    let err = expect_err(search::run_resumable(&other, &train, &test, Some(saved), 0, |_| {}), "resume under auto");
    assert!(err.contains("variation"), "{err}");
    let saved = Checkpoint::from_text(&text).expect("parse");
    search::run_resumable(&saved_cfg, &train, &test, Some(saved), 0, |_| {}).expect("same mode resumes");
}

#[test]
fn cli_resume_with_an_explicit_variation_conflicts_or_matches() {
    assert_conflict(
        &resume_with("var-node", &["--variation", "node"], &["--variation", "auto"]),
        "--variation auto (saved variation is node)",
    );
    assert_conflict(
        &resume_with("var-auto", &[], &["--variation", "node"]),
        "--variation node (saved variation is auto)",
    );
    assert_resumed(&resume_with("var-same", &["--variation", "node"], &["--variation", "node"]));
    assert_resumed(&resume_with("var-same-auto", &[], &["--variation", "auto"]));
    assert_resumed(&resume_with("var-none", &["--variation", "node"], &[]));
}

#[test]
fn cli_resume_with_an_explicit_qd_flag_conflicts_or_matches() {
    let saved = ["--qd-cells", "8", "--qd-cell-cap", "2"];
    assert_conflict(&resume_with("qd-cells", &saved, &["--qd-cells", "32"]), "--qd-cells 32 (saved qd_bins is 8)");
    assert_conflict(&resume_with("qd-cap", &saved, &["--qd-cell-cap", "5"]), "--qd-cell-cap 5 (saved qd_cell_cap is 2)");
    assert_conflict(&resume_with("qd-cvt", &saved, &["--qd-cvt"]), "--qd-cvt (saved qd_cvt is false)");
    assert_conflict(
        &resume_with("qd-policy", &saved, &["--qd-policy"]),
        "--qd-policy (saved qd_policy is false, qd_cvt is false)",
    );
    assert_conflict(
        &resume_with("qd-uniform", &saved, &["--qd-uniform"]),
        "--qd-uniform (saved qd_policy is false, qd_control is 0, qd_cvt is false)",
    );
    assert_conflict(
        &resume_with("qd-random", &saved, &["--qd-random"]),
        "--qd-random (saved qd_policy is false, qd_control is 0)",
    );
    assert_conflict(&resume_with("qd-off", &[], &["--qd"]), "--qd (saved qd_bins is 0)");
    assert_resumed(&resume_with("qd-same", &saved, &["--qd-cells", "8", "--qd-cell-cap", "2", "--qd"]));
    assert_resumed(&resume_with("qd-none", &saved, &[]));
    assert_resumed(&resume_with("qd-policy-same", &["--qd-policy"], &["--qd-policy", "--qd-cvt", "--qd-cells", "512"]));
}

#[test]
fn cli_resume_with_explicit_units_flags_conflicts_or_matches() {
    let units = ["--units", "x=m,y=m"];
    assert_conflict(
        &resume_with("units-other", &units, &["--units", "x=s,y=m"]),
        "--units x=s,y=m (saved units is m, units.target is m)",
    );
    assert_conflict(&resume_with("units-new", &[], &["--units", "x=m,y=m"]), "--units x=m,y=m (saved units is none)");
    assert_resumed(&resume_with("units-same", &units, &["--units", "x=m,y=m"]));
    assert_resumed(&resume_with("units-none", &units, &[]));
    let required = ["--units", "x=m,y=m", "--require-dimensions"];
    assert_conflict(
        &resume_with("req-new", &units, &["--require-dimensions"]),
        "--require-dimensions (saved require_dimensions is false)",
    );
    assert_resumed(&resume_with("req-same", &required, &["--require-dimensions"]));
    assert_resumed(&resume_with("req-none", &required, &[]));
}
