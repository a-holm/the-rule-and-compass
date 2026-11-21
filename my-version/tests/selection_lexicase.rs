//! #1023: epsilon-lexicase parent selection (`--selection lexicase`).

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::search::{self, Config, Selection};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b * a + 1.0 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

fn config(selection: Selection, generations: usize, threads: usize) -> Config {
    Config {
        generations,
        islands: 2,
        pop: 60,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        migration_interval: 3,
        seed: 5,
        threads,
        selection,
        lexicase_cases: 64,
        gpu: false,
        ..Default::default()
    }
}

fn archive_hash(cfg: &Config, train: &Sample, test: &Sample) -> u64 {
    search::run(cfg, train, test).archive.hash()
}

#[test]
fn lexicase_is_deterministic_and_independent_of_threads() {
    let train = sample(300);
    let test = sample(80);
    let one = config(Selection::Lexicase, 12, 1);
    let first = archive_hash(&one, &train, &test);
    assert_eq!(first, archive_hash(&one, &train, &test), "same seed, same archive");
    assert_eq!(
        first,
        archive_hash(&config(Selection::Lexicase, 12, 8), &train, &test),
        "the thread count changed the archive"
    );
    assert_ne!(
        first,
        archive_hash(&config(Selection::Lexicographic, 12, 1), &train, &test),
        "lexicase must change the trajectory"
    );
    assert_ne!(
        first,
        archive_hash(&Config { lexicase_cases: 16, ..one.clone() }, &train, &test),
        "the case count is part of the trajectory"
    );
}

fn final_state(cfg: &Config, train: &Sample, test: &Sample) -> String {
    let mut text: Option<String> = None;
    search::run_resumable(cfg, train, test, None, 1, |cp| text = Some(cp.to_text())).expect("run");
    text.expect("a checkpoint")
}

/// The selection and a non-default case count are saved; the default count
/// writes no key, and another selection writes neither.
#[test]
fn lexicase_checkpoints_carry_the_selection_and_the_case_count() {
    let train = sample(200);
    let test = sample(50);
    let text = final_state(&config(Selection::Lexicase, 3, 1), &train, &test);
    assert!(text.contains("config selection lexicase\n"), "{text}");
    assert!(text.contains("config lexicase_cases 64\n"), "{text}");
    let cp = Checkpoint::from_text(&text).expect("parse");
    assert_eq!((cp.config.selection, cp.config.lexicase_cases), (Selection::Lexicase, 64));
    let default = Config { lexicase_cases: search::DEFAULT_LEXICASE_CASES, ..config(Selection::Lexicase, 3, 1) };
    assert!(!final_state(&default, &train, &test).contains("lexicase_cases"));
    let other = Config { selection: Selection::Lexicographic, ..config(Selection::Lexicase, 3, 1) };
    let text = final_state(&other, &train, &test);
    assert!(!text.contains("selection") && !text.contains("lexicase"), "{text}");
}

fn rcsr_in(dir: &Scratch, csv: &str, args: &[&str]) -> std::process::Output {
    let data = dir.join("data.csv");
    std::fs::write(&data, csv).expect("write data");
    rcsr_cmd::rcsr()
        .arg("--data")
        .arg(&data)
        .args(["--generations", "2", "--pop", "20", "--quiet", "--no-dev-log"])
        .args(args)
        .output()
        .expect("run rcsr")
}

fn usage_error(out: &std::process::Output, needle: &str) {
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{needle}: {err}");
    assert!(err.contains(needle), "want '{needle}' in: {err}");
}

fn table(n: usize) -> String {
    let mut text = String::from("x,w,y,y2\n");
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        text.push_str(&format!("{:.6},{},{:.6},{:.6}\n", x, 1 + i % 3, 2.0 * x + 1.0, x * x));
    }
    text
}

/// The refusals of the issue's decisions 1, 2 and 6 exit 2 with the named reason.
#[test]
fn lexicase_refuses_metrics_without_a_row_loss() {
    let dir = Scratch::new("lexicase-refusals");
    let csv = table(40);
    let lex = ["--target", "y", "--vars", "x", "--cpu", "--selection", "lexicase"];
    let with = |extra: &[&str]| {
        let mut args: Vec<&str> = lex.to_vec();
        args.extend_from_slice(extra);
        rcsr_in(&dir, &csv, &args)
    };
    for metric in ["median", "interquartile", "maximum", "correlation", "correlation_signed", "signed", "rank_correlation"] {
        usage_error(&with(&["--metric", metric]), &format!("--selection lexicase needs a per-row loss; {} has none", metric));
    }
    usage_error(&with(&["--weight", "w"]), "--selection lexicase needs unweighted rows");
    let multi = rcsr_in(&dir, &csv, &["--target", "y,y2", "--vars", "x", "--cpu", "--selection", "lexicase"]);
    usage_error(&multi, "--selection lexicase is not supported with multiple targets");
    usage_error(&with(&["--variation", "postfix"]), "--selection lexicase runs on the node path only");
    usage_error(&with(&["--lexicase-cases", "1"]), "--lexicase-cases needs an integer from 2 to 4096, got '1'");
    usage_error(&with(&["--lexicase-cases", "5000"]), "--lexicase-cases needs an integer from 2 to 4096");
    // The device refusals come before the device probe, so they hold on any build.
    let gpu = ["--target", "y", "--vars", "x", "--gpu", "--selection", "lexicase"];
    usage_error(&rcsr_in(&dir, &csv, &gpu), "--selection lexicase with --gpu: the device scorer returns no per-row case errors yet");
    let pnl = ["--target", "y", "--vars", "x", "--gpu", "--selection", "lexicase", "--fitness", "pnl"];
    usage_error(&rcsr_in(&dir, &csv, &pnl), "the device long_pnl fitness returns no sub-book cases yet (#1031)");
    for metric in ["squared", "absolute", "log_squashed", "aic", "bic"] {
        let out = with(&["--metric", metric]);
        assert!(out.status.success(), "{metric}: {}", String::from_utf8_lossy(&out.stderr));
    }
}

/// `--help` and `rcsr schema` document both flags.
#[test]
fn help_and_schema_list_lexicase() {
    let help = rcsr_cmd::rcsr().arg("--help").output().expect("help");
    let help = String::from_utf8_lossy(&help.stdout);
    assert!(help.contains("--selection lexicographic|afp|lexicase"), "{help}");
    assert!(help.contains("--lexicase-cases <K>") && help.contains("default 256"), "{help}");
    let schema = rcsr_cmd::rcsr().arg("schema").output().expect("schema");
    let schema = String::from_utf8_lossy(&schema.stdout);
    assert!(schema.contains("\"--lexicase-cases\""), "{schema}");
    assert!(schema.contains("epsilon-lexicase"), "{schema}");
}

/// A two-symbol panel and its target spec for `--fitness pnl`.
fn panel(dir: &Scratch) -> (String, String) {
    let frac = |v: f64| v - v.floor();
    let mut text = String::from("t,sym,x1,x2,r,y\n");
    for s in 0..2 {
        for i in 0..300 {
            let f = (i + 1000 * s) as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            let r = 0.01 * x1 + 0.0005 * noise;
            text.push_str(&format!("{},{},{},{},{},{}\n", 1_600_000_000 + 900 * i as i64, ["a", "b"][s], x1, x2, r, r + 0.05 * x2));
        }
    }
    let data = dir.join("panel.csv");
    std::fs::write(&data, text).expect("write panel");
    let spec = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":900,\"entry_offset_seconds\":900,\"delay_seconds\":0,\
         \"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("spec.json");
    std::fs::write(&path, spec).expect("write spec");
    (data.to_string_lossy().into_owned(), path.to_string_lossy().into_owned())
}

/// Both cost limits depend on the flags and the data shape only, so the CLI
/// refuses them as usage errors before the search: `pop x K` above 2^24 under
/// the metric fitness, and kept sub-book values (`islands x pop x sub-books`)
/// above 2^28 under `--fitness pnl`.
#[test]
fn lexicase_cost_limits_exit_two() {
    let dir = Scratch::new("lexicase-limits");
    let csv = table(400);
    let matrix = rcsr_in(&dir, &csv, &["--target", "y", "--vars", "x", "--cpu", "--selection", "lexicase", "--pop", "65600"]);
    usage_error(&matrix, "--selection lexicase compares 65600 members on 256 cases per island, above the 16777216 cell limit");

    let (data, spec) = panel(&dir);
    let stored = rcsr_cmd::rcsr()
        .args(["--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "sym"])
        .args(["--split", "temporal", "--validation-frac", "0.3", "--target-spec", &spec, "--objective", "pnl"])
        .args(["--cost-bps", "5", "--fitness", "pnl", "--pnl-blocks", "4", "--cpu", "--quiet", "--no-dev-log"])
        .args(["--generations", "1", "--selection", "lexicase", "--lexicase-cases", "2", "--islands", "8", "--pop", "4200000"])
        .output()
        .expect("run rcsr");
    usage_error(&stored, "--selection lexicase keeps 268800000 sub-book case errors (8 islands x 4200000 members x 8 cases), above the 268435456 limit");
}
