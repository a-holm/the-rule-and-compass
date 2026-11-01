//! `rcsr-bench` — the benchmark harness for the engine.
//!
//! Runs the symbolic-regression engine through the library API on suites of
//! problems whose true law is known (`known-truth`, `hard`) or whose signal is
//! weak and noisy (`signal`), and reports whether the engine *recovered* the
//! law rather than overfitting the sample. See `src/bench.rs` and
//! `src/bench_suites.rs` for the suites and `docs/benchmarks.md` for the
//! criteria and recorded results.
//!
//! ```text
//! rcsr-bench --suite known-truth --noise 0.0,0.01,0.1
//! rcsr-bench --suite hard --seeds 1-10 --cpu --ndjson > hard.ndjson
//! rcsr-bench compare --base base.ndjson --treat treat.ndjson
//! rcsr-bench --list --suite all
//! ```
//!
//! Everything machine-readable goes to stdout; the text report goes to stdout
//! too, since this tool's whole output *is* the result. Usage errors exit `2`.

use rcsr::bench::{self, BackendChoice, FitChoice, SuiteConfig};
use rcsr::bench_stats::{self, Criterion};
use rcsr::bench_suites;
use rcsr::metrics::Metric;
use std::io::Write;

const EXIT_OK: i32 = 0;
const EXIT_USAGE: i32 = 2;

fn usage() -> String {
    format!(
        "rcsr-bench {} — benchmarks for rule-and-compass

USAGE:
  rcsr-bench [--suite <name>] [options]
  rcsr-bench compare --base <a.ndjson[,..]> --treat <b.ndjson[,..]> [options]
  rcsr-bench --list [--suite <name>]
  rcsr-bench --help

OPTIONS:
  --suite <name>            known-truth (default), hard, signal, all; a comma
                            list runs several
  --problems <a,b,..>       run only these problem names (unknown name: exit 2)
  --noise <a,b,..>          relative noise levels to sweep
                            (default 0.0,0.01,0.1; 0.01 = 1% of the target sd);
                            the signal suite has no noise sweep and ignores it
  --tiny                    run only the fast known-truth subset (CI-friendly)
  --generations <N>         search generations per run
                            (default 150; 30 with --tiny)
  --islands <I>             islands per run (default 2; 1 with --tiny)
  --pop <P>                 population per island (default 200; 80 with --tiny)
  --seed <S>                engine RNG seed; the same seed reproduces a run
                            exactly (default 0)
  --seeds <list>            several seeds: 1,2,3 or 1-10 (overrides --seed)
  --metric <name>           search metric (default squared)
  --threshold <r2>          test R^2 counted as recovered (default 0.99)
  --time-budget <seconds>   wall-clock limit per run (default: none)
  --threads <N>             worker threads for fitting and scoring inside an island
                            (default: the available CPU cores; 1 = serial; the
                            results do not depend on it)
  --no-front-parents        disable the 15% Pareto-front parent feedback (#76)
  --prune                   enable the front prune (#466; off by default)
  --no-prune                keep the prune off (the default)
  --reject-constant-offspring  reject input-free offspring before the constant fit (#477)
  --cpu                     force the CPU evaluator (no environment variable)
  --gpu                     require the device evaluator; exit 2 without one
  --fit-strategy <name>     exact (bench default), auto, linear, varpro, stream,
                            gpu, or default (resolve like the CLI does)
  --fit-iters <N>           constant-fit iterations
  --ops <a,b,..>            operator set
  --max-complexity <N>      maximum expression complexity
  --patience <N>            generations without improvement before stopping
  --migration-interval <N>  generations between island migrations
  --restart-interval <N>    generations between island restarts
  --variation <mode>        auto, node or postfix
  --selection <mode>        lexicographic (default) or afp (age-fitness Pareto)
  --linear-scaling          wrap every candidate in a fitted affine output map (#470)
  --mutation-mix <mix>      classic (default) or extended: insert, delete, swap, rotate (#471)
  --tree-arity <mode>       uniform (default) or balanced random-tree operator draw (#471)
  --crossover-internal <P>  probability of an internal-node crossover cut (default 0) (#471)
  --adaptive-mix            per-island edit weights from success rates (#471)
  --dispatch <mode>         auto or device (device skips the timing comparison)
  --fit-polish <mode>      f64 (default), f32 or off: arithmetic of the device fit
                            kernel polish (--fit-strategy gpu)
  --jobs <N>                runs in flight at once on the CPU (default 1);
                            seconds and hit_s are then contended
  --ndjson                  one JSON line per finished run, flushed at once,
                            then one summary line
  --json                    print one JSON object instead of the text table
  --list                    list the suite's problems and exit
  -h, --help                show this help

RECOVERY:
  A problem is recovered when the engine's Pareto front contains an expression
  whose R^2 on fresh noiseless data is at least --threshold. The reported R^2,
  complexity and test error belong to the best such front member (ties go to the
  simplest expression).

ORACLE CRITERION (known-truth, hard):
  The truth structure c0 + c1*truth is refit on the training rows with every
  constant free. A run is oracle_recovered when
  1 - R^2 <= 2 (1 - oracle R^2) + 1e-9 for some front member. At noise 0 this
  demands the law; with noise it demands what the true structure can reach.
  exact_cx is the complexity of the simplest such member, excess_cx its
  difference to truth_cx.

SIGNAL TASKS:
  The front member with the lowest error on the validation rows is selected.
  signal_r2 is its R^2 against the noiseless test signal; signal_recovered
  means signal_r2 >= 0.8.

TIME TO HIT:
  hit_gen, hit_s and hit_evals (generation, search seconds, evaluated
  candidates) are the first point at which the front held a recovered member;
  oracle_hit_* the same for the oracle criterion. null when never reached.

COMPARE OPTIONS:
  --base <files>            baseline NDJSON files (comma list, merged)
  --treat <files>           treatment NDJSON files (comma list, merged)
  --criterion <name>        recovered (default), oracle_recovered or
                            signal_recovered
  --json                    machine-readable output
  Runs are paired by (suite, problem, noise, seed). The command exits 2 when
  the two sides hold different runs or differ in anything but the search knobs.
  Pairs whose criterion is null on a side are dropped, counted and listed.
",
        env!("CARGO_PKG_VERSION")
    )
}

struct Args {
    suites: Vec<String>,
    problems: Vec<String>,
    noise: Vec<f64>,
    tiny: bool,
    generations: Option<usize>,
    islands: Option<usize>,
    pop: Option<usize>,
    seed: u32,
    seeds: Vec<u32>,
    metric: Metric,
    threshold: f64,
    time_budget: Option<f64>,
    json: bool,
    ndjson: bool,
    list: bool,
    front_parents: bool,
    cpu: bool,
    gpu: bool,
    jobs: usize,
    knobs: bench::Knobs,
    threads: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            suites: vec!["known-truth".to_string()],
            problems: Vec::new(),
            noise: vec![0.0, 0.01, 0.1],
            tiny: false,
            generations: None,
            islands: None,
            pop: None,
            seed: 0,
            seeds: Vec::new(),
            metric: Metric::Squared,
            threshold: bench::DEFAULT_THRESHOLD,
            time_budget: None,
            json: false,
            ndjson: false,
            list: false,
            front_parents: true, // #76
            cpu: false,
            gpu: false,
            jobs: 1,
            knobs: bench::Knobs::default(),
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        }
    }
}

fn parse_seeds(v: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let a: u32 = a.parse().map_err(|_| format!("--seeds: bad range '{}'", part))?;
            let b: u32 = b.parse().map_err(|_| format!("--seeds: bad range '{}'", part))?;
            if a > b {
                return Err(format!("--seeds: empty range '{}'", part));
            }
            out.extend(a..=b);
        } else {
            out.push(
                part.parse()
                    .map_err(|_| format!("--seeds: '{}' is not an integer", part))?,
            );
        }
    }
    if out.is_empty() {
        return Err("--seeds: no seeds given".to_string());
    }
    Ok(out)
}

/// Parse the command line. Returns `Err(message)` for a usage error.
fn parse(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 0;
    let take = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| format!("{} needs a value", name))
    };
    fn num<T: std::str::FromStr>(v: String, name: &str) -> Result<T, String> {
        v.parse().map_err(|_| format!("{}: '{}' is not a number", name, v))
    }
    while i < args.len() {
        match args[i].as_str() {
            "--suite" => {
                let v = take(&mut i, "--suite")?;
                a.suites = v
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                bench::resolve_suites(&a.suites).map_err(|e| format!("--suite: {}", e))?;
            }
            "--problems" => {
                a.problems = take(&mut i, "--problems")?
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
                if a.problems.is_empty() {
                    return Err("--problems: no names given".to_string());
                }
            }
            "--noise" => {
                let v = take(&mut i, "--noise")?;
                a.noise = v
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| format!("--noise: '{}' is not a number", s.trim()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if a.noise.is_empty() {
                    return Err("--noise: no levels given".to_string());
                }
                if a.noise.iter().any(|&n| !(0.0..1.0).contains(&n)) {
                    return Err("--noise: levels must lie in [0, 1)".to_string());
                }
            }
            "--tiny" => a.tiny = true,
            "--generations" => a.generations = Some(num(take(&mut i, "--generations")?, "--generations")?),
            "--islands" => a.islands = Some(num(take(&mut i, "--islands")?, "--islands")?),
            "--pop" => a.pop = Some(num(take(&mut i, "--pop")?, "--pop")?),
            "--seed" => a.seed = num(take(&mut i, "--seed")?, "--seed")?,
            "--seeds" => a.seeds = parse_seeds(&take(&mut i, "--seeds")?)?,
            "--metric" => {
                let v = take(&mut i, "--metric")?;
                a.metric = Metric::parse_opt(&v)
                    .ok_or_else(|| format!("--metric: unknown metric '{}'", v))?;
            }
            "--threshold" => a.threshold = num(take(&mut i, "--threshold")?, "--threshold")?,
            "--time-budget" => a.time_budget = Some(num(take(&mut i, "--time-budget")?, "--time-budget")?),
            "--json" => a.json = true,
            "--ndjson" => a.ndjson = true,
            "--no-front-parents" => a.front_parents = false,
            "--prune" => a.knobs.prune = Some(true),
            "--no-prune" => a.knobs.prune = Some(false),
            "--cpu" => a.cpu = true,
            "--gpu" => a.gpu = true,
            "--fit-strategy" => {
                let v = take(&mut i, "--fit-strategy")?;
                a.knobs.fit_strategy = Some(if v == "default" {
                    FitChoice::Default
                } else {
                    FitChoice::Strategy(rcsr::fit::FitStrategy::parse(&v).ok_or_else(|| {
                        format!(
                            "--fit-strategy needs exact, auto, linear, varpro, stream, gpu or default, got '{}'",
                            v
                        )
                    })?)
                });
            }
            "--fit-iters" => a.knobs.fit_iters = Some(num(take(&mut i, "--fit-iters")?, "--fit-iters")?),
            "--ops" => {
                let v = take(&mut i, "--ops")?;
                rcsr::expr::Op::parse_list_checked(&v).map_err(|e| format!("--ops: {}", e))?;
                a.knobs.ops = Some(v);
            }
            "--max-complexity" => {
                a.knobs.max_complexity = Some(num(take(&mut i, "--max-complexity")?, "--max-complexity")?)
            }
            "--patience" => a.knobs.patience = Some(num(take(&mut i, "--patience")?, "--patience")?),
            "--migration-interval" => {
                a.knobs.migration_interval =
                    Some(num(take(&mut i, "--migration-interval")?, "--migration-interval")?)
            }
            "--restart-interval" => {
                a.knobs.restart_interval =
                    Some(num(take(&mut i, "--restart-interval")?, "--restart-interval")?)
            }
            "--linear-scaling" => a.knobs.linear_scaling = true,
            "--mutation-mix" => {
                a.knobs.mutation_mix = Some(rcsr::search::MutationMix::parse(&take(&mut i, "--mutation-mix")?)?)
            }
            "--tree-arity" => {
                a.knobs.tree_arity = Some(rcsr::search::TreeArity::parse(&take(&mut i, "--tree-arity")?)?)
            }
            "--crossover-internal" => {
                a.knobs.crossover_internal =
                    Some(num(take(&mut i, "--crossover-internal")?, "--crossover-internal")?)
            }
            "--adaptive-mix" => a.knobs.adaptive_mix = true,
            "--variation" => {
                a.knobs.variation = Some(rcsr::search::VariationMode::parse(&take(&mut i, "--variation")?)?)
            }
            "--reject-constant-offspring" => a.knobs.reject_constant_offspring = true,
            "--selection" => {
                a.knobs.selection = Some(rcsr::search::Selection::parse(&take(&mut i, "--selection")?)?)
            }
            "--fit-polish" => {
                let v = take(&mut i, "--fit-polish")?;
                a.knobs.fit_polish = Some(
                    rcsr::fit::FitPolish::parse(&v)
                        .ok_or_else(|| format!("--fit-polish needs f64, f32 or off, got '{}'", v))?,
                );
            }
            "--dispatch" => {
                let v = take(&mut i, "--dispatch")?;
                a.knobs.dispatch = Some(
                    rcsr::dispatch::DispatchMode::parse(&v)
                        .ok_or_else(|| format!("--dispatch needs auto or device, got '{}'", v))?,
                );
            }
            "--jobs" => {
                a.jobs = num(take(&mut i, "--jobs")?, "--jobs")?;
                if a.jobs == 0 {
                    return Err("--jobs: must be at least 1".to_string());
                }
            }
            "--threads" => a.threads = num(take(&mut i, "--threads")?, "--threads")?,
            "--list" => a.list = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown option '{}'", other)),
        }
        i += 1;
    }
    if a.cpu && a.gpu {
        return Err("--cpu and --gpu are exclusive".to_string());
    }
    if a.json && a.ndjson {
        return Err("--json and --ndjson are exclusive".to_string());
    }
    Ok(a)
}

fn backend_of(a: &Args) -> BackendChoice {
    if a.cpu {
        BackendChoice::Cpu
    } else if a.gpu {
        BackendChoice::Gpu
    } else {
        BackendChoice::Auto
    }
}

fn to_config(a: &Args) -> SuiteConfig {
    SuiteConfig {
        noise: a.noise.clone(),
        generations: a.generations.unwrap_or(if a.tiny { 30 } else { 150 }),
        islands: a.islands.unwrap_or(if a.tiny { 1 } else { 2 }),
        pop: a.pop.unwrap_or(if a.tiny { 80 } else { 200 }),
        seed: a.seed,
        metric: a.metric,
        tiny: a.tiny,
        threshold: a.threshold,
        time_budget: a.time_budget,
        front_parents: a.front_parents,
        suites: a.suites.clone(),
        problems: a.problems.clone(),
        seeds: a.seeds.clone(),
        backend: backend_of(a),
        knobs: a.knobs.clone(),
        jobs: a.jobs,
        threads: a.threads,
    }
}

fn list_problems(suites: &[String]) -> String {
    let mut out = String::new();
    let names = bench::resolve_suites(suites).unwrap_or_default();
    for suite in names {
        if suite == "known-truth" {
            out.push_str(&format!(
                "{:<20} {:>4}  {:<28} {}\n",
                "problem", "tiny", "variables", "truth"
            ));
            for p in bench::known_truth() {
                out.push_str(&format!(
                    "{:<20} {:>4}  {:<28} {}\n",
                    p.name,
                    if p.tiny { "yes" } else { "" },
                    p.vars.join(","),
                    p.truth_string()
                ));
            }
        } else if suite == "hard" {
            out.push_str(&format!(
                "\n{:<20} {:>8}  {:<28} {}\n",
                "hard problem", "truth_cx", "variables", "truth"
            ));
            for p in bench_suites::hard() {
                out.push_str(&format!(
                    "{:<20} {:>8}  {:<28} {}\n",
                    p.name,
                    p.truth_cx,
                    p.vars.join(","),
                    p.truth_string()
                ));
            }
        } else {
            out.push_str(&format!(
                "\n{:<20} {:>9} {:>8}  {:<30} {}\n",
                "signal task", "oracle_r2", "truth_cx", "variables", "signal"
            ));
            for t in bench_suites::signal() {
                out.push_str(&format!(
                    "{:<20} {:>9.2} {:>8}  {:<30} {}\n",
                    t.name,
                    t.oracle_r2,
                    t.truth_cx,
                    t.vars.join(","),
                    t.truth_string()
                ));
            }
        }
    }
    out
}

fn compare_main(args: &[String]) -> i32 {
    let mut base: Vec<String> = Vec::new();
    let mut treat: Vec<String> = Vec::new();
    let mut criterion = Criterion::Recovered;
    let mut json = false;
    let mut i = 0;
    let fail = |msg: String| -> i32 {
        eprintln!("rcsr-bench compare: {}", msg);
        EXIT_USAGE
    };
    while i < args.len() {
        let flag = args[i].as_str();
        let value = args.get(i + 1).cloned();
        let step: Result<(), String> = match flag {
            "--base" | "--treat" | "--criterion" => match value {
                None => Err(format!("{} needs a value", flag)),
                Some(v) => {
                    i += 1;
                    match flag {
                        "--base" => {
                            base = split_files(&v);
                            Ok(())
                        }
                        "--treat" => {
                            treat = split_files(&v);
                            Ok(())
                        }
                        _ => Criterion::parse(&v)
                            .map(|c| criterion = c)
                            .ok_or_else(|| format!("--criterion: unknown criterion '{}'", v)),
                    }
                }
            },
            "--json" => {
                json = true;
                Ok(())
            }
            "-h" | "--help" => {
                print!("{}", usage());
                return EXIT_OK;
            }
            other => Err(format!("unknown option '{}'", other)),
        };
        if let Err(e) = step {
            return fail(e);
        }
        i += 1;
    }
    if base.is_empty() || treat.is_empty() {
        return fail("--base and --treat are required".to_string());
    }
    let read = |files: &[String]| -> Result<Vec<(String, String)>, String> {
        files
            .iter()
            .map(|f| {
                std::fs::read_to_string(f)
                    .map(|t| (f.clone(), t))
                    .map_err(|e| format!("{}: {}", f, e))
            })
            .collect()
    };
    let (b, t) = match (read(&base), read(&treat)) {
        (Ok(b), Ok(t)) => (b, t),
        (Err(e), _) | (_, Err(e)) => return fail(e),
    };
    match bench_stats::compare(&b, &t, criterion) {
        Ok(report) => {
            if json {
                print!("{}", report.to_json());
            } else {
                print!("{}", report.to_text());
            }
            EXIT_OK
        }
        Err(e) => fail(e),
    }
}

fn split_files(v: &str) -> Vec<String> {
    v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("compare") {
        std::process::exit(compare_main(&args[1..]));
    }
    let parsed = match parse(&args) {
        Ok(a) => a,
        Err(msg) if msg.is_empty() => {
            print!("{}", usage());
            std::process::exit(EXIT_OK);
        }
        Err(msg) => {
            eprintln!("rcsr-bench: {}\n\n{}", msg, usage());
            std::process::exit(EXIT_USAGE);
        }
    };

    if parsed.list {
        print!("{}", list_problems(&parsed.suites));
        std::process::exit(EXIT_OK);
    }

    let device = rcsr::backend::probe().available();
    if parsed.gpu && !device {
        eprintln!("rcsr-bench: --gpu needs a CUDA device, none is available");
        std::process::exit(EXIT_USAGE);
    }
    if parsed.jobs > 1 && (parsed.gpu || (!parsed.cpu && device)) {
        eprintln!("rcsr-bench: --jobs > 1 needs --cpu (a device backend would be shared)");
        std::process::exit(EXIT_USAGE);
    }

    let cfg = to_config(&parsed);
    let specs = match bench::plan(&cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rcsr-bench: {}\n\n{}", e, usage());
            std::process::exit(EXIT_USAGE);
        }
    };
    let results = if parsed.ndjson {
        let stdout = std::io::stdout();
        let cfg_ref = cfg.clone();
        let results = bench::run_plan(&specs, &cfg, &mut |r| {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{}", bench::run_ndjson(r, &cfg_ref));
            let _ = out.flush();
        });
        let report = bench::build_report(results, &cfg);
        let mut out = stdout.lock();
        let _ = writeln!(out, "{}", bench::summary_ndjson(&report, &cfg));
        let _ = out.flush();
        std::process::exit(EXIT_OK);
    } else {
        bench::run_plan(&specs, &cfg, &mut |_| {})
    };
    let report = bench::build_report(results, &cfg);
    if parsed.json {
        print!("{}", bench::report_json(&report, &cfg));
    } else {
        print!("{}", bench::report_text(&report, &cfg));
    }
    std::process::exit(EXIT_OK);
}
