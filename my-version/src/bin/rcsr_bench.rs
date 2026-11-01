//! `rcsr-bench` — the known-truth benchmark harness.
//!
//! Runs the symbolic-regression engine through the library API on a suite of
//! problems whose true law is known, and reports whether the engine *recovered*
//! each law rather than overfitting the sample. See `src/bench.rs` for the
//! suite definition and `docs/benchmarks.md` for the recovery criterion and
//! recorded results.
//!
//! ```text
//! rcsr-bench --suite known-truth --noise 0.0,0.01,0.1
//! rcsr-bench --suite known-truth --tiny --json
//! rcsr-bench --list
//! ```
//!
//! Everything machine-readable goes to stdout; the text report goes to stdout
//! too, since this tool's whole output *is* the result. Usage errors exit `2`.

use rcsr::bench::{self, SuiteConfig};
use rcsr::metrics::Metric;

const EXIT_OK: i32 = 0;
const EXIT_USAGE: i32 = 2;

fn usage() -> String {
    format!(
        "rcsr-bench {} — known-truth benchmarks for rule-and-compass

USAGE:
  rcsr-bench --suite known-truth [options]
  rcsr-bench --list
  rcsr-bench --help

OPTIONS:
  --suite <name>            problem suite (default and only value: known-truth)
  --noise <a,b,..>          relative noise levels to sweep
                            (default 0.0,0.01,0.1; 0.01 = 1% of the target sd)
  --tiny                    run only the fast subset (CI-friendly)
  --generations <N>         search generations per run
                            (default 150; 30 with --tiny)
  --islands <I>             islands per run (default 2; 1 with --tiny)
  --pop <P>                 population per island (default 200; 80 with --tiny)
  --seed <S>                engine RNG seed; the same seed reproduces a run
                            exactly (default 0)
  --metric <name>           search metric (default squared)
  --threshold <r2>          test R^2 counted as recovered (default 0.99)
  --time-budget <seconds>   wall-clock limit per run (default: none)
  --threads <N>             worker threads for fitting and scoring inside an island
                            (default: the available CPU cores; 1 = serial; the
                            results do not depend on it)
  --no-front-parents        disable the 15% Pareto-front parent feedback (#76)
  --json                    print one JSON object instead of the text table
  --list                    list the suite's problems and exit
  -h, --help                show this help

RECOVERY:
  A problem is recovered when the engine's Pareto front contains an expression
  whose R^2 on fresh noiseless data is at least --threshold. The reported R^2,
  complexity and test error belong to the best such front member (ties go to the
  simplest expression).
",
        env!("CARGO_PKG_VERSION")
    )
}

struct Args {
    suite: String,
    noise: Vec<f64>,
    tiny: bool,
    generations: Option<usize>,
    islands: Option<usize>,
    pop: Option<usize>,
    seed: u32,
    metric: Metric,
    threshold: f64,
    time_budget: Option<f64>,
    json: bool,
    list: bool,
    front_parents: bool,
    threads: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            suite: "known-truth".to_string(),
            noise: vec![0.0, 0.01, 0.1],
            tiny: false,
            generations: None,
            islands: None,
            pop: None,
            seed: 0,
            metric: Metric::Squared,
            threshold: bench::DEFAULT_THRESHOLD,
            time_budget: None,
            json: false,
            list: false,
            front_parents: true, // #76
            threads: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        }
    }
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
    while i < args.len() {
        match args[i].as_str() {
            "--suite" => a.suite = take(&mut i, "--suite")?,
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
            "--generations" => {
                a.generations = Some(
                    take(&mut i, "--generations")?
                        .parse()
                        .map_err(|_| "--generations: not an integer".to_string())?,
                )
            }
            "--islands" => {
                a.islands = Some(
                    take(&mut i, "--islands")?
                        .parse()
                        .map_err(|_| "--islands: not an integer".to_string())?,
                )
            }
            "--pop" => {
                a.pop = Some(
                    take(&mut i, "--pop")?
                        .parse()
                        .map_err(|_| "--pop: not an integer".to_string())?,
                )
            }
            "--seed" => {
                a.seed = take(&mut i, "--seed")?
                    .parse()
                    .map_err(|_| "--seed: not an integer".to_string())?
            }
            "--metric" => {
                let v = take(&mut i, "--metric")?;
                a.metric = Metric::parse_opt(&v)
                    .ok_or_else(|| format!("--metric: unknown metric '{}'", v))?;
            }
            "--threshold" => {
                a.threshold = take(&mut i, "--threshold")?
                    .parse()
                    .map_err(|_| "--threshold: not a number".to_string())?
            }
            "--time-budget" => {
                a.time_budget = Some(
                    take(&mut i, "--time-budget")?
                        .parse()
                        .map_err(|_| "--time-budget: not a number".to_string())?,
                )
            }
            "--json" => a.json = true,
            "--no-front-parents" => a.front_parents = false,
            "--threads" => {
                a.threads = take(&mut i, "--threads")?
                    .parse()
                    .map_err(|_| "--threads: not an integer".to_string())?
            }
            "--list" => a.list = true,
            "-h" | "--help" => return Err(String::new()),
            other => return Err(format!("unknown option '{}'", other)),
        }
        i += 1;
    }
    if a.suite != "known-truth" {
        return Err(format!(
            "--suite: unknown suite '{}' (try known-truth)",
            a.suite
        ));
    }
    Ok(a)
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
        threads: a.threads,
    }
}

fn list_problems() -> String {
    let mut out = String::new();
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
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
        print!("{}", list_problems());
        std::process::exit(EXIT_OK);
    }

    let cfg = to_config(&parsed);
    let report = bench::run_suite(&cfg);
    if parsed.json {
        print!("{}", bench::report_json(&report, &cfg));
    } else {
        print!("{}", bench::report_text(&report, &cfg));
    }
    std::process::exit(EXIT_OK);
}
