//! `rcsr` — command-line symbolic-regression engine.

use rcsr::expr::{self, Op};
use rcsr::metrics::Metric;
use rcsr::rng::Mt;
use rcsr::{data, search};

fn usage() -> &'static str {
    "rcsr — symbolic regression (rule-and-compass)\n\n\
USAGE:\n  \
rcsr --data <file> --target <col> [options]\n\n\
OPTIONS:\n  \
--data <file>            CSV/TSV/space-delimited data (header row required)\n  \
--target <col>           name of the target column\n  \
--vars <a,b,..>          input columns (default: all except target)\n  \
--metric absolute|squared|correlation   (default absolute)\n  \
--generations <N>        (default 200)\n  \
--islands <I>            (default 1)\n  \
--pop <P>                population per island (default 200)\n  \
--max-complexity <K>     max nodes per expression (default 30)\n  \
--seed <S>               RNG seed (default 0)\n  \
--test-frac <f>          held-out fraction (default 0.25)\n  \
--ops <list>             e.g. add,sub,mul,div,sin,exp\n  \
--out <file>             write the Pareto front as TSV\n  \
-h, --help               show this help"
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = search::Config::default();
    let mut data_path: Option<String> = None;
    let mut target: Option<String> = None;
    let mut vars: Vec<String> = Vec::new();
    let mut test_frac = 0.25;
    let mut out_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", usage());
                return;
            }
            "--data" => {
                i += 1;
                data_path = args.get(i).cloned();
            }
            "--target" => {
                i += 1;
                target = args.get(i).cloned();
            }
            "--vars" => {
                i += 1;
                vars = args
                    .get(i)
                    .map(|s| {
                        s.split(',')
                            .map(|x| x.trim().to_string())
                            .filter(|x| !x.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();
            }
            "--metric" => {
                i += 1;
                if let Some(m) = args.get(i) {
                    cfg.metric = Metric::parse(m);
                }
            }
            "--generations" => {
                i += 1;
                cfg.generations = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.generations);
            }
            "--islands" => {
                i += 1;
                cfg.islands = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.islands);
            }
            "--pop" => {
                i += 1;
                cfg.pop = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.pop);
            }
            "--max-complexity" => {
                i += 1;
                cfg.max_complexity = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(cfg.max_complexity);
            }
            "--seed" => {
                i += 1;
                cfg.seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(cfg.seed);
            }
            "--test-frac" => {
                i += 1;
                test_frac = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(test_frac);
            }
            "--ops" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    cfg.ops = Op::parse_list(s);
                }
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }

    let data_path = match data_path {
        Some(p) => p,
        None => {
            eprintln!("error: --data is required\n\n{}", usage());
            std::process::exit(2);
        }
    };
    let target = match target {
        Some(t) => t,
        None => {
            eprintln!("error: --target is required\n\n{}", usage());
            std::process::exit(2);
        }
    };

    let table = match data::read_table(&data_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    if vars.is_empty() {
        vars = table
            .names
            .iter()
            .filter(|n| **n != target)
            .cloned()
            .collect();
    }
    if vars.is_empty() {
        eprintln!("error: no input variables");
        std::process::exit(1);
    }

    let mut rng = Mt::new(cfg.seed);
    let (train, test, means, stds) = match data::prepare(&table, &vars, &target, test_frac, &mut rng) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "# rule-and-compass | {} island(s) x {} pop | {} generations | metric={}",
        cfg.islands,
        cfg.pop,
        cfg.generations,
        cfg.metric.name()
    );
    println!(
        "# target: {} = f({}) | train rows: {} | test rows: {}",
        target,
        vars.join(","),
        train.len(),
        test.len()
    );

    let t0 = std::time::Instant::now();
    let outcome = search::run(&cfg, &train, &test);
    println!(
        "# elapsed: {:.2}s | Pareto members: {}",
        t0.elapsed().as_secs_f64(),
        outcome.archive.items.len()
    );

    let sorted = outcome.archive.sorted();
    for (rank, (s, &te)) in sorted.iter().zip(outcome.test_errors.iter()).enumerate() {
        let den = expr::simplify(&expr::denormalize(&expr::simplify(&s.tree), &means, &stds));
        println!(
            "{:>3}  complexity={:<3} train_err={:<13.6} test_err={:<13.6}  {} = {}",
            rank + 1,
            s.complexity,
            s.error,
            te,
            target,
            expr::to_string(&den, &vars)
        );
    }

    if let Some(path) = out_path {
        let mut buf = String::from("rank\tcomplexity\ttrain_error\ttest_error\texpression\n");
        for (rank, (s, &te)) in sorted.iter().zip(outcome.test_errors.iter()).enumerate() {
            let den = expr::simplify(&expr::denormalize(&expr::simplify(&s.tree), &means, &stds));
            buf.push_str(&format!(
                "{}\t{}\t{:.8}\t{:.8}\t{}\n",
                rank + 1,
                s.complexity,
                s.error,
                te,
                expr::to_string(&den, &vars)
            ));
        }
        if let Err(e) = std::fs::write(&path, buf) {
            eprintln!("error writing {}: {}", path, e);
        }
    }
}
