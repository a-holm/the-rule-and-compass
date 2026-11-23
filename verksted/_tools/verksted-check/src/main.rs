//! `verksted-check`: link, contract, reference and index checks of the workshop.

mod check;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "usage:
  verksted-check links [--icm | --external] [DIR]
  verksted-check graph [--dot] [--out FILE] [DIR]
  verksted-check walk [--strict] [DIR]
  verksted-check refs [DIR]
  verksted-check index [--check] [DIR]
exit 0 clean, 1 problems (one per line on stdout), 2 usage";

fn usage() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}

fn finish(problems: &[String], what: &str) -> ExitCode {
    for p in problems {
        println!("{p}");
    }
    if problems.is_empty() {
        eprintln!("{what}: clean");
        ExitCode::SUCCESS
    } else {
        eprintln!("{what}: {} problems", problems.len());
        ExitCode::from(1)
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        return usage();
    };
    let mut flags: Vec<&str> = Vec::new();
    let mut dirs: Vec<&str> = Vec::new();
    let mut out_file: Option<&str> = None;
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        if a == "--out" && cmd == "graph" {
            match it.next() {
                Some(f) => out_file = Some(f),
                None => return usage(),
            }
        } else if a.starts_with("--") {
            flags.push(a);
        } else {
            dirs.push(a);
        }
    }
    let allowed: &[&str] = match cmd.as_str() {
        "walk" => &["--strict"],
        "index" => &["--check"],
        "links" => &["--icm", "--external"],
        "graph" => &["--dot"],
        "refs" => &[],
        _ => return usage(),
    };
    if dirs.len() > 1 || flags.iter().any(|f| !allowed.contains(f)) {
        return usage();
    }
    if cmd == "links" && flags.contains(&"--icm") && flags.contains(&"--external") {
        return usage();
    }
    let dir = PathBuf::from(dirs.first().copied().unwrap_or("verksted"));
    if !dir.is_dir() {
        eprintln!("{}: not a directory", dir.display());
        return ExitCode::from(2);
    }
    match cmd.as_str() {
        "links" if flags.contains(&"--icm") => {
            let graph = check::scan(&dir, false);
            print!("{}", check::render_icm_report(&graph));
            if graph.broken.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        "links" => {
            let external = flags.contains(&"--external");
            if external && dir.canonicalize().ok().and_then(|d| d.parent().and_then(check::crate_dir)).is_none() {
                eprintln!("links --external: no engine crate directory (package rule-and-compass) next to {}", dir.display());
                return ExitCode::from(1);
            }
            let report = check::check_tree_with(&dir, external);
            eprintln!("{} files, {} links and inputs checked", report.files, report.checked);
            finish(&report.broken, "links")
        }
        "graph" => {
            let graph = check::scan(&dir, true);
            let text = if flags.contains(&"--dot") {
                check::render_graph_dot(&graph)
            } else {
                let root_ref = check::graph_root_ref(&dir, out_file.map(Path::new));
                check::render_graph_jsonl(&graph, &root_ref)
            };
            match out_file {
                Some(f) => match fs::write(f, text) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("write {f}: {e}");
                        ExitCode::from(2)
                    }
                },
                None => {
                    print!("{text}");
                    ExitCode::SUCCESS
                }
            }
        }
        "walk" => finish(&check::walk_problems(&dir, flags.contains(&"--strict")), "walk"),
        "refs" => finish(&check::refs_problems(&dir), "refs"),
        _ => {
            if flags.contains(&"--check") {
                finish(&check::index_problems(&dir), "index")
            } else if let Err(e) = fs::write(dir.join("_index.md"), check::render_index(&dir)) {
                eprintln!("write {}/_index.md: {e}", dir.display());
                ExitCode::from(2)
            } else {
                ExitCode::SUCCESS
            }
        }
    }
}
