//! `verksted-check`: link, contract, reference and index checks of the workshop.

mod check;

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  verksted-check links [DIR]
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
    for a in rest {
        if a.starts_with("--") {
            flags.push(a);
        } else {
            dirs.push(a);
        }
    }
    let allowed: &[&str] = match cmd.as_str() {
        "walk" => &["--strict"],
        "index" => &["--check"],
        "links" | "refs" => &[],
        _ => return usage(),
    };
    if dirs.len() > 1 || flags.iter().any(|f| !allowed.contains(f)) {
        return usage();
    }
    let dir = PathBuf::from(dirs.first().copied().unwrap_or("verksted"));
    if !dir.is_dir() {
        eprintln!("{}: not a directory", dir.display());
        return ExitCode::from(2);
    }
    match cmd.as_str() {
        "links" => {
            let report = check::check_tree(&dir);
            eprintln!("{} files, {} links and inputs checked", report.files, report.checked);
            finish(&report.broken, "links")
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
