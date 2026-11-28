//! `rcsr-formal-vectors`: writes the vector files that the Lean checker `rcsr_model_check` replays.
//!
//! The vectors come from `rcsr::formal_vectors`, which calls the real library functions.

use std::path::Path;
use std::process::ExitCode;

use rcsr::formal_vectors::{golden, vectors, MODELS};

const USAGE: &str = "\
usage: rcsr-formal-vectors --golden <dir>
       rcsr-formal-vectors --bulk <dir> --count <n> [--seed <u64>]

Writes <dir>/<model>.vec for every model with a Lean counterpart under formal/.
--golden  the deterministic files committed under formal/vectors/golden/
--bulk    <n> vectors per model from --seed (default 1); <n> at least the directed set of each model
Exit: 0 written; 1 a file could not be written; 2 usage.";

fn write_all(dir: &str, text_of: impl Fn(&str) -> Option<String>) -> ExitCode {
    for model in MODELS {
        let Some(text) = text_of(model) else {
            eprintln!("rcsr-formal-vectors: --count is below the directed set of {model}");
            return ExitCode::from(2);
        };
        let path = Path::new(dir).join(format!("{model}.vec"));
        if let Err(e) = std::fs::write(&path, &text) {
            eprintln!("rcsr-formal-vectors: write {}: {e}", path.display());
            return ExitCode::from(1);
        }
        println!("{model}: {} vectors -> {}", text.lines().count() - 2, path.display());
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["--golden", dir] => write_all(dir, golden),
        ["--bulk", dir, "--count", n] | ["--bulk", dir, "--count", n, "--seed", _] => {
            let seed = match args.get(5) {
                None => Ok(1),
                Some(s) => s.parse::<u64>(),
            };
            match (n.parse::<usize>(), seed) {
                (Ok(n), Ok(seed)) => write_all(dir, |m| vectors(m, n, seed)),
                _ => {
                    eprintln!("{USAGE}");
                    ExitCode::from(2)
                }
            }
        }
        ["-h"] | ["--help"] => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
