//! Writes the h1-mini panel CSV to stdout: `panel <seed>`. Build: `rustc -O scripts/h1-mini/panel.rs -o <out>`.

#[path = "../../rcsr-engine/tests/common/h1_mini.rs"]
mod h1_mini;

use std::io::Write;

fn main() {
    let seed: u64 = match std::env::args().nth(1).and_then(|s| s.parse().ok()) {
        Some(s) => s,
        None => {
            eprintln!("usage: panel <seed>");
            std::process::exit(2);
        }
    };
    std::io::stdout().write_all(&h1_mini::h1_mini_panel_csv(seed)).expect("write the panel");
}
