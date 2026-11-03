//! No tracked file names a banned word (#684). The word list lives outside git;
//! without it the test skips.

use std::process::Command;

#[test]
fn no_banned_word_in_tracked_files() {
    let output = Command::new("bash")
        .arg("../verksted/_tools/leak-scan.sh")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run leak-scan.sh");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    match output.status.code() {
        Some(0) => {}
        Some(3) => eprintln!("{}", text.trim_end()),
        _ => panic!("leak-scan.sh failed:\n{text}"),
    }
}
