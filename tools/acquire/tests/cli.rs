use std::process::Command;

fn run(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rcsr-acquire")).args(args).output().expect("run binary");
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

#[test]
fn help_lists_the_subcommands() {
    let (code, out, _) = run(&["--help"]);
    assert_eq!(code, 0);
    for needle in ["rcsr-acquire list", "rcsr-acquire fetch", "rcsr-acquire verify", "--skip-404", "--through"] {
        assert!(out.contains(needle), "missing {needle}");
    }
}

#[test]
fn usage_errors_exit_2_without_network() {
    for args in [
        &[][..],
        &["bogus"],
        &["fetch", "--interval", "15m", "--from", "2026-08"],
        &["fetch", "--interval", "15m", "--from", "2026-13", "--symbols", "BTCUSDT"],
        &["fetch", "--interval", "15m", "--from", "2026-08", "--symbols", "BTC/USDT"],
        &["fetch", "--interval", "15m", "--from", "2026-08", "--symbols", "BTCUSDT", "--jobs", "0"],
        &["fetch", "--interval", "15m", "--from", "2026-08", "--to", "2999-01", "--symbols", "BTCUSDT"],
        &["fetch", "--interval", "15m", "--from", "2026-09", "--to", "2026-08", "--symbols", "BTCUSDT"],
        &["verify"],
        &["verify", "--interval", "15m", "--through", "soon"],
    ] {
        let (code, _, err) = run(args);
        assert_eq!(code, 2, "{args:?}: {err}");
        assert!(err.contains("rcsr-acquire:") || err.contains("USAGE"), "{args:?}");
    }
}
