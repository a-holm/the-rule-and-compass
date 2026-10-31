//! `rcsr-rss`: sample the RSS and thread count of a child process (#123).
//!
//! Runs the command after `--`, polls `/proc/<pid>/status` every interval and
//! writes one CSV row per sample:
//!
//! ```text
//! elapsed_s,vmrss_kib,vmhwm_kib,threads
//! ```
//!
//! The thread count identifies the pipeline phases without instrumenting the
//! binary. A JSON summary goes to stdout. The exit status is the child's.
//! Linux only.
//!
//! ```text
//! rcsr-rss --csv /tmp/prep-rss.csv --interval 0.025 -- \
//!   target/release/rcsr prep --input raw_15m ... --out panel.rcsb
//! ```

use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::time::{Duration, Instant};

const USAGE: &str = "rcsr-rss: sample RSS and threads of a child process

USAGE:
  rcsr-rss [--csv FILE] [--interval SECONDS] -- COMMAND [ARGS...]

OPTIONS:
  --csv FILE            write the per-sample CSV here
  --interval SECONDS    sampling interval (default: 0.025)
  -h, --help            this text
";

/// (VmRSS KiB, VmHWM KiB, Threads), or None when the process is gone.
fn read_status(pid: u32) -> Option<(u64, u64, u64)> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let (mut rss, mut hwm, mut threads) = (None, None, None);
    for line in text.lines() {
        let field = |key: &str| -> Option<u64> {
            line.strip_prefix(key)?.split_whitespace().next()?.parse().ok()
        };
        if let Some(v) = field("VmRSS:") {
            rss = Some(v);
        } else if let Some(v) = field("VmHWM:") {
            hwm = Some(v);
        } else if let Some(v) = field("Threads:") {
            threads = Some(v);
        }
    }
    Some((rss?, hwm?, threads?))
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn run(argv: &[String]) -> Result<i32, String> {
    let mut csv: Option<String> = None;
    let mut interval = 0.025f64;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(0);
            }
            "--csv" => {
                csv = Some(argv.get(i + 1).ok_or("--csv needs a value")?.clone());
                i += 2;
            }
            "--interval" => {
                let v = argv.get(i + 1).ok_or("--interval needs a value")?;
                interval = v.parse().map_err(|_| format!("invalid --interval: {v:?}"))?;
                if !(interval.is_finite() && interval >= 0.0) {
                    return Err(format!("invalid --interval: {v:?}"));
                }
                i += 2;
            }
            "--" => {
                i += 1;
                break;
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }
    let cmd = &argv[i.min(argv.len())..];
    if cmd.is_empty() {
        return Err("nothing to run; put the command after --".to_string());
    }

    let started = Instant::now();
    let mut child = Command::new(&cmd[0])
        .args(&cmd[1..])
        .spawn()
        .map_err(|e| format!("start {}: {e}", cmd[0]))?;
    let mut csv_file = match &csv {
        Some(p) => {
            let mut f = File::create(p).map_err(|e| format!("create {p}: {e}"))?;
            writeln!(f, "elapsed_s,vmrss_kib,vmhwm_kib,threads").map_err(|e| e.to_string())?;
            Some(f)
        }
        None => None,
    };
    let (mut samples, mut peak_rss, mut peak_hwm, mut peak_threads) = (0u64, 0u64, 0u64, 0u64);
    let status = loop {
        if let Some((rss, hwm, threads)) = read_status(child.id()) {
            let elapsed = started.elapsed().as_secs_f64();
            samples += 1;
            peak_rss = peak_rss.max(rss);
            peak_hwm = peak_hwm.max(hwm);
            peak_threads = peak_threads.max(threads);
            if let Some(f) = csv_file.as_mut() {
                writeln!(f, "{elapsed:.3},{rss},{hwm},{threads}").map_err(|e| e.to_string())?;
                f.flush().map_err(|e| e.to_string())?;
            }
        }
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        std::thread::sleep(Duration::from_secs_f64(interval));
    };
    let code = {
        use std::os::unix::process::ExitStatusExt;
        status.code().unwrap_or_else(|| -status.signal().unwrap_or(0))
    };
    let wall = started.elapsed().as_secs_f64();
    let command = cmd.iter().map(|s| json_string(s)).collect::<Vec<_>>().join(", ");
    let csv_json = csv.as_deref().map(json_string).unwrap_or_else(|| "null".to_string());
    println!(
        "{{\"command\": [{command}], \"csv\": {csv_json}, \"exit_code\": {code}, \
         \"interval_s\": {interval:?}, \"peak_rss_bytes\": {}, \"peak_rss_kib_sampled\": {peak_rss}, \
         \"peak_threads\": {peak_threads}, \"peak_vmhwm_kib\": {peak_hwm}, \"samples\": {samples}, \
         \"wall_s\": {:?}}}",
        peak_hwm * 1024,
        round3(wall)
    );
    Ok(code)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => std::process::exit(code & 0xff),
        Err(e) => {
            eprintln!("rcsr-rss: error: {e}");
            std::process::exit(2);
        }
    }
}
