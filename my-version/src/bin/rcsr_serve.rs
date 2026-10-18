//! `rcsr_serve` — a local web front end for the `rcsr` engine.
//!
//! The server binds a loopback address, serves the embedded single-page app and
//! exposes the engine over JSON/NDJSON. It has no network dependency at runtime
//! and uses only the standard library.
//!
//! ```text
//! rcsr_serve --port 8080
//! rcsr_serve --host 127.0.0.1 --port 0 --open
//! ```

use rcsr::gui;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() -> String {
    format!(
        "rcsr_serve {VERSION} — local web front end for rule-and-compass

USAGE:
  rcsr_serve [--host <addr>] [--port <N>] [--open]

OPTIONS:
  --host <addr>   address to bind (default 127.0.0.1; loopback only)
  --port <N>      TCP port (default 8080; 0 asks the OS for a free port)
  --open          print the URL to open in a browser; never launches one
  -h, --help      show this help
  -V, --version   print the version

ENDPOINTS:
  GET  /             the single-page application
  GET  /api/schema   the same document as `rcsr schema`
  GET  /api/example  a small bundled CSV
  POST /api/run      {{\"args\":[\"--data\",\"-\",...],\"table\":\"...\"}} streamed as NDJSON
"
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut host = "127.0.0.1".to_string();
    let mut port = 8080u16;
    let mut open = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                return;
            }
            "-V" | "--version" => {
                println!("rcsr_serve {}", VERSION);
                return;
            }
            "--host" => {
                i += 1;
                match args.get(i) {
                    Some(v) => host = v.clone(),
                    None => {
                        eprintln!("error: --host requires a value");
                        std::process::exit(2);
                    }
                }
            }
            "--port" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<u16>().ok()) {
                    Some(v) => port = v,
                    None => {
                        eprintln!("error: --port needs an integer from 0 to 65535");
                        std::process::exit(2);
                    }
                }
            }
            "--open" => open = true,
            other => {
                eprintln!("error: unknown argument '{}'\n\n{}", other, usage());
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let server = match gui::serve(&host, port) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(3);
        }
    };
    println!("rcsr_serve listening on {}", server.url());
    println!("data stays local; the server has no network dependency beyond this socket");
    if open {
        println!("open {} in a browser", server.url());
    }
    println!("press Ctrl+C to stop");
    loop {
        std::thread::park();
    }
}
