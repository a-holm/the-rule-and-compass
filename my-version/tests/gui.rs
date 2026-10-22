//! Headless tests for the local web front end.
//!
//! These start the real server on port 0, speak HTTP/1.1 over a socket and read
//! the real NDJSON stream. No browser, display or screenshot tool is involved;
//! the visual layer is covered by checking the embedded assets and the events
//! they consume.

use rcsr::gui;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

fn get(addr: SocketAddr, path: &str) -> (u16, String) {
    request(addr, &format!("GET {} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n", path))
}

fn post(addr: SocketAddr, path: &str, body: &str) -> (u16, String) {
    request(
        addr,
        &format!(
            "POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path,
            body.len(),
            body
        ),
    )
}

fn request(addr: SocketAddr, raw: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .unwrap();
    stream.write_all(raw.as_bytes()).unwrap();
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes).unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let (head, body) = text
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no header terminator in {:?}", text));
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    (status, body.to_string())
}

fn example_table() -> String {
    let mut text = String::from("x,y\n");
    for i in 0..20 {
        let x = i as f64 - 9.5;
        let y = 2.0 * x + 1.0;
        text.push_str(&format!("{},{}\n", x, y));
    }
    text
}

#[test]
fn schema_endpoint_returns_the_cli_contract() {
    let server = gui::serve("127.0.0.1", 0).expect("start server");
    let (status, body) = get(server.addr(), "/api/schema");
    assert_eq!(status, 200);
    assert!(gui::is_valid_json(&body), "schema is not JSON: {}", body);
    assert!(body.contains("\"options\""));
    assert!(body.contains("\"operators\""));
    assert!(body.contains("\"exit_codes\""));
    // One source of truth: the server serves exactly what `rcsr schema` prints.
    assert_eq!(body, rcsr::schema::json());
}

#[test]
fn every_flag_has_a_type_default_and_description() {
    // The document is generated from one table, so the remaining risk is a
    // half-written entry, not drift between two copies.
    let doc = rcsr::schema::json();
    assert!(gui::is_valid_json(&doc), "schema is not valid JSON");
    let opts = doc
        .split("\"name\":\"")
        .skip(1)
        .collect::<Vec<_>>();
    assert!(
        opts.len() >= 50,
        "expected the full option table, found {}",
        opts.len()
    );
    for entry in opts {
        assert!(entry.starts_with("--"), "malformed option entry: {entry}");
        for field in ["\"type\":", "\"default\":", "\"description\":"] {
            assert!(entry.contains(field), "option missing {field}: {entry}");
        }
        // An empty description is the failure this guards against.
        assert!(
            !entry.contains("\"description\":\"\""),
            "option has an empty description: {entry}"
        );
    }
}


/// #38: the form is generated from `rcsr schema`, so every option it renders
/// must reach a deliberate outcome. A documented flag may be supported or
/// rejected, but never reported as "unknown argument".
#[test]
fn every_schema_option_is_supported_or_explicitly_rejected() {
    let doc = rcsr::schema::json();
    assert!(gui::is_valid_json(&doc), "schema is not valid JSON");
    // Only the root search options; subcommands carry their own lists.
    let root = doc.split("\"operators\"").next().expect("root options");
    let mut seen = 0;
    for entry in root.split("\"name\":\"--").skip(1) {
        let name = format!("--{}", entry.split('"').next().unwrap_or(""));
        // Find this entry's type field before the next option.
        let tail = entry.split("\"name\":\"--").next().unwrap_or(entry);
        let ty = tail
            .split("\"type\":\"")
            .nth(1)
            .and_then(|t| t.split('"').next())
            .unwrap_or("");
        let mut args = vec![name.clone()];
        if ty != "flag" {
            args.push("1".to_string());
        }
        match gui::parse_web_args_for_test(&args) {
            Ok(()) => {}
            Err(message) => assert!(
                !message.contains("unknown argument"),
                "the GUI renders {name} but rejects it as unknown: {message}"
            ),
        }
        seen += 1;
    }
    assert!(seen >= 50, "expected the full option table, found {seen}");
    for (name, reason) in gui::rejected_options() {
        assert!(
            doc.contains(&format!("\"{}\"", name)),
            "unsupported option {name} is not in the schema"
        );
        assert!(!reason.is_empty(), "{name} has an empty reason");
    }
}

#[test]
fn run_endpoint_streams_ndjson_ending_in_done() {
    let server = gui::serve("127.0.0.1", 0).expect("start server");
    let body = format!(
        "{{\"args\":[\"--data\",\"-\",\"--target\",\"y\",\"--vars\",\"x\",\"--generations\",\"6\",\"--pop\",\"40\",\"--islands\",\"1\",\"--max-complexity\",\"8\",\"--patience\",\"3\",\"--seed\",\"1\"],\"table\":{}}}",
        json_string(&example_table())
    );
    let (status, stream) = post(server.addr(), "/api/run", &body);
    assert_eq!(status, 200, "{}", stream);
    let lines: Vec<&str> = stream.lines().filter(|line| !line.trim().is_empty()).collect();
    assert!(lines.len() > 2, "expected a stream, got {:?}", lines);
    for line in &lines {
        assert!(
            gui::is_valid_json(line),
            "stream line is not JSON: {}",
            line
        );
        assert!(line.contains("\"event\""));
    }
    let last = lines.last().unwrap();
    assert!(
        last.contains("\"event\":\"done\""),
        "stream did not end with done: {}",
        last
    );
    assert!(lines.iter().any(|line| line.contains("\"event\":\"start\"")));
    assert!(lines
        .iter()
        .any(|line| line.contains("\"event\":\"solution\"")));
}

#[test]
fn malformed_request_returns_400_without_hanging() {
    let server = gui::serve("127.0.0.1", 0).expect("start server");
    let (status, _) = request(server.addr(), "NONSENSE\r\n\r\n");
    assert_eq!(status, 400);
    let (status, _) = post(server.addr(), "/api/run", "{\"args\":");
    assert_eq!(status, 400);
    let (status, _) = post(server.addr(), "/api/run", "{\"args\":[\"--data\"]}");
    assert_eq!(status, 400);
}

#[test]
fn root_serves_the_single_page_app() {
    let server = gui::serve("127.0.0.1", 0).expect("start server");
    let (status, html) = get(server.addr(), "/");
    assert_eq!(status, 200);
    assert!(html.contains("rule-and-compass"));
    assert!(html.contains("app.js"));
    let (status, js) = get(server.addr(), "/app.js");
    assert_eq!(status, 200);
    assert!(js.contains("/api/schema"));
    assert!(js.contains("state.schema.options"));
}

#[test]
fn served_assets_reference_no_absolute_external_url() {
    for (name, asset) in [
        ("index.html", include_str!("../gui/index.html")),
        ("app.js", include_str!("../gui/app.js")),
        ("style.css", include_str!("../gui/style.css")),
    ] {
        assert!(
            !asset.contains("http://") && !asset.contains("https://"),
            "{} contains an absolute external URL",
            name
        );
    }
}

#[test]
fn server_stops_when_dropped() {
    let server = gui::serve("127.0.0.1", 0).expect("start server");
    let addr = server.addr();
    drop(server);
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match TcpStream::connect_timeout(&addr, Duration::from_millis(100)) {
            Err(_) => break,
            Ok(_) if Instant::now() >= deadline => panic!("server still accepting after drop"),
            Ok(_) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
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
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// #223: the web front end parses the same flags, so it must refuse the same
/// contradiction. Silently fitting on the CPU would give the browser run a
/// different objective from the one the request named. #229 makes
/// `log_squashed` a device objective, #230 makes the rank family one (the
/// exact polish), and #222 makes `absolute` one, so all are accepted with
/// `gpu`.
#[test]
fn the_web_front_end_refuses_explicit_gpu_outside_the_native_objectives() {
    let args = |metric: &str| {
        vec![
            "--fit-strategy".to_string(),
            "gpu".to_string(),
            "--metric".to_string(),
            metric.to_string(),
        ]
    };
    for metric in [
        "squared",
        "aic",
        "aicc",
        "bic",
        "log_squashed",
        "absolute",
        "rank_correlation",
        "rank_correlation_signed",
        "median",
        // #234: log_loss has a native device fit objective (fit-only).
        "log_loss",
    ] {
        assert!(
            gui::parse_web_args_for_test(&args(metric)).is_ok(),
            "--metric {metric} has a device fit objective and must be accepted"
        );
    }
    for metric in ["interquartile", "hinge"] {
        let err = gui::parse_web_args_for_test(&args(metric))
            .expect_err(&format!("--metric {metric} must be refused"));
        assert!(
            err.contains("minimises squared error, log-squashed loss, mean"),
            "--metric {metric}: {err}"
        );
        assert!(err.contains("--metric squared"), "--metric {metric}: {err}");
    }
    assert!(gui::parse_web_args_for_test(&[
        "--fit-strategy".to_string(),
        "exact".to_string(),
        "--metric".to_string(),
        "absolute".to_string(),
    ])
    .is_ok());
}
