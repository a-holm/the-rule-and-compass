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
    for (name, _, _, _) in gui::SCHEMA_OPTIONS {
        assert!(
            body.contains(&format!("\"name\":\"{}\"", name)),
            "schema omits {}",
            name
        );
    }
}

#[test]
fn schema_options_match_the_cli_schema_function() {
    // `schema()` is private to the `rcsr` binary, so keep the library copy
    // honest by reading the table in `src/main.rs` and comparing every field.
    let main = include_str!("../src/main.rs");
    let start = main
        .find("let opts = [")
        .expect("schema opts table not found");
    let after = &main[start..];
    let end = after.find("\n    ];").expect("end of schema opts table");
    let block = &after[..end];
    let mut from_cli: Vec<(String, String, String, String)> = Vec::new();
    for line in block.lines() {
        let line = line.trim();
        if !line.starts_with("(\"") {
            continue;
        }
        let strings = rust_strings(line);
        assert!(
            strings.len() >= 4,
            "cannot parse schema option line: {}",
            line
        );
        from_cli.push((
            strings[0].clone(),
            strings[1].clone(),
            strings[2].clone(),
            strings[3].clone(),
        ));
    }
    assert!(!from_cli.is_empty());
    assert_eq!(from_cli.len(), gui::SCHEMA_OPTIONS.len());
    for (cli, ours) in from_cli.iter().zip(gui::SCHEMA_OPTIONS.iter()) {
        assert_eq!(
            (cli.0.as_str(), cli.1.as_str(), cli.2.as_str(), cli.3.as_str()),
            *ours,
            "GUI schema drifted from src/main.rs"
        );
    }
}

/// Extract the Rust string literals on one schema line, honouring backslash
/// escapes so a description may contain quotes and commas.
fn rust_strings(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            i += 1;
            continue;
        }
        i += 1;
        let mut s = String::new();
        while i < bytes.len() && bytes[i] != b'"' {
            if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 1;
                match bytes[i] {
                    b'n' => s.push('\n'),
                    b'r' => s.push('\r'),
                    b't' => s.push('\t'),
                    b'\\' => s.push('\\'),
                    b'"' => s.push('"'),
                    other => s.push(other as char),
                }
            } else {
                s.push(bytes[i] as char);
            }
            i += 1;
        }
        i += 1;
        out.push(s);
    }
    out
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
