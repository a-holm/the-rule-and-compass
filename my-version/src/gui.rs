//! Local web front end for `rcsr`.
//!
//! The server is a deliberately small HTTP/1.1 implementation over
//! [`std::net::TcpListener`]: one request per connection, `Connection: close`,
//! no crates, no async and no network access beyond the loopback interface.
//! The single-page app is embedded with [`include_str!`], so the binary is
//! self-contained and works offline.
//!
//! Endpoints:
//!
//! * `GET /` — the embedded single-page application;
//! * `GET /api/schema` — the same JSON document as `rcsr schema`;
//! * `GET /api/example` — a small bundled table for the one-click path;
//! * `POST /api/run` — `{"args":["--data","-",…],"table":"…"}`. The search runs
//!   in-process through the library pipeline (`parse_table` →
//!   `prepare_full_weighted_with` → `search::run`) and the response is
//!   `application/x-ndjson`, one event per line, flushed as it is produced.
//!
//! The stream uses the same event names and payloads as `rcsr --ndjson`, plus a
//! `bic` field on `solution` (the CLI's JSON object carries it, the CLI's
//! NDJSON event does not). The front end never parses human-readable output.

use crate::data::{self, Center, DType, DateOnlyPolicy, FillMethod, Scale, SplitMode, TimeUnit};
use crate::diagnostics;
use crate::expr::{self, Op};
use crate::fit;
use crate::metrics::{self, Metric};
use crate::rng::Mt;
use crate::search;
use crate::stability;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const MAX_HEADER: usize = 64 * 1024;
const MAX_BODY: usize = 32 * 1024 * 1024;
const HTML: &str = include_str!("../gui/index.html");
const APP_JS: &str = include_str!("../gui/app.js");
const STYLE_CSS: &str = include_str!("../gui/style.css");
const EXAMPLE_CSV: &str = "x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n10,21\n11,23\n12,25\n13,27\n14,29\n15,31\n16,33\n17,35\n18,37\n19,39\n";

/// `rcsr schema` as a JSON document. The option table lives in
/// [`crate::schema::OPTIONS`]; this endpoint serves exactly that document, so
/// there is no second copy to drift.
pub fn schema_json() -> String {
    // Single source of truth: the same document `rcsr schema` prints.
    crate::schema::json()
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// A running local server. Dropping it stops the accept loop and joins it.
pub struct Server {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Server {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Ask the accept loop to stop. The `Drop` implementation calls this too.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.shutdown();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Bind `host:port` and serve until the returned [`Server`] is dropped.
///
/// A port of `0` asks the operating system for a free port; read it back with
/// [`Server::addr`]. The accept loop is non-blocking so shutdown never waits on
/// a connection; each connection is handled on its own thread.
pub fn serve(host: &str, port: u16) -> Result<Server, String> {
    let listener = TcpListener::bind((host, port))
        .map_err(|e| format!("cannot bind {}:{}: {}", host, port, e))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("cannot configure listener: {}", e))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("cannot read local address: {}", e))?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    let handle = thread::spawn(move || {
        while !flag.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((stream, _)) => {
                    thread::spawn(move || handle_connection(stream));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });
    Ok(Server {
        addr,
        shutdown,
        handle: Some(handle),
    })
}

fn handle_connection(mut stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let request = match read_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            let _ = send_json(&mut stream, 400, &format!("{{\"error\":{}}}", jstr(&e)));
            return;
        }
    };
    let path = request.path.split('?').next().unwrap_or(&request.path);
    let result = match (request.method.as_str(), path) {
        ("GET", "/") | ("GET", "/index.html") => {
            send_response(&mut stream, 200, "text/html; charset=utf-8", HTML.as_bytes())
        }
        ("GET", "/app.js") => send_response(
            &mut stream,
            200,
            "text/javascript; charset=utf-8",
            APP_JS.as_bytes(),
        ),
        ("GET", "/style.css") => send_response(
            &mut stream,
            200,
            "text/css; charset=utf-8",
            STYLE_CSS.as_bytes(),
        ),
        ("GET", "/api/schema") => {
            let body = schema_json();
            send_response(
                &mut stream,
                200,
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        ("GET", "/api/example") => send_response(
            &mut stream,
            200,
            "text/csv; charset=utf-8",
            EXAMPLE_CSV.as_bytes(),
        ),
        ("POST", "/api/run") => handle_run(request, &mut stream),
        ("GET", "/favicon.ico") => send_response(&mut stream, 204, "text/plain", b""),
        (method, _) if method != "GET" && method != "POST" => {
            send_json(&mut stream, 405, "{\"error\":\"method not allowed\"}")
        }
        _ => send_json(&mut stream, 404, "{\"error\":\"not found\"}"),
    };
    let _ = result;
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
        if buf.len() > MAX_HEADER {
            return Err("request headers too large".to_string());
        }
        let n = stream
            .read(&mut chunk)
            .map_err(|e| format!("cannot read request: {}", e))?;
        if n == 0 {
            return Err("connection closed before the request was complete".to_string());
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let header = std::str::from_utf8(&buf[..header_end])
        .map_err(|_| "request headers are not valid UTF-8".to_string())?;
    let rest = &buf[header_end..];
    let mut lines = header.split("\r\n");
    let request_line = lines.next().ok_or_else(|| "empty request".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "missing request method".to_string())?
        .to_string();
    let path = parts
        .next()
        .ok_or_else(|| "missing request path".to_string())?
        .to_string();
    let version = parts
        .next()
        .ok_or_else(|| "missing HTTP version".to_string())?;
    if !version.starts_with("HTTP/1.") {
        return Err(format!("unsupported HTTP version '{}'", version));
    }
    if !matches!(method.as_str(), "GET" | "POST" | "HEAD") {
        return Err(format!("unsupported method '{}'", method));
    }

    let mut content_length = 0usize;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| "malformed header line".to_string())?;
        if name.trim().eq_ignore_ascii_case("content-length") {
            content_length = value
                .trim()
                .parse::<usize>()
                .map_err(|_| "invalid Content-Length".to_string())?;
            if content_length > MAX_BODY {
                return Err("request body too large".to_string());
            }
        }
    }

    let mut body = rest.to_vec();
    while body.len() < content_length {
        let n = stream
            .read(&mut chunk)
            .map_err(|e| format!("cannot read request body: {}", e))?;
        if n == 0 {
            return Err("request body ended early".to_string());
        }
        body.extend_from_slice(&chunk[..n]);
        if body.len() > MAX_BODY {
            return Err("request body too large".to_string());
        }
    }
    body.truncate(content_length);
    Ok(Request { method, path, body })
}

/// Return the byte offset *after* the blank line that ends the headers.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .or_else(|| {
            buf.windows(2)
                .position(|w| w == b"\n\n")
                .map(|i| i + 2)
        })
}

fn send_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status,
        reason(status),
        content_type,
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn send_json(stream: &mut TcpStream, status: u16, body: &str) -> std::io::Result<()> {
    send_response(stream, status, "application/json; charset=utf-8", body.as_bytes())
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

/// The subset of JSON the endpoint accepts: one object with an `args` string
/// array and an optional `table` string.
enum Json {
    Null,
    Other,
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

/// True when `text` is one complete JSON value.
///
/// The endpoint itself only needs the parser below; the headless test uses
/// this to check that every streamed line is valid JSON without adding a
/// dependency.
pub fn is_valid_json(text: &str) -> bool {
    parse_json(text).is_ok()
}

fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = JsonParser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let value = p.value()?;
    p.ws();
    if p.pos != p.bytes.len() {
        return Err(format!("trailing text at byte {}", p.pos));
    }
    Ok(value)
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl JsonParser<'_> {
    fn ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn eat(&mut self, b: u8) -> Result<(), String> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", b as char, self.pos))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => {
                self.literal("true")?;
                Ok(Json::Other)
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Json::Other)
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Json::Null)
            }
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            Some(c) => Err(format!("unexpected '{}' at byte {}", c as char, self.pos)),
            None => Err("unexpected end of JSON".to_string()),
        }
    }

    fn literal(&mut self, lit: &str) -> Result<(), String> {
        if self.bytes[self.pos..].starts_with(lit.as_bytes()) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("expected {} at byte {}", lit, self.pos))
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(items));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.eat(b':')?;
            let value = self.value()?;
            items.push((key, value));
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
        Ok(Json::Object(items))
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
            }
        }
        Ok(Json::Array(items))
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = self
                .peek()
                .ok_or_else(|| "unterminated string".to_string())?;
            match c {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let esc = self
                        .peek()
                        .ok_or_else(|| "unterminated escape".to_string())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.hex4()?;
                            let ch = char::from_u32(code)
                                .ok_or_else(|| "invalid unicode escape".to_string())?;
                            out.push(ch);
                        }
                        other => {
                            return Err(format!(
                                "invalid escape '\\{}' at byte {}",
                                other as char, self.pos
                            ))
                        }
                    }
                }
                _ => {
                    let start = self.pos;
                    while let Some(b) = self.peek() {
                        if b == b'"' || b == b'\\' {
                            break;
                        }
                        self.pos += 1;
                    }
                    let piece = std::str::from_utf8(&self.bytes[start..self.pos])
                        .map_err(|_| "string is not valid UTF-8".to_string())?;
                    out.push_str(piece);
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.bytes.len() {
            return Err("truncated unicode escape".to_string());
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| "invalid unicode escape".to_string())?;
        self.pos += 4;
        u32::from_str_radix(s, 16).map_err(|_| "invalid unicode escape".to_string())
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("");
        s.parse::<f64>()
            .map(|_| Json::Other)
            .map_err(|_| format!("invalid number '{}'", s))
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

// ---------------------------------------------------------------------------
// /api/run
// ---------------------------------------------------------------------------

struct RunRequest {
    args: Vec<String>,
    table: Option<String>,
}

fn parse_run_request(body: &[u8]) -> Result<RunRequest, String> {
    let text = std::str::from_utf8(body).map_err(|_| "body is not valid UTF-8".to_string())?;
    let parsed = parse_json(text)?;
    let items = match parsed {
        Json::Object(items) => items,
        _ => return Err("body must be a JSON object".to_string()),
    };
    let mut args = None;
    let mut table = None;
    for (key, value) in items {
        match key.as_str() {
            "args" => match value {
                Json::Array(values) => {
                    let mut out = Vec::new();
                    for v in values {
                        match v {
                            Json::String(s) => out.push(s),
                            _ => return Err("\"args\" must be an array of strings".to_string()),
                        }
                    }
                    args = Some(out);
                }
                _ => return Err("\"args\" must be an array of strings".to_string()),
            },
            "table" => match value {
                Json::String(s) => table = Some(s),
                Json::Null => table = None,
                _ => return Err("\"table\" must be a string or null".to_string()),
            },
            _ => {}
        }
    }
    Ok(RunRequest {
        args: args.ok_or_else(|| "missing \"args\" array".to_string())?,
        table,
    })
}

fn handle_run(request: Request, stream: &mut TcpStream) -> std::io::Result<()> {
    let parsed = match parse_run_request(&request.body) {
        Ok(r) => r,
        Err(e) => return send_json(stream, 400, &format!("{{\"error\":{}}}", jstr(&e))),
    };
    let plan = match RunPlan::new(parsed) {
        Ok(p) => p,
        Err(e) => return send_json(stream, 400, &format!("{{\"error\":{}}}", jstr(&e))),
    };

    let head = "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    stream.flush()?;
    let writer = match stream.try_clone() {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            return send_json(
                stream,
                500,
                &format!("{{\"error\":{}}}", jstr(&format!("cannot stream: {}", e))),
            )
        }
    };
    if let Err(e) = plan.execute(&writer) {
        emit(
            &writer,
            &format!(
                "{{\"event\":\"error\",\"code\":3,\"message\":{}}}",
                jstr(&e)
            ),
        );
    }
    Ok(())
}

fn emit(writer: &Arc<Mutex<TcpStream>>, line: &str) {
    if let Ok(mut w) = writer.lock() {
        let _ = w.write_all(line.as_bytes());
        let _ = w.write_all(b"\n");
        let _ = w.flush();
    }
}

struct RunPlan {
    cfg: search::Config,
    prepared: data::Prepared,
    table: data::Table,
    dtypes: Vec<DType>,
    vars: Vec<String>,
    variable_types: Vec<data::VariableDType>,
    target: String,
    notes: Vec<String>,
    weight: Option<String>,
    null_runs: usize,
    bootstrap_resamples: usize,
    cv_mode: Option<data::CvMode>,
    cv_k: usize,
    stability_runs: usize,
    top: usize,
    eval_expr: Option<String>,
}

struct RawOptions {
    data: Option<String>,
    target: Option<String>,
    vars: Vec<String>,
    dtype: Option<String>,
    max_levels: usize,
    test_frac: f64,
    weight: Option<String>,
    fill: FillMethod,
    outlier_iqr: Option<f64>,
    smooth: usize,
    smooth_centred: bool,
    allow_interpolate: bool,
    allow_lockbox: bool,
    validation_frac: f64,
    date_only: DateOnlyPolicy,
    time_unit: Option<TimeUnit>,
    tz: Option<i64>,
    skip_malformed: bool,
    center: Center,
    scale: Scale,
    filter: Option<String>,
    exclude: Option<String>,
    train_select: Option<String>,
    test_select: Option<String>,
    split: Option<String>,
    time: Option<String>,
    group_by: Option<String>,
    gap: usize,
    folds: usize,
    lags: usize,
    history_frac: f64,
    gap_break: f64,
    null_runs: usize,
    bootstrap_resamples: usize,
    cv: Option<String>,
    cv_k: usize,
    stability_runs: usize,
    top: usize,
    eval_expr: Option<String>,
}

impl Default for RawOptions {
    fn default() -> Self {
        RawOptions {
            data: None,
            target: None,
            vars: Vec::new(),
            dtype: None,
            max_levels: 16,
            test_frac: 0.25,
            weight: None,
            fill: FillMethod::None,
            outlier_iqr: None,
            smooth: 0,
            smooth_centred: false,
            allow_interpolate: false,
            allow_lockbox: false,
            validation_frac: 0.0,
            date_only: DateOnlyPolicy::Error,
            time_unit: None,
            tz: None,
            skip_malformed: false,
            center: Center::Mean,
            scale: Scale::Std,
            filter: None,
            exclude: None,
            train_select: None,
            test_select: None,
            split: None,
            time: None,
            group_by: None,
            gap: 0,
            folds: 1,
            lags: 0,
            history_frac: 0.0,
            gap_break: 0.0,
            null_runs: 0,
            bootstrap_resamples: 0,
            cv: None,
            cv_k: 5,
            stability_runs: 0,
            top: 0,
            eval_expr: None,
        }
    }
}

/// CLI options the browser UI deliberately does not implement, with the reason
/// shown to the user. Every other schema option has an arm in [`parse_args`].
const UNSUPPORTED: &[(&str, &str)] = &[
    ("--max-memory", "the server does not enforce the CLI memory guard"),
    ("--force", "the server does not enforce the CLI memory guard"),
    ("--gpu-memory", "the server does not take a device memory cap"),
    ("--dispatch", "the server does not force the device route"),
    ("--select-by", "the GUI uses the engine's default selection criterion"),
    ("--objective", "the GUI does not run the long/flat P&L objective"),
    ("--position", "the GUI does not run the long/flat P&L objective"),
    ("--allow-short", "the GUI does not run the long/flat P&L objective"),
    ("--threshold", "the GUI does not run the long/flat P&L objective"),
    ("--cost-bps", "the GUI does not run the long/flat P&L objective"),
    ("--slippage-bps", "the GUI does not run the long/flat P&L objective"),
    ("--cash-return", "the GUI does not run the long/flat P&L objective"),
    ("--funding", "the GUI does not run the long/flat P&L objective"),
    ("--max-position", "the GUI does not run the long/flat P&L objective"),
    ("--min-trades", "the GUI does not run the long/flat P&L objective"),
    ("--min-holding", "the GUI does not run the long/flat P&L objective"),
    ("--baseline-samples", "the GUI baseline uses the engine's default sample size"),
    ("--null-block", "the GUI null control only shuffles single rows"),
    ("--implicit", "the implicit target-as-input transform is not wired into the form"),
    ("--implicit-derivative", "the implicit target-as-input transform is not wired into the form"),
    ("--units", "unit declarations are not wired into the browser form"),
    ("--require-dimensions", "unit declarations are not wired into the browser form"),
    ("--seed-expr", "seed expressions are not parsed by the server"),
    ("--seed-file", "seed files are not parsed by the server"),
    ("--function", "building-block definitions are not parsed by the server"),
    ("--allow-backend-change", "the GUI does not resume checkpoints"),
    ("--allow-build-change", "the GUI does not resume checkpoints"),
    ("--run-manifest", "the server owns the response stream; use the CLI to write a manifest"),
    ("--json-out", "the server owns the response stream; use the CLI to write a result file"),
    ("--out", "the server owns the response stream; use the CLI to write the Pareto TSV"),
    ("--save", "checkpoints depend on a process lifecycle the stream does not have"),
    ("--save-every", "checkpoints depend on a process lifecycle the stream does not have"),
    ("--resume", "checkpoints depend on a process lifecycle the stream does not have"),
    ("--report", "Markdown reports are produced by the CLI"),
    ("--report-out", "Markdown reports are produced by the CLI"),
    ("--report-repeats", "Markdown benchmark reports are produced by the CLI"),
];

/// The user-facing rejection for a documented CLI option the browser does not
/// implement. Keeping it in one place avoids a "rendered but unknown" message.
fn unsupported_error(flag: &str) -> String {
    let reason = UNSUPPORTED
        .iter()
        .find(|(name, _)| *name == flag)
        .map(|(_, reason)| *reason)
        .unwrap_or("not supported by the web front end");
    format!(
        "{flag} is a CLI option but is not supported by the web front end; use the rcsr CLI for it ({reason})"
    )
}

/// Parse the arguments the browser sends. Public so the schema-contract test
/// can prove every documented option is either accepted or explicitly
/// rejected, instead of only comparing JSON documents.
pub fn parse_web_args_for_test(args: &[String]) -> Result<(), String> {
    parse_args(args).map(|_| ())
}

/// The documented options the browser rejects, with the reason.
pub fn rejected_options() -> &'static [(&'static str, &'static str)] {
    UNSUPPORTED
}

fn parse_args(args: &[String]) -> Result<(RawOptions, search::Config), String> {
    let mut raw = RawOptions::default();
    let mut cfg = search::Config::default();
    // #91: the GPU is opt-in here too; a default run is CPU.
    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].clone();
        let take = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{} requires a value", flag))
        };
        match flag.as_str() {
            "--data" => raw.data = Some(take(&mut i)?),
            "--target" => raw.target = Some(take(&mut i)?),
            "--vars" => {
                raw.vars = take(&mut i)?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--dtype" => raw.dtype = Some(take(&mut i)?),
            "--max-levels" => raw.max_levels = int(&take(&mut i)?, "--max-levels", 1)?,
            "--skip-malformed" => raw.skip_malformed = true,
            "--test-frac" => raw.test_frac = float(&take(&mut i)?, "--test-frac")?,
            "--validation-frac" => {
                raw.validation_frac = float(&take(&mut i)?, "--validation-frac")?
            },
            "--weight" => raw.weight = Some(take(&mut i)?),
            "--fill" => raw.fill = FillMethod::parse(&take(&mut i)?)?,
            "--outlier-iqr" => {
                let v = float(&take(&mut i)?, "--outlier-iqr")?;
                raw.outlier_iqr = if v > 0.0 { Some(v) } else { None };
            }
            "--smooth" => raw.smooth = int(&take(&mut i)?, "--smooth", 0)?,
            "--smooth-centred" => {
                raw.smooth = int(&take(&mut i)?, "--smooth-centred", 0)?;
                raw.smooth_centred = raw.smooth > 0;
            }
            "--allow-interpolate" => raw.allow_interpolate = true,
            "--allow-lockbox" => raw.allow_lockbox = true,
            "--center" => raw.center = Center::parse(&take(&mut i)?)?,
            "--scale" => raw.scale = Scale::parse(&take(&mut i)?)?,
            "--filter" => raw.filter = Some(take(&mut i)?),
            "--exclude" => raw.exclude = Some(take(&mut i)?),
            "--train-select" => raw.train_select = Some(take(&mut i)?),
            "--test-select" => raw.test_select = Some(take(&mut i)?),
            "--split" => raw.split = Some(take(&mut i)?),
            "--time" => raw.time = Some(take(&mut i)?),
            "--date-only" => raw.date_only = DateOnlyPolicy::parse(&take(&mut i)?)?,
            "--time-unit" => raw.time_unit = Some(TimeUnit::parse(&take(&mut i)?)?),
            "--tz" => raw.tz = Some(data::TimeSpec::parse_tz(&take(&mut i)?)?),
            "--group-by" => raw.group_by = Some(take(&mut i)?),
            "--gap" => raw.gap = int(&take(&mut i)?, "--gap", 0)?,
            "--folds" => raw.folds = int(&take(&mut i)?, "--folds", 1)?,
            "--lags" => raw.lags = int(&take(&mut i)?, "--lags", 0)?,
            "--history-frac" => raw.history_frac = float(&take(&mut i)?, "--history-frac")?,
            "--gap-break" => raw.gap_break = float(&take(&mut i)?, "--gap-break")?,
            "--null-runs" => raw.null_runs = int(&take(&mut i)?, "--null-runs", 0)?,
            "--bootstrap-error" => {
                raw.bootstrap_resamples = int(&take(&mut i)?, "--bootstrap-error", 0)?
            }
            "--cv" => raw.cv = Some(take(&mut i)?),
            "--cv-k" => raw.cv_k = int(&take(&mut i)?, "--cv-k", 2)?,
            "--stability" => raw.stability_runs = int(&take(&mut i)?, "--stability", 0)?,
            "--metric" => {
                let v = take(&mut i)?;
                cfg.metric = Metric::parse_opt(&v)
                    .ok_or_else(|| format!("unknown --metric '{}'", v))?;
            }
            "--generations" => cfg.generations = int(&take(&mut i)?, "--generations", 1)?,
            "--islands" => cfg.islands = int(&take(&mut i)?, "--islands", 1)?,
            "--pop" => cfg.pop = int(&take(&mut i)?, "--pop", 2)?,
            "--max-complexity" => cfg.max_complexity = int(&take(&mut i)?, "--max-complexity", 1)?,
            "--op-complexity" => cfg.complexity_model.apply_op_costs(&take(&mut i)?)?,
            "--complexity-weights" => cfg.complexity_model.apply_weights(&take(&mut i)?)?,
            "--depth-weight" => {
                let v = float(&take(&mut i)?, "--depth-weight")?;
                cfg.complexity_model.set_depth_weight(v)?;
            }
            "--seed" => cfg.seed = u32int(&take(&mut i)?, "--seed")?,
            "--ops" => cfg.ops = Op::parse_list(&take(&mut i)?),
            "--eval-expr" => raw.eval_expr = Some(take(&mut i)?),
            "--fit-iters" => cfg.fit_iters = int(&take(&mut i)?, "--fit-iters", 0)?,
            "--fit-report" => cfg.fit_report = true,
            "--fit-samples" => cfg.fit_samples = int(&take(&mut i)?, "--fit-samples", 0)?,
            "--fit-strategy" => {
                let v = take(&mut i)?;
                cfg.fit_strategy = fit::FitStrategy::parse(&v)
                    .ok_or_else(|| format!("unknown --fit-strategy '{}'", v))?;
            }
            "--no-fit" => cfg.fit_iters = 0,
            "--no-dedup" => cfg.dedup = false,
            "--no-front-parents" => cfg.front_parents = false,
            "--no-simd" => cfg.simd = false,
            "--qd" => {
                if cfg.qd_bins == 0 {
                    cfg.qd_bins = 16;
                }
            }
            "--qd-cells" => {
                let v = int(&take(&mut i)?, "--qd-cells", 2)?;
                if v > 256 {
                    return Err(format!("--qd-cells needs an integer in 2..=256, got '{}'", v));
                }
                cfg.qd_bins = v;
            }
            "--qd-cell-cap" => {
                let v = int(&take(&mut i)?, "--qd-cell-cap", 1)?;
                if v > 64 {
                    return Err(format!("--qd-cell-cap needs an integer in 1..=64, got '{}'", v));
                }
                cfg.qd_cell_cap = v;
            }
            "--pipeline-depth" => cfg.pipeline_depth = int(&take(&mut i)?, "--pipeline-depth", 0)?,
            "--threads" => cfg.threads = int(&take(&mut i)?, "--threads", 1)?,
            "--gpu" => cfg.gpu = true,
            "--no-gpu" | "--cpu" => cfg.gpu = false,
            "--predictor-samples" => {
                cfg.predictor_samples = Some(int(&take(&mut i)?, "--predictor-samples", 2)?)
            }
            "--patience" => cfg.patience = int(&take(&mut i)?, "--patience", 0)?,
            "--time-budget" => cfg.time_budget = Some(float(&take(&mut i)?, "--time-budget")?),
            "--top" => raw.top = int(&take(&mut i)?, "--top", 0)?,
            // Presentation-only CLI flags: the HTTP response is always NDJSON.
            "--format" | "--json" | "--ndjson" | "--progress" | "--quiet" => {
                if flag == "--format" {
                    let _ = take(&mut i)?;
                }
            }
            // Documented CLI options the browser deliberately does not
            // implement. They are rejected by name with a reason, never called
            // "unknown"; tests/gui.rs covers every schema option.
            other if UNSUPPORTED.iter().any(|(name, _)| *name == other) => {
                return Err(unsupported_error(other));
            }
            // Not part of the browser MVP: these either write server-side
            // files or depend on a process lifecycle the stream does not have.
            "--out" | "--save" | "--save-every" | "--resume" | "--report"
            | "--report-out" | "--report-repeats" => {
                return Err(format!(
                    "{flag} is a CLI option but is not supported by the web front end; use the rcsr CLI for it"
                ))
            }
            other => {
                // #58: a documented CLI option must never be reported as
                // "unknown" just because the browser UI has not wired it yet.
                if crate::schema::has_option(other) {
                    return Err(format!(
                        "{other} is a CLI option but is not supported by the web front end; use the rcsr CLI for it"
                    ));
                }
                return Err(format!("unknown argument '{}'", other));
            }
        }
        i += 1;
    }
    // #223 + #229 + #222: the batched GPU LM minimises squared error,
    // log-squashed loss or mean absolute error. The web front end must refuse a
    // metric with no native device objective, not silently fit it on the CPU.
    if cfg.fit_strategy == fit::FitStrategy::Gpu {
        if let Some(why) = fit::gpu_fit_metric_error(cfg.metric) {
            return Err(why);
        }
    }
    Ok((raw, cfg))
}

/// Parse the arguments the browser sends. Public so the schema-contract test
/// can prove every documented option is either accepted or explicitly
/// rejected, instead of only comparing JSON documents.
pub fn parse_web_args(args: &[String]) -> Result<(), String> {
    parse_args(args).map(|_| ())
}

fn int(text: &str, flag: &str, min: usize) -> Result<usize, String> {
    match text.parse::<usize>() {
        Ok(v) if v >= min => Ok(v),
        _ => Err(format!("{} needs an integer >= {}, got '{}'", flag, min, text)),
    }
}

fn float(text: &str, flag: &str) -> Result<f64, String> {
    match text.parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        _ => Err(format!("{} needs a finite number, got '{}'", flag, text)),
    }
}

fn u32int(text: &str, flag: &str) -> Result<u32, String> {
    match text.parse::<u32>() {
        Ok(v) => Ok(v),
        _ => Err(format!("{} needs an integer from 0 to 4294967295, got '{}'", flag, text)),
    }
}

fn variables_detail_json(types: &[data::VariableDType]) -> String {
    types
        .iter()
        .map(|v| {
            format!(
                "{{\"name\":{},\"dtype\":{},\"source\":{}}}",
                jstr(&v.name),
                jstr(v.dtype.name()),
                v.source
                    .as_ref()
                    .map(|s| jstr(s))
                    .unwrap_or_else(|| "null".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_filter(flag: &str, src: Option<&str>, table: &data::Table) -> Result<Option<data::Filter>, String> {
    let Some(src) = src else {
        return Ok(None);
    };
    let filter_vars: Vec<String> = (0..table.names.len())
        .filter(|&j| table.numeric[j])
        .map(|j| table.names[j].clone())
        .collect();
    match expr::parse(src, &filter_vars) {
        Ok(tree) => Ok(Some(data::Filter {
            tree,
            vars: filter_vars,
            expression: src.to_string(),
        })),
        Err(e) => Err(format!("cannot parse {}: {} (expression: {})", flag, e, src)),
    }
}

fn column_index(table: &data::Table, name: &str, what: &str) -> Result<usize, String> {
    table
        .index_of(name)
        .map_err(|_| format!("{} column '{}' not found (columns: {:?})", what, name, table.names))
}

impl RunPlan {
    fn new(request: RunRequest) -> Result<RunPlan, String> {
        let (raw, cfg) = parse_args(&request.args)?;

        let data_path = raw
            .data
            .clone()
            .or_else(|| request.table.as_ref().map(|_| "-".to_string()))
            .ok_or_else(|| "--data is required (or send a table)".to_string())?;
        let target = raw
            .target
            .clone()
            .ok_or_else(|| "--target is required".to_string())?;
        let text = if data_path == "-" {
            request
                .table
                .clone()
                .ok_or_else(|| "with --data - the request must carry a \"table\" string".to_string())?
        } else {
            std::fs::read_to_string(&data_path)
                .map_err(|e| format!("{}: {}", data_path, e))?
        };
        let table = data::parse_table_with(&text, raw.skip_malformed)?;

        if raw.weight.is_some() && !metrics::weights_supported(cfg.metric) {
            return Err(format!(
                "--weight is not supported by the '{}' metric; weighted metrics: {}",
                cfg.metric.name(),
                metrics::weighted_metric_names().join(", ")
            ));
        }
        if raw.weight.is_some() && raw.bootstrap_resamples > 0 {
            return Err("--weight cannot be combined with --bootstrap-error".to_string());
        }

        let target_idx = column_index(&table, &target, "target")?;
        let weight_col = match &raw.weight {
            Some(name) => Some(column_index(&table, name, "weight")?),
            None => None,
        };

        let filter = parse_filter("--filter", raw.filter.as_deref(), &table)?;
        let exclude = parse_filter("--exclude", raw.exclude.as_deref(), &table)?;
        let train_select = parse_filter("--train-select", raw.train_select.as_deref(), &table)?;
        let test_select = parse_filter("--test-select", raw.test_select.as_deref(), &table)?;

        let mut time_col = match &raw.time {
            Some(name) => Some(column_index(&table, name, "time")?),
            None => None,
        };
        let mut group_col = match &raw.group_by {
            Some(name) => Some(column_index(&table, name, "group")?),
            None => None,
        };

        let vars_explicit = !raw.vars.is_empty();
        let mut vars = raw.vars.clone();
        if vars.is_empty() {
            vars = (0..table.names.len())
                .filter(|&j| {
                    table.numeric[j]
                        && table.names[j] != target
                        && Some(j) != time_col
                        && Some(j) != group_col
                        && Some(j) != weight_col
                })
                .map(|j| table.names[j].clone())
                .collect();
        }

        let dtype_spec = match raw.dtype.as_deref() {
            Some(s) => data::parse_dtype_spec(s)?,
            None => Vec::new(),
        };
        for (name, _) in &dtype_spec {
            if name == &target {
                return Err(format!(
                    "column '{}' is the target; --dtype applies to input variables",
                    name
                ));
            }
            let j = column_index(&table, name, "dtype")?;
            if Some(j) == time_col || Some(j) == group_col || Some(j) == weight_col {
                return Err(format!(
                    "column '{}' is used for --time/--group-by/--weight and cannot be an input",
                    name
                ));
            }
            if !vars.iter().any(|v| v == name) {
                if vars_explicit {
                    return Err(format!(
                        "column '{}' is not an input variable; add it to --vars or drop it from --dtype",
                        name
                    ));
                }
                vars.push(name.clone());
            }
        }
        if vars.is_empty() {
            return Err("no input variables".to_string());
        }

        let known_vars: Vec<usize> = vars.iter().filter_map(|v| table.index_of(v).ok()).collect();
        let exclusion = |a: Option<usize>, b: Option<usize>| -> Vec<usize> {
            let mut e = vec![target_idx];
            e.extend_from_slice(&known_vars);
            if let Some(j) = a {
                e.push(j);
            }
            if let Some(j) = b {
                e.push(j);
            }
            if let Some(j) = weight_col {
                e.push(j);
            }
            e
        };

        let mut notes: Vec<String> = Vec::new();
        let requested = match raw.split.as_deref() {
            None | Some("auto") => None,
            Some(s) => Some(SplitMode::parse(s)?),
        };
        let mode = if let Some(m) = requested {
            m
        } else if raw.split.as_deref() == Some("auto") {
            let (m, tc, gc, nt) = data::suggest_split(&table, target_idx, &known_vars);
            if time_col.is_none() {
                time_col = tc;
            }
            if group_col.is_none() {
                group_col = gc;
            }
            notes.extend(nt);
            notes.push(format!("using the {} split", m.name()));
            m
        } else {
            match (time_col.is_some(), group_col.is_some()) {
                (true, true) => SplitMode::TemporalGroup,
                (true, false) => SplitMode::Temporal,
                (false, true) => SplitMode::Group,
                (false, false) => {
                    let (guess, _, _, nt) = data::suggest_split(&table, target_idx, &known_vars);
                    if guess != SplitMode::Random {
                        notes.push(format!(
                            "warning: the rows do not look exchangeable ({}); consider --split auto",
                            nt.join("; ")
                        ));
                    }
                    SplitMode::Random
                }
            }
        };

        if matches!(mode, SplitMode::Temporal | SplitMode::TemporalGroup)
            && time_col.is_none()
        {
            match data::find_time_col(&table, &exclusion(None, group_col)) {
                Some(j) => {
                    notes.push(format!("time column: '{}'", table.names[j]));
                    time_col = Some(j);
                }
                None => {
                    return Err(format!(
                        "--split {} needs --time <col> (no time-like column found)",
                        mode.name()
                    ))
                }
            }
        }
        if matches!(mode, SplitMode::Group | SplitMode::TemporalGroup)
            && group_col.is_none()
        {
            match data::find_group_col(&table, &exclusion(time_col, None)) {
                Some(j) => {
                    notes.push(format!("group column: '{}'", table.names[j]));
                    group_col = Some(j);
                }
                None => {
                    return Err(format!(
                        "--split {} needs --group-by <col> (no group-like column found)",
                        mode.name()
                    ))
                }
            }
        }
        if raw.gap > 0 && !matches!(mode, SplitMode::Temporal | SplitMode::TemporalGroup) {
            notes.push("--gap only applies to temporal splits; ignored".to_string());
        }

        let typed = data::apply_dtypes(&table, &vars, &dtype_spec, raw.max_levels)
            .map_err(|e| e.message)?;
        let mut table = typed.table;
        vars = typed.vars;
        let mut dtypes = typed.dtypes;
        let mut variable_types = typed.variables;
        let time_spec = data::TimeSpec {
            date_only: raw.date_only,
            unit: raw.time_unit,
            tz: raw.tz,
        };
        let mut lag_report = data::LagReport::default();
        if raw.lags > 0 || raw.history_frac > 0.0 {
            let lag_spec = data::LagSpec {
                lags: raw.lags,
                history_frac: raw.history_frac,
                gap_break: raw.gap_break,
                time: time_spec.clone(),
            };
            let (t2, v2, rep) = data::add_lags(&table, &vars, &lag_spec, time_col, group_col)?;
            table = t2;
            vars = v2;
            lag_report = rep;
        }
        notes.extend(lag_report.notes.iter().cloned());

        if dtypes.len() < vars.len() {
            let base = dtypes.len();
            for v in &vars[base..] {
                let source = v.rsplit_once("_lag").and_then(|(stem, n)| {
                    if n.parse::<usize>().is_ok() {
                        variable_types
                            .iter()
                            .find(|t| t.name == stem)
                            .map(|t| (t.dtype, t.name.clone()))
                    } else {
                        None
                    }
                });
                let (kind, src) = match source {
                    Some((k, s)) => (k, Some(s)),
                    None => (DType::Continuous, None),
                };
                dtypes.push(kind);
                variable_types.push(data::VariableDType {
                    name: v.clone(),
                    dtype: kind,
                    source: src,
                });
            }
        }

        let spec = data::SplitSpec {
            mode,
            time_col,
            group_col,
            test_frac: raw.test_frac,
            gap: raw.gap,
            validation_frac: raw.validation_frac,
            folds: raw.folds.max(1),
            train_select,
            test_select,
            exclude,
            time: time_spec,
            allow_lockbox: raw.allow_lockbox,
            declared_lockbox: None,
            purge_seconds: None,
            scale_column: None,
            trading: None,
            canonical_train_order: false,
        };
        let cv_dtypes = dtypes.clone();
        let prep = data::PrepSpec {
            fill: raw.fill,
            outlier_iqr: raw.outlier_iqr,
            smooth: raw.smooth,
            smooth_centred: raw.smooth_centred,
            allow_interpolate: raw.allow_interpolate,
            center: raw.center,
            scale: raw.scale,
            filter,
            dtypes,
        };
        let mut rng = Mt::new(cfg.seed);
        let prepared = data::prepare_full_weighted_with(
            &table,
            &vars,
            &target,
            raw.weight.as_deref(),
            &spec,
            &mut rng,
            &prep,
        )?;
        notes.extend(prepared.report.notes.iter().cloned());

        let cv_mode = match raw.cv.as_deref() {
            Some("k-fold") => Some(data::CvMode::KFold),
            Some("rolling") => Some(data::CvMode::Rolling),
            Some("purged") => Some(data::CvMode::Purged),
            Some(other) => {
                return Err(format!(
                    "unknown --cv '{}' (expected k-fold, rolling or purged)",
                    other
                ))
            }
            None => None,
        };

        if let Some(src) = &raw.eval_expr {
            expr::parse(src, &vars).map_err(|e| format!("cannot parse --eval-expr: {}", e))?;
        }

        // #223: resolve the fit arm like the CLI does. The web front end is a
        // fresh run, never a resume.
        let mut cfg = cfg;
        let named = request.args.iter().position(|a| a == "--fit-strategy");
        let fit_request = match named {
            None => Some(fit::FitRequest::Unset),
            Some(_) if cfg.fit_strategy == fit::FitStrategy::Auto => Some(fit::FitRequest::Auto),
            Some(_) => None,
        };
        if let Some(fit_request) = fit_request {
            let resolved = search::resolve_fit_strategy(&cfg, &prepared.train, fit_request);
            cfg.fit_strategy = resolved.strategy;
            cfg.fit_admission = resolved.admission;
        }

        Ok(RunPlan {
            cfg,
            prepared,
            table,
            dtypes: cv_dtypes,
            vars,
            variable_types,
            target,
            notes,
            weight: raw.weight,
            null_runs: raw.null_runs,
            bootstrap_resamples: raw.bootstrap_resamples,
            cv_mode,
            cv_k: raw.cv_k.max(2),
            stability_runs: raw.stability_runs,
            top: raw.top,
            eval_expr: raw.eval_expr,
        })
    }

    fn execute(mut self, writer: &Arc<Mutex<TcpStream>>) -> Result<(), String> {
        let prepared = &self.prepared;
        let start = Instant::now();
        let folds_json: Vec<String> = prepared
            .fold_samples
            .iter()
            .map(|f| f.len().to_string())
            .collect();
        emit(
            writer,
            &format!(
                "{{\"event\":\"start\",\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"variables_detail\":[{}],\"train_rows\":{},\"test_rows\":{},\"split\":{{\"mode\":{},\"gap\":{},\"folds\":[{}]}}}}",
                jstr(VERSION),
                self.cfg.seed,
                jstr(self.cfg.metric.name()),
                jstr(&self.target),
                self.vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                variables_detail_json(&self.variable_types),
                prepared.train.len(),
                prepared.test.len(),
                jstr(prepared.report.mode),
                prepared.report.gap,
                folds_json.join(",")
            ),
        );
        if prepared.prep.active() {
            let obj = prepared.prep.json();
            let inner = obj.trim_matches(|c| c == '{' || c == '}');
            emit(writer, &format!("{{\"event\":\"preparation\",{}}}", inner));
        }
        for note in &self.notes {
            emit(
                writer,
                &format!(
                    "{{\"event\":\"note\",\"message\":{}}}",
                    jstr(note)
                ),
            );
        }

        if let Some(src) = &self.eval_expr {
            let line = self.eval_line(src)?;
            emit(writer, &line);
            emit(
                writer,
                &format!(
                    "{{\"event\":\"done\",\"generations_run\":0,\"converged_early\":false,\"elapsed_seconds\":{},\"pareto_members\":0}}",
                    jnum(start.elapsed().as_secs_f64())
                ),
            );
            return Ok(());
        }

        let progress_writer = writer.clone();
        let progress = Arc::new(move |p: search::Progress| {
            emit(
                &progress_writer,
                &format!(
                    "{{\"event\":\"progress\",\"generation\":{},\"total\":{},\"best_error\":{},\"pareto_members\":{},\"convergence\":{}}}",
                    p.generation,
                    p.total_generations,
                    jnum(p.best_error),
                    p.pareto_members,
                    p.convergence.json()
                ),
            );
        });
        self.cfg.progress = Some(progress);

        let outcome = search::run(&self.cfg, &prepared.train, &prepared.test);
        let elapsed = start.elapsed().as_secs_f64();

        let fold_samples: Vec<data::Sample> = if prepared.fold_samples.is_empty() {
            vec![prepared.test.clone()]
        } else {
            prepared.fold_samples.clone()
        };
        let sorted = outcome.archive.sorted();
        let mut boot_rng = Mt::new(self.cfg.seed ^ 0xB007_57AA);
        let mut rows: Vec<Row> = sorted
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let den = expr::simplify_display(&expr::denormalize(
                    &expr::simplify_display(&s.tree),
                    &prepared.means,
                    &prepared.stds,
                ));
                let fold_errors: Vec<f64> = fold_samples
                    .iter()
                    .map(|f| search::eval_error(&s.tree, f, self.cfg.metric))
                    .collect();
                let test_error = if fold_errors.is_empty() {
                    f64::INFINITY
                } else {
                    fold_errors.iter().sum::<f64>() / fold_errors.len() as f64
                };
                let pred: Vec<f64> = prepared
                    .train
                    .x
                    .iter()
                    .map(|row| expr::eval(&s.tree, row))
                    .collect();
                let resid: Vec<f64> = pred
                    .iter()
                    .zip(&prepared.train.y)
                    .map(|(p, t)| p - t)
                    .collect();
                let rss: f64 = resid.iter().map(|e| e * e).sum();
                let kconst = expr::count_constants(&s.tree);
                let (_, aicc, bic) =
                    diagnostics::information_criteria(prepared.train.len(), rss, kconst);
                let dw = diagnostics::durbin_watson(&resid);
                let resid_tests = diagnostics::residual_tests(&resid, &pred);
                let train_error = search::eval_error(&s.tree, &prepared.train, self.cfg.metric);
                let overfit = if train_error > 0.0 {
                    test_error / train_error
                } else if test_error > 0.0 {
                    f64::INFINITY
                } else {
                    1.0
                };
                let bootstrap = if self.bootstrap_resamples > 0 {
                    Some(diagnostics::bootstrap_error(
                        &s.tree,
                        &prepared.train,
                        self.cfg.metric,
                        self.bootstrap_resamples,
                        self.cfg.fit_iters,
                        &mut boot_rng,
                    ))
                } else {
                    None
                };
                Row {
                    rank: k + 1,
                    complexity: s.cost,
                    k_constants: kconst,
                    train_error,
                    test_error,
                    aicc,
                    bic,
                    dw,
                    resid_tests,
                    bootstrap,
                    overfit,
                    fold_errors,
                    constant_errors: if prepared.train.w.is_empty() {
                        fit::constant_errors(&s.tree, &prepared.train.x, &prepared.train.y)
                    } else {
                        fit::constant_errors_weighted(
                            &s.tree,
                            &prepared.train.x,
                            &prepared.train.y,
                            &prepared.train.w,
                        )
                    },
                    shape: stability::shape(&expr::simplify(&s.tree)),
                    stability: None,
                    expression: expr::to_string(&den, &self.vars),
                }
            })
            .collect();

        let best_resid = rows
            .iter()
            .min_by(|a, b| {
                a.train_error
                    .partial_cmp(&b.train_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.resid_tests);

        let baselines =
            diagnostics::Baselines::fit(&prepared.train, &fold_samples, self.cfg.metric);

        let stab: Option<stability::Stability> = if self.stability_runs > 0 {
            let st = stability::run(
                &self.cfg,
                &prepared.train,
                &prepared.test,
                self.stability_runs,
                &self.vars,
            );
            for r in rows.iter_mut() {
                r.stability = Some(st.frequency(&r.shape));
            }
            Some(st)
        } else {
            None
        };

        let cv: Option<CvReport> = match self.cv_mode {
            Some(mode) => Some(self.cross_validate(mode)?),
            None => None,
        };

        let mut null_control: Option<diagnostics::NullControl> = None;
        if self.null_runs > 0 {
            let observed = outcome
                .archive
                .items
                .iter()
                .map(|s| s.error)
                .fold(f64::INFINITY, f64::min);
            let saved_progress = self.cfg.progress.take();
            let mut best_null = Vec::with_capacity(self.null_runs);
            for r in 0..self.null_runs {
                let mut rng =
                    Mt::new(self.cfg.seed ^ (0x51ED_2701u32.wrapping_mul(r as u32 + 1)));
                let mut y = prepared.train.y.clone();
                for j in (1..y.len()).rev() {
                    let t = rng.usize_below(j + 1);
                    y.swap(j, t);
                }
                let shuffled = data::Sample {
                    x: prepared.train.x.clone(),
                    y,
                    w: prepared.train.w.clone(),
                    ys: Vec::new(),
                    pairs: None,
                };
                let out = search::run(&self.cfg, &shuffled, &shuffled);
                let b = out
                    .archive
                    .items
                    .iter()
                    .map(|s| s.error)
                    .fold(f64::INFINITY, f64::min);
                best_null.push(b);
            }
            self.cfg.progress = saved_progress;
            null_control = Some(diagnostics::NullControl::summarise(observed, best_null));
        }

        if self.top > 0 {
            rows.truncate(self.top);
        }

        for r in &rows {
            emit(
                writer,
                &format!(
                    "{{\"event\":\"solution\",\"rank\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{},\"stability\":{},\"expression\":{}}}",
                    r.rank,
                    r.complexity,
                    r.k_constants,
                    jnum(r.train_error),
                    jnum(r.test_error),
                    jnum(r.aicc),
                    jnum(r.bic),
                    jnum(r.dw),
                    jnum(r.overfit),
                    r.fold_errors.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                    bootstrap_json(&r.bootstrap),
                    constant_errors_json(&r.constant_errors),
                    r.stability.map(jnum).unwrap_or_else(|| "null".to_string()),
                    jstr(&r.expression)
                ),
            );
        }
        if let Some(c) = &null_control {
            emit(
                writer,
                &format!(
                    "{{\"event\":\"null_control\",\"runs\":{},\"min\":{},\"median\":{},\"p_value\":{},\"verdict\":{}}}",
                    c.runs,
                    jnum(c.null_min),
                    jnum(c.null_median),
                    jnum(c.p_value),
                    jstr(c.verdict())
                ),
            );
        }
        emit(
            writer,
            &format!(
                "{{\"event\":\"baselines\",\"metric\":{},\"mean\":{{\"train_error\":{},\"test_error\":{}}},\"linear\":{{\"train_error\":{},\"test_error\":{}}},\"knn\":{{\"k\":{},\"train_error\":{},\"test_error\":{}}}}}",
                jstr(self.cfg.metric.name()),
                jnum(baselines.mean.train_error),
                jnum(baselines.mean.test_error),
                jnum(baselines.linear.train_error),
                jnum(baselines.linear.test_error),
                baselines.knn_k,
                jnum(baselines.knn.train_error),
                jnum(baselines.knn.test_error)
            ),
        );
        if let Some(cv) = &cv {
            let folds: Vec<String> = cv
                .folds
                .iter()
                .map(|r| {
                    format!(
                        "{{\"train_error\":{},\"test_error\":{},\"complexity\":{},\"expression\":{}}}",
                        jnum(r.train_error),
                        jnum(r.test_error),
                        r.complexity,
                        jstr(&r.expression)
                    )
                })
                .collect();
            emit(
                writer,
                &format!(
                    "{{\"event\":\"cross_validation\",\"mode\":{},\"k\":{},\"mean_train_error\":{},\"mean_test_error\":{},\"std_test_error\":{},\"folds\":[{}]}}",
                    jstr(cv.mode),
                    cv.k,
                    jnum(cv.mean_train()),
                    jnum(cv.mean_test()),
                    jnum(cv.std_test()),
                    folds.join(",")
                ),
            );
        }
        if let Some(rt) = &best_resid {
            emit(
                writer,
                &format!(
                    "{{\"event\":\"residual_tests\",\"n\":{},\"lags\":{},\"ljung_box\":{{\"statistic\":{},\"df\":{},\"p_value\":{}}},\"breusch_pagan\":{{\"statistic\":{},\"df\":{},\"p_value\":{}}},\"skewness\":{},\"kurtosis\":{}}}",
                    rt.n,
                    rt.lags,
                    jnum(rt.ljung_box.statistic),
                    rt.ljung_box.lags,
                    jnum(rt.ljung_box.p_value),
                    jnum(rt.breusch_pagan.statistic),
                    rt.breusch_pagan.df,
                    jnum(rt.breusch_pagan.p_value),
                    jnum(rt.skewness),
                    jnum(rt.kurtosis)
                ),
            );
        }
        if let Some(st) = &stab {
            let forms: Vec<String> = st
                .forms
                .iter()
                .map(|f| {
                    format!(
                        "{{\"shape\":{},\"count\":{},\"frequency\":{},\"example\":{}}}",
                        jstr(&f.shape),
                        f.count,
                        jnum(f.frequency),
                        jstr(&f.example)
                    )
                })
                .collect();
            let variables: Vec<String> = st
                .variables
                .iter()
                .map(|(i, c)| {
                    format!(
                        "{{\"index\":{},\"name\":{},\"count\":{},\"frequency\":{}}}",
                        i,
                        jstr(self.vars.get(*i).map(|s| s.as_str()).unwrap_or("")),
                        c,
                        jnum(if st.runs > 0 {
                            *c as f64 / st.runs as f64
                        } else {
                            0.0
                        })
                    )
                })
                .collect();
            emit(
                writer,
                &format!(
                    "{{\"event\":\"stability\",\"runs\":{},\"found\":{},\"forms\":[{}],\"variables\":[{}]}}",
                    st.runs,
                    st.found,
                    forms.join(","),
                    variables.join(",")
                ),
            );
        }
        emit(
            writer,
            &format!(
                "{{\"event\":\"done\",\"generations_run\":{},\"converged_early\":{},\"elapsed_seconds\":{},\"pareto_members\":{}}}",
                outcome.generations,
                outcome.converged_early,
                jnum(elapsed),
                outcome.archive.items.len()
            ),
        );
        Ok(())
    }

    fn cross_validate(&self, mode: data::CvMode) -> Result<CvReport, String> {
        let folds = data::cv_folds_typed(
            &self.table,
            &self.vars,
            &self.target,
            self.weight.as_deref(),
            mode,
            self.cv_k,
            self.prepared.report.gap,
            self.cfg.seed,
            &self.dtypes,
        )?;
        let mut fold_rows = Vec::new();
        for f in &folds {
            let out = search::run(&self.cfg, &f.train, &f.test);
            let best = out.archive.sorted().into_iter().min_by(|a, b| {
                a.error
                    .partial_cmp(&b.error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(s) = best {
                let test_error = search::eval_error(&s.tree, &f.test, self.cfg.metric);
                let train_error = search::eval_error(&s.tree, &f.train, self.cfg.metric);
                let den = expr::simplify_display(&expr::denormalize(
                    &expr::simplify_display(&s.tree),
                    &f.means,
                    &f.stds,
                ));
                fold_rows.push(CvFoldRow {
                    train_error,
                    test_error,
                    complexity: s.cost,
                    expression: expr::to_string(&den, &self.vars),
                });
            }
        }
        Ok(CvReport {
            mode: mode.name(),
            k: self.cv_k,
            folds: fold_rows,
        })
    }

    fn eval_line(&self, src: &str) -> Result<String, String> {
        let tree = expr::parse(src, &self.vars).map_err(|e| format!("{}", e))?;
        let k = expr::count_constants(&tree);
        let complexity = self.cfg.complexity_model.complexity(&tree);
        let shown = expr::to_string(&tree, &self.vars);
        let prepared = &self.prepared;
        let raw_row = |row: &[f64]| -> Vec<f64> {
            row.iter()
                .enumerate()
                .map(|(j, v)| {
                    v * prepared.stds.get(j).copied().unwrap_or(1.0)
                        + prepared.means.get(j).copied().unwrap_or(0.0)
                })
                .collect()
        };
        let score = |s: &data::Sample| -> f64 {
            let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(&tree, &raw_row(r))).collect();
            metrics::score_sample(&pred, s, self.cfg.metric, k)
        };
        let train_error = score(&prepared.train);
        let fold_errors: Vec<f64> = prepared.fold_samples.iter().map(score).collect();
        let test_error = if fold_errors.is_empty() {
            f64::INFINITY
        } else {
            fold_errors.iter().sum::<f64>() / fold_errors.len() as f64
        };
        Ok(format!(
            "{{\"event\":\"eval\",\"metric\":{},\"expression\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{}}}",
            jstr(self.cfg.metric.name()),
            jstr(&shown),
            complexity,
            k,
            jnum(train_error),
            jnum(test_error)
        ))
    }
}

struct Row {
    rank: usize,
    complexity: f64,
    k_constants: usize,
    train_error: f64,
    test_error: f64,
    aicc: f64,
    bic: f64,
    dw: f64,
    resid_tests: diagnostics::ResidualTests,
    bootstrap: Option<diagnostics::BootstrapError>,
    overfit: f64,
    fold_errors: Vec<f64>,
    constant_errors: Option<fit::ConstantErrors>,
    shape: String,
    stability: Option<f64>,
    expression: String,
}

struct CvReport {
    mode: &'static str,
    k: usize,
    folds: Vec<CvFoldRow>,
}

struct CvFoldRow {
    train_error: f64,
    test_error: f64,
    complexity: f64,
    expression: String,
}

impl CvReport {
    fn mean_train(&self) -> f64 {
        mean_of(self.folds.iter().map(|r| r.train_error))
    }
    fn mean_test(&self) -> f64 {
        mean_of(self.folds.iter().map(|r| r.test_error))
    }
    fn std_test(&self) -> f64 {
        let n = self.folds.len();
        if n == 0 {
            return 0.0;
        }
        let m = self.mean_test();
        (self.folds.iter().map(|r| (r.test_error - m).powi(2)).sum::<f64>() / n as f64).sqrt()
    }
}

fn mean_of(it: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = it.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn bootstrap_json(b: &Option<diagnostics::BootstrapError>) -> String {
    match b {
        Some(b) => format!(
            "{{\"resamples\":{},\"used\":{},\"apparent\":{},\"oob\":{},\"no_information\":{},\"relative_overfitting\":{},\"weight\":{},\"estimate\":{}}}",
            b.resamples,
            b.used,
            jnum(b.apparent),
            jnum(b.oob),
            jnum(b.no_information),
            jnum(b.relative_overfitting),
            jnum(b.weight),
            jnum(b.estimate)
        ),
        None => "null".to_string(),
    }
}

fn constant_errors_json(ce: &Option<fit::ConstantErrors>) -> String {
    match ce {
        Some(ce) => {
            let items: Vec<String> = ce
                .values
                .iter()
                .zip(&ce.std_errors)
                .zip(&ce.relative)
                .map(|((v, se), rel)| {
                    format!(
                        "{{\"value\":{},\"std_error\":{},\"relative\":{}}}",
                        jnum(*v),
                        jnum(*se),
                        jnum(*rel)
                    )
                })
                .collect();
            format!(
                "{{\"dof\":{},\"ill_conditioned\":{},\"constants\":[{}]}}",
                ce.dof,
                ce.ill_conditioned,
                items.join(",")
            )
        }
        None => "null".to_string(),
    }
}
