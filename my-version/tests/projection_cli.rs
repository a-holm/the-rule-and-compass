//! #294: the CLI records the resolved projection and the unprojected escape
//! hatch produces the same search result on the same bytes.

use std::process::Command;

use rcsr::data;
use rcsr::store;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-projection-cli-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn wide_store(dir: &std::path::Path) -> String {
    // 32 columns: ts, symbol, target y and 29 numeric variables. Only 20 are
    // requested; the projection must not read the other nine.
    let mut text = String::from("ts,symbol,y");
    for c in 0..29 {
        text.push_str(&format!(",x{}", c));
    }
    text.push('\n');
    for r in 0..512 {
        let sym = if r % 2 == 0 { "AAA" } else { "BBB" };
        let ts = 1_700_000_000i64 + (r as i64 / 2) * 3600;
        let y = (r as f64 * 0.07).sin() + if r % 2 == 0 { 0.5 } else { -0.5 };
        text.push_str(&format!("{},{},{}", ts, sym, y));
        for c in 0..29 {
            text.push_str(&format!(",{}", (c + 1) as f64 * 0.1 + (r % 31) as f64 * 0.01));
        }
        text.push('\n');
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let path = dir.join("wide.rcsb").to_string_lossy().into_owned();
    store::write_store_with_time(&path, &table, 128, Some(0)).unwrap();
    path
}

fn run(path: &str, manifest: &str, no_project: bool) -> String {
    let vars: Vec<String> = (0..20).map(|c| format!("x{}", c)).collect();
    let mut args = vec![
        "--data".to_string(),
        path.to_string(),
        "--target".to_string(),
        "y".to_string(),
        "--vars".to_string(),
        vars.join(","),
        "--time".to_string(),
        "ts".to_string(),
        "--group-by".to_string(),
        "symbol".to_string(),
        "--generations".to_string(),
        "2".to_string(),
        "--pop".to_string(),
        "12".to_string(),
        "--seed".to_string(),
        "7".to_string(),
        "--cpu".to_string(),
        "--json".to_string(),
        "--run-manifest".to_string(),
        manifest.to_string(),
    ];
    if no_project {
        args.push("--no-project".to_string());
    }
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(&args)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn the_run_manifest_records_the_projection_and_the_escape_hatch_matches() {
    let dir = tmp_dir("manifest");
    let path = wide_store(&dir);
    let m1 = dir.join("projected.json").to_string_lossy().into_owned();
    let m2 = dir.join("full.json").to_string_lossy().into_owned();
    let a = run(&path, &m1, false);
    let b = run(&path, &m2, true);
    assert!(
        a.contains("\"full_rows\":true"),
        "the projected path still loads every projected row: {a}"
    );

    let projected = std::fs::read_to_string(&m1).unwrap();
    let full = std::fs::read_to_string(&m2).unwrap();
    let p: serde_lite::Value = serde_lite::parse(&projected).unwrap();
    let f: serde_lite::Value = serde_lite::parse(&full).unwrap();
    let proj = p.get("projection").expect("projection object");
    assert_eq!(proj.get("hash").unwrap().as_str().unwrap().len(), 16);
    let cols: Vec<&str> = proj
        .get("columns")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(cols.contains(&"ts"), "{cols:?}");
    assert!(cols.contains(&"symbol"), "{cols:?}");
    assert!(cols.contains(&"y"), "{cols:?}");
    for c in 0..20 {
        assert!(cols.contains(&format!("x{}", c).as_str()), "missing x{c}: {cols:?}");
    }
    assert!(!cols.contains(&"x20"), "unrequested column read: {cols:?}");
    assert_eq!(cols.len(), 23, "ts + symbol + y + 20 vars: {cols:?}");
    assert!(
        f.get("projection").map(|v| v.is_null()).unwrap_or(true),
        "--no-project must record null"
    );

    // The two runs read the same values, so the search result is identical.
    let pa = a.find("\"pareto\":").map(|i| &a[i..]).unwrap_or("");
    let pb = b.find("\"pareto\":").map(|i| &b[i..]).unwrap_or("");
    assert!(pa.starts_with("\"pareto\":"), "missing pareto in projected JSON");
    assert_eq!(pa, pb, "projection changed the search result");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A minimal JSON reader for the manifest assertions; the crate has no
/// dependencies by design, and the test only needs objects/arrays/strings.
mod serde_lite {
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    pub enum Value {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Value>),
        Obj(Vec<(String, Value)>),
    }

    impl Value {
        pub fn get(&self, key: &str) -> Option<&Value> {
            match self {
                Value::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        pub fn is_null(&self) -> bool {
            matches!(self, Value::Null)
        }
        pub fn as_str(&self) -> Option<&str> {
            match self {
                Value::Str(s) => Some(s),
                _ => None,
            }
        }
        pub fn as_array(&self) -> Option<&[Value]> {
            match self {
                Value::Arr(a) => Some(a),
                _ => None,
            }
        }
    }

    pub fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser {
            b: text.as_bytes(),
            i: 0,
        };
        let v = p.value()?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("trailing JSON at {}", p.i));
        }
        Ok(v)
    }

    struct Parser<'a> {
        b: &'a [u8],
        i: usize,
    }

    impl Parser<'_> {
        fn ws(&mut self) {
            while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
                self.i += 1;
            }
        }
        fn value(&mut self) -> Result<Value, String> {
            self.ws();
            match self.b.get(self.i).copied() {
                Some(b'{') => self.object(),
                Some(b'[') => self.array(),
                Some(b'"') => Ok(Value::Str(self.string()?)),
                Some(b't') => {
                    self.i += 4;
                    Ok(Value::Bool(true))
                }
                Some(b'f') => {
                    self.i += 5;
                    Ok(Value::Bool(false))
                }
                Some(b'n') => {
                    self.i += 4;
                    Ok(Value::Null)
                }
                Some(_) => self.number(),
                None => Err("unexpected end".to_string()),
            }
        }
        fn object(&mut self) -> Result<Value, String> {
            self.i += 1;
            let mut kv = Vec::new();
            loop {
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Obj(kv));
                }
                let k = self.string()?;
                self.ws();
                if self.b.get(self.i) != Some(&b':') {
                    return Err("expected ':'".to_string());
                }
                self.i += 1;
                kv.push((k, self.value()?));
                self.ws();
                match self.b.get(self.i) {
                    Some(b',') => self.i += 1,
                    Some(b'}') => {
                        self.i += 1;
                        return Ok(Value::Obj(kv));
                    }
                    _ => return Err("expected ',' or '}'".to_string()),
                }
            }
        }
        fn array(&mut self) -> Result<Value, String> {
            self.i += 1;
            let mut a = Vec::new();
            loop {
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Arr(a));
                }
                a.push(self.value()?);
                self.ws();
                match self.b.get(self.i) {
                    Some(b',') => self.i += 1,
                    Some(b']') => {
                        self.i += 1;
                        return Ok(Value::Arr(a));
                    }
                    _ => return Err("expected ',' or ']'".to_string()),
                }
            }
        }
        fn string(&mut self) -> Result<String, String> {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err("expected string".to_string());
            }
            self.i += 1;
            let mut s = String::new();
            while let Some(&c) = self.b.get(self.i) {
                self.i += 1;
                match c {
                    b'"' => return Ok(s),
                    b'\\' => {
                        let e = *self.b.get(self.i).ok_or("bad escape")?;
                        self.i += 1;
                        s.push(match e {
                            b'n' => '\n',
                            b'r' => '\r',
                            b't' => '\t',
                            b'"' => '"',
                            b'\\' => '\\',
                            b'/' => '/',
                            b'u' => {
                                let hex = std::str::from_utf8(
                                    &self.b[self.i..self.i + 4],
                                )
                                .map_err(|e| e.to_string())?;
                                self.i += 4;
                                char::from_u32(u32::from_str_radix(hex, 16).unwrap())
                                    .unwrap_or('\u{fffd}')
                            }
                            other => other as char,
                        });
                    }
                    _ => s.push(c as char),
                }
            }
            Err("unterminated string".to_string())
        }
        fn number(&mut self) -> Result<Value, String> {
            let start = self.i;
            while let Some(&c) = self.b.get(self.i) {
                if (c as char).is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.i += 1;
                } else {
                    break;
                }
            }
            std::str::from_utf8(&self.b[start..self.i])
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .map(Value::Num)
                .ok_or_else(|| "bad number".to_string())
        }
    }
}
