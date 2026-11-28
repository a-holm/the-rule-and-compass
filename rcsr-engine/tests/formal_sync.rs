//! The Lean models under `formal/` stay tied to the Rust they model.
//!
//! `formal/models.toml` lists, for each model, the source spans it was written from, with the
//! SHA-256 of each span. A changed span fails here, in the pull request that changes it, without
//! a Lean toolchain.
//! The committed golden vectors are compared with what the Rust functions return today.
//! Rules: `docs/formal-verification.md`.

use std::path::{Path, PathBuf};

use rcsr::formal_vectors;
use rcsr::manifest::sha256_hex;

#[derive(Debug, Default, Clone, PartialEq)]
struct Span {
    role: String,
    file: String,
    anchor: String,
    close: Option<String>,
    sha256: String,
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Model {
    id: String,
    project: String,
    lean: String,
    vectors: Option<String>,
    spans: Vec<Span>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// `key = "value"`: no escapes, so a backslash or an inner quote is an error.
fn string_value(line: &str) -> Result<(&str, &str), String> {
    let (key, rest) = line.split_once(" = ").ok_or_else(|| format!("not `key = \"value\"`: {line}"))?;
    let value = rest
        .strip_prefix('"')
        .and_then(|r| r.strip_suffix('"'))
        .ok_or_else(|| format!("value is not a quoted string: {line}"))?;
    if value.contains('"') || value.contains('\\') {
        return Err(format!("quote or backslash inside a value: {line}"));
    }
    Ok((key, value))
}

fn parse_manifest(text: &str) -> Result<Vec<Model>, String> {
    let mut models: Vec<Model> = Vec::new();
    let mut in_span = false;
    let mut saw_version = false;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !saw_version {
            if line != "version = 1" {
                return Err(format!("first entry must be `version = 1`, found: {line}"));
            }
            saw_version = true;
            continue;
        }
        if line == "[[model]]" {
            models.push(Model::default());
            in_span = false;
            continue;
        }
        if line == "[[model.span]]" {
            let m = models.last_mut().ok_or("[[model.span]] before any [[model]]")?;
            m.spans.push(Span::default());
            in_span = true;
            continue;
        }
        let (key, value) = string_value(line)?;
        let m = models.last_mut().ok_or_else(|| format!("entry before any [[model]]: {line}"))?;
        let v = value.to_string();
        if in_span {
            let s = m.spans.last_mut().expect("in_span implies a span");
            match key {
                "role" => s.role = v,
                "file" => s.file = v,
                "anchor" => s.anchor = v,
                "close" => s.close = Some(v),
                "sha256" => s.sha256 = v,
                _ => return Err(format!("unknown span key: {line}")),
            }
        } else {
            match key {
                "id" => m.id = v,
                "project" => m.project = v,
                "lean" => m.lean = v,
                "vectors" => m.vectors = Some(v),
                _ => return Err(format!("unknown model key: {line}")),
            }
        }
    }
    if !saw_version {
        return Err("empty manifest".into());
    }
    for m in &models {
        if m.id.is_empty() || m.project.is_empty() || m.lean.is_empty() || m.spans.is_empty() {
            return Err(format!("model `{}` lacks id, project, lean or a span", m.id));
        }
        for s in &m.spans {
            if !matches!(s.role.as_str(), "modelled" | "mirror" | "assumption") {
                return Err(format!("model {}: unknown role `{}`", m.id, s.role));
            }
            if s.file.is_empty() || s.anchor.is_empty() || s.sha256.len() != 64 {
                return Err(format!("model {}: span lacks file, anchor or a 64-digit sha256", m.id));
            }
        }
    }
    Ok(models)
}

/// The normalised text of a span: from the one line that equals `anchor` after trimming to the
/// first later line that is the anchor line's indentation followed by `close` (the anchor line
/// alone without `close`). Trailing white space and the anchor line's indentation are removed
/// from every line; lines are joined by `\n` with a final `\n`.
fn span_text(source: &str, anchor: &str, close: Option<&str>) -> Result<String, String> {
    let lines: Vec<&str> = source.lines().collect();
    let hits: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].trim() == anchor).collect();
    if hits.len() != 1 {
        return Err(format!("anchor matches {} lines, need exactly 1", hits.len()));
    }
    let start = hits[0];
    let indent = &lines[start][..lines[start].len() - lines[start].trim_start().len()];
    let end = match close {
        None => start,
        Some(c) => {
            let want = format!("{indent}{c}");
            (start + 1..lines.len())
                .find(|&i| lines[i].trim_end() == want)
                .ok_or_else(|| format!("no closing line `{want}` after the anchor"))?
        }
    };
    let mut out = String::new();
    for line in &lines[start..=end] {
        let line = line.trim_end();
        out.push_str(line.strip_prefix(indent).unwrap_or(line.trim_start()));
        out.push('\n');
    }
    Ok(out)
}

fn load_manifest() -> Vec<Model> {
    let path = root().join("formal/models.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    parse_manifest(&text).unwrap_or_else(|e| panic!("formal/models.toml: {e}"))
}

#[test]
fn manifest_spans_match_sources() {
    let mut stale = Vec::new();
    for m in load_manifest() {
        for s in &m.spans {
            let path = root().join(&s.file);
            let source = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    stale.push(format!("model {}: cannot read {}: {e}", m.id, s.file));
                    continue;
                }
            };
            match span_text(&source, &s.anchor, s.close.as_deref()) {
                Err(e) => stale.push(format!("model {}: {} span \"{}\": {e}", m.id, s.file, s.anchor)),
                Ok(text) => {
                    let now = sha256_hex(text.as_bytes());
                    if now != s.sha256 {
                        stale.push(format!(
                            "model {} is stale: {} span \"{}\" has sha256 {now}, manifest has {}. Review {}, then follow docs/formal-verification.md \"When a modelled function changes\".",
                            m.id, s.file, s.anchor, s.sha256, m.lean
                        ));
                    }
                }
            }
        }
    }
    assert!(stale.is_empty(), "\n{}", stale.join("\n"));
}

#[test]
fn manifest_lean_files_exist_and_ids_are_unique() {
    let models = load_manifest();
    assert!(!models.is_empty(), "formal/models.toml lists no model");
    let mut ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    ids.sort_unstable();
    let n = ids.len();
    ids.dedup();
    assert_eq!(ids.len(), n, "duplicate model id in formal/models.toml");
    for m in &models {
        assert!(root().join(&m.lean).is_file(), "model {}: {} is missing", m.id, m.lean);
    }
}

#[test]
fn span_text_strips_indentation_and_trailing_space() {
    let a = "fn f() {\n    1\n}\n";
    let b = "mod m {\n    fn f() {   \n        1\t\n    }\n}\n";
    let ta = span_text(a, "fn f() {", Some("}")).unwrap();
    assert_eq!(ta, "fn f() {\n    1\n}\n");
    assert_eq!(span_text(b, "fn f() {", Some("}")).unwrap(), ta);
    assert_eq!(span_text("const A: u8 = 1;\nconst B: u8 = 2;\n", "const B: u8 = 2;", None).unwrap(), "const B: u8 = 2;\n");
}

#[test]
fn span_text_sees_a_one_character_change() {
    let a = span_text("fn f() {\n    1\n}\n", "fn f() {", Some("}")).unwrap();
    let b = span_text("fn f() {\n    2\n}\n", "fn f() {", Some("}")).unwrap();
    assert_ne!(sha256_hex(a.as_bytes()), sha256_hex(b.as_bytes()));
}

#[test]
fn span_text_refuses_a_missing_or_repeated_anchor_and_a_missing_close() {
    assert!(span_text("fn g() {\n}\n", "fn f() {", Some("}")).unwrap_err().contains("0 lines"));
    assert!(span_text("fn f() {\n}\nfn f() {\n}\n", "fn f() {", Some("}")).unwrap_err().contains("2 lines"));
    assert!(span_text("fn f() {\n    1\n", "fn f() {", Some("}")).unwrap_err().contains("no closing line"));
}

#[test]
fn parser_refuses_what_it_does_not_know() {
    let good = "version = 1\n\n# note\n[[model]]\nid = \"m\"\nproject = \"P0\"\nlean = \"formal/x.lean\"\n[[model.span]]\nrole = \"modelled\"\nfile = \"a.rs\"\nanchor = \"fn f() {\"\nclose = \"}\"\nsha256 = \"0000000000000000000000000000000000000000000000000000000000000000\"\n";
    let models = parse_manifest(good).unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].spans[0].close.as_deref(), Some("}"));
    assert!(parse_manifest(&good.replace("version = 1", "version = 2")).is_err());
    assert!(parse_manifest(&good.replace("role = \"modelled\"", "role = \"copy\"")).is_err());
    assert!(parse_manifest(&good.replace("project = \"P0\"", "owner = \"x\"")).is_err());
    assert!(parse_manifest(&good.replace("id = \"m\"", "id = m")).is_err());
    assert!(parse_manifest(&good.replace("sha256 = \"0000", "sha256 = \"00")).is_err());
    assert!(parse_manifest(&good.replace("[[model.span]]", "[model.span]")).is_err());
    assert!(parse_manifest("").is_err());
}

#[test]
fn manifest_models_have_golden_vectors() {
    let models = load_manifest();
    let mut named: Vec<String> = models.iter().filter_map(|m| m.vectors.clone()).collect();
    named.sort();
    let mut generated: Vec<String> = formal_vectors::MODELS.iter().map(|s| s.to_string()).collect();
    generated.sort();
    assert_eq!(named, generated, "the `vectors` names of formal/models.toml and formal_vectors::MODELS differ");
    for name in &named {
        let path = root().join(format!("formal/vectors/golden/{name}.vec"));
        assert!(path.is_file(), "{} is missing", path.display());
    }
}

#[test]
fn golden_vectors_match_rust() {
    for model in formal_vectors::MODELS {
        let path = root().join(format!("formal/vectors/golden/{model}.vec"));
        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let now = formal_vectors::golden(model).expect("a generator for every name in MODELS");
        if committed != now {
            let line = committed.lines().zip(now.lines()).position(|(a, b)| a != b).map_or(0, |i| i + 1);
            panic!(
                "golden vectors of {model} differ from the Rust function (first differing line {line}; committed has {} lines, Rust gives {}). The function changed behaviour: follow docs/formal-verification.md \"When a modelled function changes\".",
                committed.lines().count(),
                now.lines().count()
            );
        }
    }
}

#[test]
fn golden_files_have_the_declared_shape() {
    for model in formal_vectors::MODELS {
        let text = formal_vectors::golden(model).expect("a generator for every name in MODELS");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), formal_vectors::GOLDEN_COUNT + 2, "{model}");
        assert_eq!(lines[0], format!("rcsr-vec 1 {model} sig=w,w->w"));
        assert_eq!(lines[lines.len() - 1], format!("end {}", formal_vectors::GOLDEN_COUNT));
        assert!(formal_vectors::directed_len(model).expect("known model") <= formal_vectors::GOLDEN_COUNT);
        assert!(formal_vectors::vectors(model, 0, 1).is_none(), "a count below the directed set is refused");
    }
    assert!(formal_vectors::vectors("no_such_model", 5000, 1).is_none());
}
