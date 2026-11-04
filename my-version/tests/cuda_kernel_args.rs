//! Guard: no CUDA kernel receives host memory. Text-level lint over
//! `src/cuda/*.cu` and `*.h`; see `docs/gpu-backend.md`, "Kernel arguments are
//! device memory". It reads source files only and needs no GPU or CUDA build.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// L2 exceptions: `(function, parameter or "*" for all, reason)`. Every entry
/// must match at least one launch, otherwise `every_allowlist_entry_is_used`
/// fails.
const ALLOW_L2: &[(&str, &str, &str)] = &[
    (
        "rc_ring_launch_feature",
        "out",
        "device column from rc_enrich_new_output (src/enrich_gpu.rs)",
    ),
    ("rc_tail_donchian", "*", TAIL_REASON),
    ("rc_tail_stoch", "*", TAIL_REASON),
    ("rc_tail_atr", "*", TAIL_REASON),
    ("rc_tail_batr", "*", TAIL_REASON),
    ("rc_tail_gapatr", "*", TAIL_REASON),
    ("rc_tail_volrat", "*", TAIL_REASON),
    ("rc_tail_adx", "*", TAIL_REASON),
    ("rc_tail_bb_z", "*", TAIL_REASON),
    ("rc_tail_ma_slope", "*", TAIL_REASON),
    ("rc_tail_tod", "*", TAIL_REASON),
    ("rc_tail_exotic", "*", TAIL_REASON),
];

/// The only file that may contain kernel launch syntax.
const GUARD_HEADER: &str = "launch_guard.h";

const TAIL_REASON: &str = "called only from enrich.cu with c->d_* fields and rc_checked_malloc scratch; no Rust declaration exists";

#[derive(Debug, Clone, PartialEq)]
struct Finding {
    rule: &'static str,
    file: String,
    line: usize,
    func: String,
    what: String,
}

struct Report {
    findings: Vec<Finding>,
    used_allow: BTreeSet<usize>,
}

/// Blanks comments, string and char literals; keeps length and newlines.
fn clean(src: &str) -> Vec<u8> {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    while i < b.len() {
        let blank_to = |out: &mut Vec<u8>, from: usize, to: usize| {
            for x in &mut out[from..to] {
                if *x != b'\n' {
                    *x = b' ';
                }
            }
        };
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            let mut j = i;
            while j < b.len() && b[j] != b'\n' {
                j += 1;
            }
            blank_to(&mut out, i, j);
            i = j;
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let mut j = i + 2;
            while j + 1 < b.len() && !(b[j] == b'*' && b[j + 1] == b'/') {
                j += 1;
            }
            let end = (j + 2).min(b.len());
            blank_to(&mut out, i, end);
            i = end;
        } else if b[i] == b'"' || b[i] == b'\'' {
            let q = b[i];
            let mut j = i + 1;
            while j < b.len() && b[j] != q && b[j] != b'\n' {
                if b[j] == b'\\' {
                    j += 1;
                }
                j += 1;
            }
            let end = (j + 1).min(b.len());
            blank_to(&mut out, i + 1, end.saturating_sub(1).max(i + 1));
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// Blanks preprocessor lines (with continuations); keeps length and newlines.
fn blank_preprocessor(text: &[u8]) -> Vec<u8> {
    let mut out = text.to_vec();
    let mut i = 0;
    while i < text.len() {
        let mut ls = i;
        while ls < text.len() && (text[ls] == b' ' || text[ls] == b'\t') {
            ls += 1;
        }
        let is_pp = ls < text.len() && text[ls] == b'#';
        let mut j = i;
        loop {
            while j < text.len() && text[j] != b'\n' {
                j += 1;
            }
            let continued = j > 0 && text[j - 1] == b'\\';
            if is_pp {
                for x in &mut out[i..j] {
                    *x = b' ';
                }
            }
            if !(is_pp && continued) || j >= text.len() {
                break;
            }
            j += 1;
            i = j;
        }
        i = j + 1;
    }
    out
}

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

fn match_close(t: &[u8], open: usize, o: u8, c: u8) -> Option<usize> {
    let mut depth = 0usize;
    for (k, &x) in t.iter().enumerate().skip(open) {
        if x == o {
            depth += 1;
        } else if x == c {
            depth -= 1;
            if depth == 0 {
                return Some(k);
            }
        }
    }
    None
}

fn split_top(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if ch == ',' && depth == 0 {
            parts.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_string());
    }
    parts
}

fn idents(s: &str) -> Vec<&str> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty() && !w.as_bytes()[0].is_ascii_digit())
        .collect()
}

struct Func {
    name: String,
    header: String,
    params: Vec<String>,
    body: (usize, usize),
}

const KEYWORDS: &[&str] = &["if", "for", "while", "switch", "catch", "return", "sizeof"];

fn find_functions(t: &[u8]) -> Vec<Func> {
    let mut funcs = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if t[i] != b'{' {
            i += 1;
            continue;
        }
        let mut p = i;
        while p > 0 && t[p - 1].is_ascii_whitespace() {
            p -= 1;
        }
        // `const`, `noexcept` and similar suffixes between `)` and `{`.
        let mut q = p;
        loop {
            let mut s = q;
            while s > 0 && is_ident(t[s - 1]) {
                s -= 1;
            }
            let w = &t[s..q];
            if !w.is_empty() && (w == b"const" || w == b"noexcept") {
                q = s;
                while q > 0 && t[q - 1].is_ascii_whitespace() {
                    q -= 1;
                }
            } else {
                break;
            }
        }
        if q == 0 || t[q - 1] != b')' {
            i += 1;
            continue;
        }
        let close = q - 1;
        let mut depth = 0i32;
        let mut open = None;
        for k in (0..=close).rev() {
            if t[k] == b')' {
                depth += 1;
            } else if t[k] == b'(' {
                depth -= 1;
                if depth == 0 {
                    open = Some(k);
                    break;
                }
            }
        }
        let Some(open) = open else {
            i += 1;
            continue;
        };
        let mut ne = open;
        while ne > 0 && t[ne - 1].is_ascii_whitespace() {
            ne -= 1;
        }
        let mut ns = ne;
        while ns > 0 && is_ident(t[ns - 1]) {
            ns -= 1;
        }
        let name = String::from_utf8_lossy(&t[ns..ne]).to_string();
        let end = match_close(t, i, b'{', b'}').unwrap_or(t.len() - 1);
        if name.is_empty() || KEYWORDS.contains(&name.as_str()) {
            i += 1;
            continue;
        }
        let mut hs = ns;
        while hs > 0 && !matches!(t[hs - 1], b';' | b'{' | b'}') {
            hs -= 1;
        }
        let header = String::from_utf8_lossy(&t[hs..open]).to_string();
        let params = split_top(&String::from_utf8_lossy(&t[open + 1..close]));
        funcs.push(Func { name, header, params, body: (i, end) });
        i = end + 1;
    }
    funcs
}

struct Launch {
    pos: usize,
    args: Vec<String>,
}

fn find_launches(t: &[u8]) -> Vec<Launch> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= t.len() {
        if &t[i..i + 3] == b"<<<" {
            let mut j = i + 3;
            while j + 3 <= t.len() && &t[j..j + 3] != b">>>" {
                j += 1;
            }
            j += 3;
            while j < t.len() && t[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < t.len() && t[j] == b'(' {
                if let Some(c) = match_close(t, j, b'(', b')') {
                    let args = split_top(&String::from_utf8_lossy(&t[j + 1..c]));
                    out.push(Launch { pos: i, args });
                    i = c;
                }
            }
        }
        i += 1;
    }
    out.extend(find_rc_launches(t));
    out
}

/// `rc_launch(name, kernel, grid, block, smem, stream, args...)` calls; the
/// kernel argument may carry template arguments with commas.
fn find_rc_launches(t: &[u8]) -> Vec<Launch> {
    let mut out = Vec::new();
    let key = b"rc_launch";
    let mut i = 0;
    while i + key.len() < t.len() {
        let hit = &t[i..i + key.len()] == key
            && (i == 0 || !is_ident(t[i - 1]))
            && t[i + key.len()..].iter().find(|c| !c.is_ascii_whitespace()) == Some(&b'(');
        if !hit {
            i += 1;
            continue;
        }
        let open = i + key.len() + t[i + key.len()..].iter().position(|&c| c == b'(').unwrap();
        let Some(close) = match_close(t, open, b'(', b')') else {
            i += 1;
            continue;
        };
        let inner = String::from_utf8_lossy(&t[open + 1..close]).to_string();
        let Some(comma) = inner.find(',') else {
            i = close;
            continue;
        };
        let rest = &inner[comma + 1..];
        let start = rest.len() - rest.trim_start().len();
        let kb = rest.as_bytes();
        let mut k = start;
        while k < kb.len() && is_ident(kb[k]) {
            k += 1;
        }
        let mut e = k;
        while e < kb.len() && kb[e].is_ascii_whitespace() {
            e += 1;
        }
        if e < kb.len() && kb[e] == b'<' {
            k = match_close(kb, e, b'<', b'>').map_or(kb.len(), |c| c + 1);
        }
        let tail = rest[k..].trim_start().strip_prefix(',').unwrap_or("");
        let args = split_top(tail).into_iter().skip(4).collect();
        out.push(Launch { pos: i, args });
        i = close;
    }
    out
}

fn line_of(t: &[u8], pos: usize) -> usize {
    1 + t[..pos.min(t.len())].iter().filter(|&&c| c == b'\n').count()
}

/// Name and element type of a pointer parameter, or None for non-pointers.
fn pointer_param(p: &str) -> Option<(String, String)> {
    if !p.contains('*') {
        return None;
    }
    let words = idents(p);
    let name = words.last()?.to_string();
    let ty = words
        .iter()
        .rev()
        .skip(1)
        .find(|w| !matches!(**w, "const" | "__restrict__" | "volatile" | "struct" | "unsigned"))
        .map(|w| w.to_string())
        .unwrap_or_default();
    Some((name, ty))
}

/// First identifier of the argument, when the argument is that pointer bare or
/// offset (`p`, `p + k`, `(T *)p`).
fn arg_passes(arg: &str, name: &str) -> bool {
    let mut s = arg.trim();
    loop {
        s = s.trim_start_matches(['&', ' ']);
        if let Some(rest) = s.strip_prefix('(') {
            if let Some(end) = rest.find(')') {
                let inner = &rest[..end];
                if inner.contains('*') && !inner.contains(name) {
                    s = &rest[end + 1..];
                    continue;
                }
            }
        }
        break;
    }
    let s = s.trim_start_matches('(').trim_start();
    let Some(rest) = s.strip_prefix(name) else {
        return false;
    };
    if rest.as_bytes().first().is_some_and(|&c| is_ident(c)) {
        return false;
    }
    let rest = rest.trim_start();
    rest.is_empty() || rest.starts_with('+') || rest.starts_with(')')
}

fn assigned_alloc_locals(body: &str) -> Vec<String> {
    let mut names = Vec::new();
    let b = body.as_bytes();
    for kw in ["malloc", "calloc", "realloc"] {
        let mut from = 0;
        while let Some(off) = body[from..].find(kw) {
            let at = from + off;
            from = at + kw.len();
            if at > 0 && is_ident(b[at - 1]) {
                continue;
            }
            if b.get(from).is_some_and(|&c| is_ident(c)) {
                continue;
            }
            let head = body[..at].trim_end();
            let head = strip_cast(head);
            if let Some(h) = head.strip_suffix('=') {
                if let Some(n) = idents(h).last() {
                    names.push(n.to_string());
                }
            }
        }
    }
    let mut from = 0;
    while let Some(off) = body[from..].find("new") {
        let at = from + off;
        from = at + 3;
        if (at > 0 && is_ident(b[at - 1])) || b.get(from).is_some_and(|&c| is_ident(c)) {
            continue;
        }
        if let Some(h) = body[..at].trim_end().strip_suffix('=') {
            if let Some(n) = idents(h).last() {
                names.push(n.to_string());
            }
        }
    }
    names
}

fn strip_cast(head: &str) -> &str {
    let h = head.trim_end();
    if h.ends_with(')') {
        let mut depth = 0i32;
        for (k, c) in h.char_indices().rev() {
            match c {
                ')' => depth += 1,
                '(' => {
                    depth -= 1;
                    if depth == 0 {
                        return h[..k].trim_end();
                    }
                }
                _ => {}
            }
        }
    }
    h
}

fn host_staging_ident(arg: &str) -> Option<String> {
    if arg.contains("->host") {
        return Some("->host".to_string());
    }
    idents(arg)
        .into_iter()
        .find(|w| w.starts_with("h_") || *w == "fs_host" || *w == "host")
        .map(|w| w.to_string())
}

fn lint_sources(files: &[(String, String)], allow: &[(&str, &str, &str)]) -> Report {
    let cleaned: Vec<(String, Vec<u8>)> =
        files.iter().map(|(n, s)| (n.clone(), clean(s))).collect();

    let mut ctx_types = BTreeSet::new();
    for (_, t) in &cleaned {
        let text = String::from_utf8_lossy(t).to_string();
        let mut from = 0;
        while let Some(off) = text[from..].find("calloc") {
            let at = from + off;
            from = at + 6;
            let head = text[..at].trim_end();
            let Some(head) = head.strip_suffix(')') else { continue };
            let Some(cast) = head.rfind('(') else { continue };
            let cast = &head[cast + 1..];
            let Some(ty) = cast.strip_suffix('*').map(str::trim) else { continue };
            if !ty.is_empty() && ty.bytes().all(is_ident) {
                let tail = &text[from..];
                let sizeof = format!("sizeof({ty})");
                let stmt_end = tail.find(';').unwrap_or(tail.len());
                if tail[..stmt_end].replace(' ', "").contains(&sizeof) {
                    ctx_types.insert(ty.to_string());
                }
            }
        }
    }

    let mut findings = Vec::new();
    let mut used_allow = BTreeSet::new();
    for (file, t) in &cleaned {
        if file == GUARD_HEADER {
            continue;
        }
        let mut at = 0;
        while let Some(off) = t[at..].windows(3).position(|w| w == b"<<<") {
            findings.push(Finding {
                rule: "L5",
                file: file.clone(),
                line: line_of(t, at + off),
                func: String::new(),
                what: format!("raw kernel launch syntax; use rc_launch ({GUARD_HEADER})"),
            });
            at += off + 3;
        }
        let flat = blank_preprocessor(t);
        let funcs = find_functions(&flat);
        let launches = find_launches(t);
        for l in &launches {
            if !funcs.iter().any(|f| f.body.0 < l.pos && l.pos < f.body.1) {
                findings.push(Finding {
                    rule: "parse",
                    file: file.clone(),
                    line: line_of(t, l.pos),
                    func: String::new(),
                    what: "launch outside any function body".to_string(),
                });
            }
        }
        for f in &funcs {
            let device_side = f.header.contains("__global__") || f.header.contains("__device__");
            let fline = line_of(t, f.body.0);
            if device_side {
                for p in &f.params {
                    if let Some((pname, ty)) = pointer_param(p) {
                        if ctx_types.contains(&ty) {
                            findings.push(Finding {
                                rule: "L1",
                                file: file.clone(),
                                line: fline,
                                func: f.name.clone(),
                                what: format!("device function takes host context `{ty} *{pname}`"),
                            });
                        }
                    }
                }
            }
            let exported = !device_side
                && !f.header.split_whitespace().any(|w| w == "static")
                && f.name.starts_with("rc_");
            let body = String::from_utf8_lossy(&flat[f.body.0..f.body.1]).to_string();
            let locals = assigned_alloc_locals(&body);
            for l in launches.iter().filter(|l| f.body.0 < l.pos && l.pos < f.body.1) {
                let line = line_of(t, l.pos);
                if exported {
                    for p in &f.params {
                        let Some((pname, _)) = pointer_param(p) else { continue };
                        if !l.args.iter().any(|a| arg_passes(a, &pname)) {
                            continue;
                        }
                        let hit = allow
                            .iter()
                            .position(|(fun, par, _)| *fun == f.name && (*par == "*" || *par == pname));
                        if let Some(k) = hit {
                            used_allow.insert(k);
                        } else {
                            findings.push(Finding {
                                rule: "L2",
                                file: file.clone(),
                                line,
                                func: f.name.clone(),
                                what: format!("pointer parameter `{pname}` is a launch argument"),
                            });
                        }
                    }
                }
                for name in &locals {
                    if l.args.iter().any(|a| arg_passes(a, name)) {
                        findings.push(Finding {
                            rule: "L3",
                            file: file.clone(),
                            line,
                            func: f.name.clone(),
                            what: format!("host allocation `{name}` is a launch argument"),
                        });
                    }
                }
                for a in &l.args {
                    if let Some(id) = host_staging_ident(a) {
                        findings.push(Finding {
                            rule: "L4",
                            file: file.clone(),
                            line,
                            func: f.name.clone(),
                            what: format!("launch argument names host staging `{id}`"),
                        });
                    }
                }
            }
        }
    }
    Report { findings, used_allow }
}

fn read_dir_sources(dir: &Path) -> Vec<(String, String)> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "cu" || ext == "h" {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            files.push((name, fs::read_to_string(&path).unwrap()));
        }
    }
    files.sort();
    assert!(!files.is_empty(), "no .cu or .h files in {}", dir.display());
    files
}

fn cuda_dir() -> String {
    std::env::var("RCSR_LINT_CUDA_DIR")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/src/cuda").to_string())
}

fn render(found: &[Finding]) -> String {
    let mut s = String::new();
    for f in found {
        let _ = writeln!(s, "{} {}:{} {}: {}", f.rule, f.file, f.line, f.func, f.what);
    }
    s
}

#[test]
fn kernels_receive_only_device_memory() {
    let files = read_dir_sources(Path::new(&cuda_dir()));
    let report = lint_sources(&files, ALLOW_L2);
    assert!(
        report.findings.is_empty(),
        "host memory reaches a kernel ({} findings):\n{}",
        report.findings.len(),
        render(&report.findings)
    );
}

#[test]
fn every_allowlist_entry_is_used() {
    let files = read_dir_sources(Path::new(&cuda_dir()));
    let report = lint_sources(&files, ALLOW_L2);
    for (k, (fun, par, reason)) in ALLOW_L2.iter().enumerate() {
        assert!(!reason.trim().is_empty(), "allowlist entry {fun}/{par} has no reason");
        assert!(
            report.used_allow.contains(&k),
            "stale allowlist entry {fun}/{par}: no launch matches"
        );
    }
}

fn one(src: &str) -> Vec<Finding> {
    lint_sources(&[("snippet.cu".to_string(), src.to_string())], &[]).findings
}

fn rules(src: &str) -> Vec<&'static str> {
    one(src).iter().map(|f| f.rule).collect()
}

#[test]
fn each_rule_fires_on_its_snippet() {
    let ctx = "struct RcX { int n; double *d_a; };\n\
               RcX *mk(void) { RcX *c = (RcX *)calloc(1, sizeof(RcX)); return c; }\n";

    // L1
    let l1_bad = format!("{ctx}__global__ void k(RcX *c, double *o) {{ o[0] = c->n; }}\n");
    assert_eq!(rules(&l1_bad), ["L1"]);
    let l1_ok = format!("{ctx}__global__ void k(int n, const double *a, double *o) {{ o[0] = n + a[0]; }}\n");
    assert!(rules(&l1_ok).is_empty());

    // L2
    let l2_bad = "extern \"C\" int rc_run(double *out, int n) {\n\
                  rc_launch(\"k\", k, 1, 32, 0, 0, out + 4, n);\n return 0; }\n__global__ void k(double *o, int n) { o[0] = n; }\n";
    assert_eq!(rules(l2_bad), ["L2"]);
    let l2_ok = "extern \"C\" int rc_run(double *out, int n) {\n\
                 double *d = 0; cudaMalloc((void **)&d, 8);\n\
                 rc_launch(\"k\", k, 1, 32, 0, 0, d, n);\n cudaMemcpy(out, d, 8, cudaMemcpyDeviceToHost);\n return 0; }\n\
                 __global__ void k(double *o, int n) { o[0] = n; }\n";
    assert!(rules(l2_ok).is_empty());

    // L3
    let l3_bad = "static void f(int n) {\n double *p = (double *)malloc(8 * n);\n\
                  rc_launch(\"k\", k, 1, 32, 0, 0, p, n);\n}\n";
    assert_eq!(rules(l3_bad), ["L3"]);
    let l3_ok = "static void f(int n) {\n double *p = (double *)rc_checked_malloc(8 * n);\n\
                 rc_launch(\"k\", k, 1, 32, 0, 0, p, n);\n}\n";
    assert!(rules(l3_ok).is_empty());

    // L4
    let l4_bad = "static void f(int n, Ring *r) {\n rc_launch(\"k\", k, 1, 32, 0, 0, r->h_stage, n);\n}\n";
    assert_eq!(rules(l4_bad), ["L4"]);
    let l4_ok = "static void f(int n, Ring *r) {\n rc_launch(\"k\", k, 1, 32, 0, 0, r->d_stage, n);\n}\n";
    assert!(rules(l4_ok).is_empty());

    // L5
    let l5_bad = "static void f(int n) {\n k<<<1, 32>>>(d, n);\n}\n";
    assert_eq!(rules(l5_bad), ["L5"]);
    let l5_ok = "static void f(int n) {\n rc_launch(\"k\", k, 1, 32, 0, 0, d, n);\n}\n";
    assert!(rules(l5_ok).is_empty());
}

#[test]
fn rc_launch_arguments_follow_the_rules() {
    // Template kernel arguments carry commas; the launch arguments start after them.
    let bad = "static void f(int n, Ring *r) {\n\
               rc_launch(\"k<A, B>\", k<A, B>, 1, 32, 0, 0, r->h_stage, n);\n}\n";
    assert_eq!(rules(bad), ["L4"]);
    let ok = "static void f(int n, Ring *r) {\n\
              rc_launch(\"k<A, B>\", k<A, B>, nb, 32, 0, s, r->d_stage, n);\n}\n";
    assert!(rules(ok).is_empty());
    // The grid and block arguments are not launch arguments.
    let cfg = "static void f(Ring *r) {\n rc_launch(\"k\", k, r->h_n, 32, 0, 0, r->d_a);\n}\n";
    assert!(rules(cfg).is_empty());
    // L2 follows pointer parameters through rc_launch.
    let l2 = "extern \"C\" int rc_run(double *out, int n) {\n\
              rc_launch(\"k\", k, 1, 32, 0, 0, out + 4, n);\n return 0; }\n";
    assert_eq!(rules(l2), ["L2"]);
}

#[test]
fn lint_parses_launch_shapes() {
    // Multi-line launch inside a macro argument, comments and strings ignored.
    let src = "#define LAUNCH(c, call) do { call; } while (0)\n\
               static void f(Ring *r) {\n\
               // k<<<1, 1>>>(r->h_x);\n\
               const char *s = \"k<<<1, 1>>>(h_y)\";\n\
               LAUNCH(c, (rc_launch(\"k\", k, 1, 32, 0, 0, r->d_a,\n\
                                       r->h_pinned)));\n}\n";
    let found = one(src);
    assert_eq!(found.len(), 1, "{}", render(&found));
    assert_eq!((found[0].rule, found[0].line), ("L4", 5));

    // A launch outside any function body is reported, not skipped.
    assert_eq!(rules("LAUNCH(rc_launch(\"k\", k, 1, 32, 0, 0, a));\n"), ["parse"]);

    // An allowlist entry silences exactly its parameter and is marked used.
    let src = "extern \"C\" int rc_ok(double *a, double *b) { rc_launch(\"k\", k, 1, 1, 0, 0, a, b); return 0; }\n";
    let r = lint_sources(
        &[("s.cu".to_string(), src.to_string())],
        &[("rc_ok", "a", "test")],
    );
    assert_eq!(r.findings.len(), 1);
    assert!(r.findings[0].what.contains("`b`"));
    assert!(r.used_allow.contains(&0));
}

#[test]
fn arrow_arguments_split_on_commas() {
    let src = "extern \"C\" int rc_run(Ctx *c, double *out) {\n\
               rc_launch(\"k\", k, 1, 32, 0, 0, c->d_a, c->n, out);\n return 0; }\n";
    let found = one(src);
    assert_eq!(found.len(), 1, "{}", render(&found));
    assert!(found[0].what.contains("`out`"));
}
