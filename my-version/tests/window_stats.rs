//! #364: rolling window statistics against a high-precision reference, the
//! locality of each window, and the CPU feature-phase time.
//!
//! The reference recomputes every column from the raw bars with direct
//! double-double window sums and centred variance/covariance. Glue that the
//! engine evaluates in plain f64 (logs, ratios, clips) is repeated verbatim, so
//! the reported error is the error of the window statistics alone.

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options};
use rcsr::store::{self, FileStore};

const STEP: i64 = 900;
const T0: i64 = 1_700_000_000 - (1_700_000_000 % 86_400);

#[derive(Clone, Copy)]
struct Row {
    t: i64,
    sym: usize,
    o: f64,
    h: f64,
    l: f64,
    c: f64,
    v: f64,
    qv: f64,
    trades: f64,
    tb: f64,
    tq: f64,
}

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 32)
}

fn sym_name(k: usize) -> String {
    match k {
        0 => "BTCUSDT".to_string(),
        1 => "ETHUSDT".to_string(),
        _ => format!("SYM{k:02}"),
    }
}

/// Same shape as the #303 parity fixture: holes, a NaN close, zero volume,
/// H==L bars, tied returns, late listings.
fn fixture(symbols: usize, per: usize) -> Vec<Row> {
    let late = 3_000.min(per / 4);
    let mut rows = Vec::with_capacity(symbols * per);
    for k in 0..symbols {
        let mut close = 100.0 + k as f64;
        let start = if k >= 15 { late } else { 0 };
        for i in start..per {
            if i % 997 == 500 && k % 4 == 0 {
                continue;
            }
            if k == 3 && (per / 2..per / 2 + 20).contains(&i) {
                continue;
            }
            if k == 5 && (per * 3 / 4..per * 3 / 4 + 130).contains(&i) {
                continue;
            }
            let t = T0 + i as i64 * STEP;
            let mut ret = if i % 5 == 0 { 0.002 } else { 0.0005 * ((k % 3) as f64 - 1.0) };
            if k % 2 == 0 && k > 1 {
                ret += ((mix(k as u64, i as u64) % 2001) as f64 - 1000.0) / 1.0e6;
            }
            ret += ((mix(k as u64 + 77, i as u64) % 2001) as f64 - 1000.0) / 2.0e5;
            let open = close;
            let mut next = close * (1.0 + ret);
            let mut high = open.max(next) + 0.1;
            let mut low = open.min(next) - 0.1;
            if i % 1013 == 3 {
                next = open;
                high = open;
                low = open;
            }
            let mut v = 10.0 + (i % 97) as f64;
            let mut trades = 10.0 + (i % 53) as f64;
            let mut tb = v * (0.3 + 0.4 * ((i % 7) as f64) / 7.0);
            if i % 701 == 0 && k % 2 == 1 {
                v = 0.0;
                trades = 0.0;
                tb = 0.0;
            }
            let qv = v * next;
            let tq = tb * next;
            let nan_close = k == 9 && i % 1500 == 7;
            rows.push(Row {
                t,
                sym: k,
                o: open,
                h: high,
                l: low,
                c: if nan_close { f64::NAN } else { next },
                v,
                qv,
                trades,
                tb,
                tq,
            });
            if !nan_close {
                close = next;
            }
        }
    }
    rows
}

fn write_store(path: &str, rows: &[Row]) {
    let mut text = String::with_capacity(rows.len() * 120);
    text.push_str("ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote\n");
    for r in rows {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            r.t,
            sym_name(r.sym),
            r.o,
            r.h,
            r.l,
            r.c,
            r.v,
            r.qv,
            r.trades,
            r.tb,
            r.tq
        ));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, 65_536, Some(0)).unwrap();
}

fn work_dir(tag: &str) -> std::path::PathBuf {
    let base = std::env::var("RCSR_PARITY_TMP")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let dir = base.join(format!("rcsr-wstats-{}-{}", tag, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn options(data: &str, out: &str, names: &[String]) -> Options {
    Options {
        data: data.to_string(),
        out: Some(out.to_string()),
        features: vec![names.join(",")],
        allow_lockbox: true,
        gpu: GpuMode::Off,
        chunk_rows: 65_536,
        ..Default::default()
    }
}

fn read_column(path: &str, name: &str) -> Vec<f64> {
    let s = FileStore::open(path).unwrap();
    let c = s.header().column_index(name).unwrap_or_else(|| panic!("column '{name}' missing"));
    let mut out = Vec::with_capacity(s.n_rows());
    for g in 0..s.n_groups() {
        out.extend(s.read_f64_group(g, c).unwrap());
    }
    out
}

fn build(rows: &[Row], names: &[String], tag: &str) -> (String, rcsr::enrich::Report) {
    let dir = work_dir(tag);
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, rows);
    let rep = enrich::build(&options(&src, &out, names)).unwrap();
    (out, rep)
}

// ---------------------------------------------------------------------------
// Double-double reference arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Dd(f64, f64);

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn fast(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

impl Dd {
    fn f(x: f64) -> Dd {
        Dd(x, 0.0)
    }
    fn add(self, o: Dd) -> Dd {
        let (s, e) = two_sum(self.0, o.0);
        let (t, g) = two_sum(self.1, o.1);
        let (s, e) = fast(s, e + t);
        let (h, l) = fast(s, e + g);
        Dd(h, l)
    }
    fn sub(self, o: Dd) -> Dd {
        self.add(Dd(-o.0, -o.1))
    }
    fn mul(self, o: Dd) -> Dd {
        let p = self.0 * o.0;
        let e = self.0.mul_add(o.0, -p) + (self.0 * o.1 + self.1 * o.0);
        let (h, l) = fast(p, e);
        Dd(h, l)
    }
    fn div(self, o: Dd) -> Dd {
        let q1 = self.0 / o.0;
        let r = self.sub(o.mul(Dd::f(q1)));
        let q2 = r.0 / o.0;
        let r = r.sub(o.mul(Dd::f(q2)));
        let q3 = r.0 / o.0;
        let (h, l) = fast(q1, q2);
        Dd(h, l).add(Dd::f(q3))
    }
    fn v(self) -> f64 {
        self.0 + self.1
    }
}

fn contig(t: &[i64], i: usize, w: usize) -> bool {
    i + 1 >= w && t[i + 1 - w] == t[i] - (w as i64 - 1) * STEP
}

/// Mean of the finite cells of each contiguous window; NaN if there are none.
fn ref_mean(x: &[f64], w: usize, t: &[i64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    for i in w.saturating_sub(1)..x.len() {
        if !contig(t, i, w) {
            continue;
        }
        let (mut s, mut c) = (Dd::f(0.0), 0usize);
        for &v in &x[i + 1 - w..=i] {
            if v.is_finite() {
                s = s.add(Dd::f(v));
                c += 1;
            }
        }
        if c > 0 {
            out[i] = s.div(Dd::f(c as f64)).v();
        }
    }
    out
}

/// Centred population variance of windows whose cells are all finite.
fn ref_var(x: &[f64], w: usize, t: &[i64]) -> Vec<Option<Dd>> {
    let mut out = vec![None; x.len()];
    for i in w.saturating_sub(1)..x.len() {
        if !contig(t, i, w) || x[i + 1 - w..=i].iter().any(|v| !v.is_finite()) {
            continue;
        }
        let wd = Dd::f(w as f64);
        let mut s = Dd::f(0.0);
        for &v in &x[i + 1 - w..=i] {
            s = s.add(Dd::f(v));
        }
        let m = s.div(wd);
        let mut q = Dd::f(0.0);
        for &v in &x[i + 1 - w..=i] {
            let d = Dd::f(v).sub(m);
            q = q.add(d.mul(d));
        }
        out[i] = Some(q.div(wd));
    }
    out
}

fn ref_std(x: &[f64], w: usize, t: &[i64]) -> Vec<f64> {
    ref_var(x, w, t).into_iter().map(|v| v.map_or(f64::NAN, |d| d.v().max(0.0).sqrt())).collect()
}

/// Centred covariance of windows where both series are fully finite.
fn ref_cov(x: &[f64], y: &[f64], w: usize, t: &[i64]) -> Vec<Option<Dd>> {
    let mut out = vec![None; x.len()];
    for i in w.saturating_sub(1)..x.len() {
        let r = i + 1 - w..=i;
        if !contig(t, i, w) || x[r.clone()].iter().chain(y[r.clone()].iter()).any(|v| !v.is_finite()) {
            continue;
        }
        let wd = Dd::f(w as f64);
        let (mut sx, mut sy) = (Dd::f(0.0), Dd::f(0.0));
        for j in r.clone() {
            sx = sx.add(Dd::f(x[j]));
            sy = sy.add(Dd::f(y[j]));
        }
        let (mx, my) = (sx.div(wd), sy.div(wd));
        let mut q = Dd::f(0.0);
        for j in r {
            q = q.add(Dd::f(x[j]).sub(mx).mul(Dd::f(y[j]).sub(my)));
        }
        out[i] = Some(q.div(wd));
    }
    out
}

fn ret_k(c: &[f64], t: &[i64], k: usize) -> Vec<f64> {
    let mut out = vec![f64::NAN; c.len()];
    for i in k..c.len() {
        if t[i] - t[i - k] == k as i64 * STEP && c[i] > 0.0 && c[i - k] > 0.0 {
            out[i] = (c[i] / c[i - k]).ln();
        }
    }
    out
}

fn rlag(x: &[f64], t: &[i64]) -> Vec<f64> {
    let mut out = vec![f64::NAN; x.len()];
    for i in 1..x.len() {
        if t[i] - t[i - 1] == STEP {
            out[i] = x[i - 1];
        }
    }
    out
}

fn div(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && b != 0.0 {
        a / b
    } else {
        f64::NAN
    }
}

fn div_pos(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() && b > 0.0 {
        a / b
    } else {
        f64::NAN
    }
}

const FAMILIES: &[&str] = &[
    "ma96", "ma672", "vol16", "vol96", "vol_ratio", "vol_z2880", "bb96", "bb_width96", "parkinson96", "rv_ratio",
    "vz96", "ou_hl96", "acf1", "vr4", "rskew96", "semivar_asym", "jump_share",
];

/// Reference columns for one symbol. `None` marks a cell the reference does
/// not define (a partial window for the centred statistics).
fn reference(sym: &[Row]) -> Vec<(&'static str, Vec<Option<f64>>)> {
    let t: Vec<i64> = sym.iter().map(|r| r.t).collect();
    let c: Vec<f64> = sym.iter().map(|r| r.c).collect();
    let h: Vec<f64> = sym.iter().map(|r| r.h).collect();
    let l: Vec<f64> = sym.iter().map(|r| r.l).collect();
    let v: Vec<f64> = sym.iter().map(|r| r.v).collect();
    let n = sym.len();
    let some = |x: Vec<f64>| -> Vec<Option<f64>> { x.into_iter().map(Some).collect() };
    let mut out: Vec<(&'static str, Vec<Option<f64>>)> = Vec::new();

    let ma96 = ref_mean(&c, 96, &t);
    let ma672 = ref_mean(&c, 672, &t);
    let ratio = |m: &[f64]| -> Vec<f64> {
        (0..n).map(|i| if c[i] > 0.0 { div(c[i], m[i]) - 1.0 } else { f64::NAN }).collect()
    };
    out.push(("ma96", some(ratio(&ma96))));
    out.push(("ma672", some(ratio(&ma672))));

    let lr = ret_k(&c, &t, 1);
    let vol16 = ref_std(&lr, 16, &t);
    let vol96 = ref_std(&lr, 96, &t);
    let vol672 = ref_std(&lr, 672, &t);
    out.push(("vol16", some(vol16)));
    out.push(("vol96", some(vol96.clone())));
    out.push(("vol_ratio", some((0..n).map(|i| div_pos(vol96[i], vol672[i])).collect())));
    let zm = ref_mean(&vol96, 2880, &t);
    let zs = ref_std(&vol96, 2880, &t);
    out.push((
        "vol_z2880",
        some((0..n).map(|i| if zs[i] > 0.0 { div(vol96[i] - zm[i], zs[i]) } else { f64::NAN }).collect()),
    ));

    let sd96 = ref_std(&c, 96, &t);
    out.push((
        "bb96",
        some(
            (0..n)
                .map(|i| {
                    if sd96[i] > 0.0 {
                        let z = div(c[i] - ma96[i], 2.0 * sd96[i]);
                        if z.is_finite() { z.max(-1.0).min(1.0) } else { f64::NAN }
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
    ));
    out.push((
        "bb_width96",
        some((0..n).map(|i| if sd96[i].is_finite() { div_pos(4.0 * sd96[i], ma96[i]) } else { f64::NAN }).collect()),
    ));

    let hl2: Vec<f64> = (0..n)
        .map(|i| if h[i] > 0.0 && l[i] > 0.0 { let x = (h[i] / l[i].max(1e-12)).ln(); x * x } else { f64::NAN })
        .collect();
    let hl2m = ref_mean(&hl2, 96, &t);
    out.push((
        "parkinson96",
        some(hl2m.iter().map(|m| if m.is_finite() { (m / (4.0 * std::f64::consts::LN_2)).sqrt() } else { f64::NAN }).collect()),
    ));

    let lr2: Vec<f64> = lr.iter().map(|r| if r.is_finite() { r * r } else { f64::NAN }).collect();
    let rv96 = ref_mean(&lr2, 96, &t);
    let rv672 = ref_mean(&lr2, 672, &t);
    out.push((
        "rv_ratio",
        some((0..n).map(|i| if rv96[i] > 0.0 && rv672[i] > 0.0 { (rv96[i] / rv672[i]).ln() } else { f64::NAN }).collect()),
    ));

    let lv: Vec<f64> = v.iter().map(|x| if *x >= 0.0 { (1.0 + x).ln() } else { f64::NAN }).collect();
    let lvm = ref_mean(&lv, 96, &t);
    let lvs = ref_std(&lv, 96, &t);
    out.push((
        "vz96",
        some((0..n).map(|i| if lvs[i] > 0.0 && lvm[i].is_finite() { (lv[i] - lvm[i]) / lvs[i] } else { f64::NAN }).collect()),
    ));

    // ou_hl96: AR(1) of x = ln C - ln SMA96, centred in the window.
    let x: Vec<f64> = (0..n)
        .map(|i| if c[i] > 0.0 && ma96[i] > 0.0 { c[i].ln() - ma96[i].ln() } else { f64::NAN })
        .collect();
    let xl = rlag(&x, &t);
    let cov = ref_cov(&x, &xl, 96, &t);
    let var = ref_var(&xl, 96, &t);
    out.push((
        "ou_hl96",
        (0..n)
            .map(|i| {
                let (cv, vr) = (cov[i]?, var[i]?);
                if vr.v() <= 0.0 {
                    return Some(f64::NAN);
                }
                let omphi = vr.sub(cv).div(vr).v();
                let ln_phi = if omphi <= 1e-6 {
                    0.999999f64.ln()
                } else if omphi >= 1.0 - 1e-6 {
                    1e-6f64.ln()
                } else {
                    (-omphi).ln_1p()
                };
                let hl = -std::f64::consts::LN_2 / ln_phi;
                Some((1.0 + hl.max(0.0).min(1000.0)).ln())
            })
            .collect(),
    ));

    let lrl = rlag(&lr, &t);
    let acov = ref_cov(&lr, &lrl, 384, &t);
    let sd_lr = ref_std(&lr, 384, &t);
    let sd_lrl = ref_std(&lrl, 384, &t);
    out.push((
        "acf1",
        (0..n)
            .map(|i| {
                let cv = acov[i]?;
                let den = sd_lr[i] * sd_lrl[i];
                Some(if den > 0.0 { cv.v() / den } else { f64::NAN })
            })
            .collect(),
    ));
    let r4 = ret_k(&c, &t, 4);
    let r4_sd = ref_std(&r4, 384, &t);
    out.push((
        "vr4",
        some(
            (0..n)
                .map(|i| {
                    let den = 4.0 * sd_lr[i] * sd_lr[i];
                    if den > 0.0 { div(r4_sd[i] * r4_sd[i], den) } else { f64::NAN }
                })
                .collect(),
        ),
    ));

    let lr_m = ref_mean(&lr, 96, &t);
    let d2: Vec<f64> = lr.iter().zip(&lr_m).map(|(r, m)| (r - m) * (r - m)).collect();
    let d3: Vec<f64> = lr.iter().zip(&lr_m).map(|(r, m)| (r - m) * (r - m) * (r - m)).collect();
    let m2 = ref_mean(&d2, 96, &t);
    let m3 = ref_mean(&d3, 96, &t);
    out.push((
        "rskew96",
        some((0..n).map(|i| if m2[i] > 0.0 { div(m3[i], m2[i].powf(1.5)) } else { f64::NAN }).collect()),
    ));

    let dn2: Vec<f64> = lr
        .iter()
        .map(|r| if !r.is_finite() { f64::NAN } else if *r < 0.0 { r * r } else { 0.0 })
        .collect();
    let up2: Vec<f64> = lr
        .iter()
        .map(|r| if !r.is_finite() { f64::NAN } else if *r > 0.0 { r * r } else { 0.0 })
        .collect();
    let rsm = ref_mean(&dn2, 96, &t);
    let rsp = ref_mean(&up2, 96, &t);
    out.push((
        "semivar_asym",
        some((0..n).map(|i| if rv96[i] > 0.0 { div(rsm[i] - rsp[i], rv96[i]) } else { f64::NAN }).collect()),
    ));
    let abs_prod: Vec<f64> = lr.iter().zip(&lrl).map(|(a, b)| a.abs() * b.abs()).collect();
    let bv = ref_mean(&abs_prod, 96, &t);
    out.push((
        "jump_share",
        some(
            (0..n)
                .map(|i| {
                    if rv96[i] > 0.0 {
                        (rv96[i] - (std::f64::consts::PI / 2.0) * bv[i]).max(0.0) / rv96[i]
                    } else {
                        f64::NAN
                    }
                })
                .collect(),
        ),
    ));
    out
}

struct Err {
    max_rel: f64,
    max_scaled: f64,
    compared: usize,
    mask: usize,
}

fn compare(engine: &[f64], refc: &[Option<f64>], e: &mut Err) {
    for (a, b) in engine.iter().zip(refc) {
        let Some(b) = b else { continue };
        if a.is_finite() != b.is_finite() {
            e.mask += 1;
            continue;
        }
        if !b.is_finite() {
            continue;
        }
        let d = (a - b).abs();
        e.compared += 1;
        if *b != 0.0 {
            e.max_rel = e.max_rel.max(d / b.abs());
        } else if d > 0.0 {
            e.max_rel = f64::INFINITY;
        }
        e.max_scaled = e.max_scaled.max(d / (1.0 + b.abs()));
    }
}

fn measure(symbols: usize, per: usize) -> Vec<(&'static str, Err)> {
    let rows = fixture(symbols, per);
    let names: Vec<String> = FAMILIES.iter().map(|s| s.to_string()).collect();
    let (out, _) = build(&rows, &names, "ref");
    let mut errs: Vec<(&'static str, Err)> =
        FAMILIES.iter().map(|f| (*f, Err { max_rel: 0.0, max_scaled: 0.0, compared: 0, mask: 0 })).collect();
    let cols: Vec<Vec<f64>> = FAMILIES.iter().map(|f| read_column(&out, f)).collect();
    let mut start = 0;
    while start < rows.len() {
        let end = start + rows[start..].iter().take_while(|r| r.sym == rows[start].sym).count();
        let refs = reference(&rows[start..end]);
        for (k, (name, rc)) in refs.iter().enumerate() {
            assert_eq!(*name, FAMILIES[k]);
            compare(&cols[k][start..end], rc, &mut errs[k].1);
        }
        start = end;
    }
    errs
}

/// Prints the max error of every rolling family against the reference.
#[test]
fn rolling_families_error_table() {
    let (symbols, per) = (
        std::env::var("RCSR_WS_SYMBOLS").ok().and_then(|v| v.parse().ok()).unwrap_or(4),
        std::env::var("RCSR_WS_PER").ok().and_then(|v| v.parse().ok()).unwrap_or(12_000),
    );
    let errs = measure(symbols, per);
    eprintln!("[wstats] {symbols} symbols x {per} bars");
    eprintln!("[wstats] {:<14} {:>10} {:>12} {:>12} {:>6}", "column", "compared", "max_rel", "max_scaled", "mask");
    for (name, e) in &errs {
        eprintln!("[wstats] {:<14} {:>10} {:>12.3e} {:>12.3e} {:>6}", name, e.compared, e.max_rel, e.max_scaled, e.mask);
    }
    for (name, e) in &errs {
        assert!(e.compared > 0, "{name}: nothing compared");
        assert_eq!(e.mask, 0, "{name}: NaN mask differs from the reference");
    }
}

const LOCAL: &[&str] = &[
    "ma96", "ma672", "vol16", "vol96", "vol_ratio", "vol_z2880", "bb96", "bb_width96", "parkinson96", "rv_ratio",
    "vz96", "ou_hl96", "acf1", "vr4", "rskew96", "semivar_asym", "jump_share", "vol_z96", "taker_ratio96",
    "trade_intensity96", "amihud96", "cmf20", "vwap_dev96",
];

/// A symbol enriched alone and inside a 24-symbol panel gives identical bits.
#[test]
fn a_symbol_alone_equals_the_symbol_in_a_panel() {
    let per = 6_000;
    let rows = fixture(24, per);
    let names: Vec<String> = LOCAL.iter().map(|s| s.to_string()).collect();
    let (panel_out, _) = build(&rows, &names, "panel");
    let mut bad = Vec::new();
    for k in [0usize, 3, 5, 9, 16] {
        let alone: Vec<Row> = rows.iter().filter(|r| r.sym == k).copied().collect();
        let first = rows.iter().position(|r| r.sym == k).unwrap();
        let (alone_out, _) = build(&alone, &names, &format!("alone{k}"));
        for name in LOCAL {
            let a = read_column(&alone_out, name);
            let p = read_column(&panel_out, name);
            let p = &p[first..first + alone.len()];
            let diff = a.iter().zip(p).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            if diff > 0 {
                bad.push(format!("{name} sym {k}: {diff} of {} cells differ", a.len()));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// CPU feature-phase time on the 24-symbol, about 476k-row fixture.
#[test]
#[ignore = "timing probe: cargo test --release --test window_stats cpu_feature_phase -- --ignored --nocapture"]
fn cpu_feature_phase_time() {
    let rows = fixture(24, 21_000);
    let names: Vec<String> = enrich::engine_names();
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let (_, rep) = build(&rows, &names, "time");
        best = best.min(rep.feature_seconds);
        eprintln!("[wstats] rows {} columns {} feature_seconds {:.3}", rep.rows, names.len(), rep.feature_seconds);
    }
    eprintln!("[wstats] best feature_seconds {best:.3}");
}
