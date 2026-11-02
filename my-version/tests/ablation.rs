//! Add-one-family ablation: arm definitions and manifests, placebo and
//! permuted columns, scoring of a real run, and the ship/drop verdict.

use rcsr::ablation::{self, Definition, SeedResult};
use rcsr::data;
use rcsr::devlog::Json;
use rcsr::ladder::{self, Fold, Options, Panel};
use rcsr::protocol::{self, WindowRow};
use rcsr::rng::Mt;
use rcsr::walkforward::Window;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

fn definition() -> Definition {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json");
    ablation::parse_definition(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-ablation-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn gauss(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// --- definition, manifests, plan ---------------------------------------------

#[test]
fn the_committed_definition_has_the_six_arms() {
    let d = definition();
    let sizes: Vec<(String, usize)> = d.arms.iter().map(|a| (a.name.clone(), d.features(a).len())).collect();
    assert_eq!(
        sizes,
        vec![
            ("A0".to_string(), 10),
            ("A1".to_string(), 16),
            ("A2".to_string(), 22),
            ("A3".to_string(), 26),
            ("A4".to_string(), 29),
            ("A5".to_string(), 22)
        ]
    );
    assert!((8..=12).contains(&d.b0.len()), "B0 holds 8 to 12 columns");
    let a4 = d.features(d.arm("A4").unwrap());
    assert_eq!(a4.iter().filter(|f| f.starts_with("placebo_")).count(), 3);
    assert_eq!(&a4[..26], &d.features(d.arm("A3").unwrap())[..], "A4 is A3 plus the placebo columns");
    let a5 = d.features(d.arm("A5").unwrap());
    assert_eq!(a5.iter().filter(|f| f.starts_with("perm_")).count(), 12);
    assert_eq!(&d.features(d.arm("A2").unwrap())[..16], &d.features(d.arm("A1").unwrap())[..]);
    let hashes: Vec<String> = d.arms.iter().map(|a| ablation::feature_list_sha256(&d.features(a))).collect();
    let mut uniq = hashes.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), hashes.len(), "every arm has its own feature-list hash");
    assert_eq!(d.rule.families.len(), 3);
    assert_eq!(d.generation_budget, 300);
}

#[test]
fn a_definition_with_an_unknown_group_is_refused() {
    let text = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json")).unwrap();
    let bad = text.replace("\"add\": [\"candle_continuous\"]}", "\"add\": [\"nope\"]}");
    assert!(ablation::parse_definition(&bad).unwrap_err().contains("unknown group"));
}

#[test]
fn the_plan_writes_one_manifest_per_arm_and_every_command() {
    let d = definition();
    let files = ablation::plan_files(&ablation::PlanInput {
        def: &d,
        data: "panel.rcsb",
        target: "y60",
        time: "ts",
        group: Some("symbol"),
        return_col: Some("fwd60"),
        spec_columns: &[],
        target_spec: None,
        windows: "protocol/windows.csv",
        scheme: "expanding",
        out_dir: "out",
        finalist: false,
        search_args: "--objective pnl --cost-bps 12 --validation-frac 0.2",
        score_args: "",
        verdict_args: "",
    });
    let get = |n: &str| files.iter().find(|(k, _)| k == n).unwrap_or_else(|| panic!("no {n}")).1.clone();
    for arm in &d.arms {
        let m = Json::parse(&get(&format!("{}.arm.json", arm.name))).unwrap();
        assert_eq!(m.get("arm").unwrap().as_str(), Some(arm.name.as_str()));
        let feats: Vec<String> = match m.get("features").unwrap() {
            Json::Arr(a) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            _ => panic!(),
        };
        assert_eq!(feats, d.features(arm));
        assert_eq!(m.get("feature_list_sha256").unwrap().as_str().unwrap(), ablation::feature_list_sha256(&feats));
        assert_eq!(m.get("generation_budget").unwrap().as_f64(), Some(300.0));
        match m.get("seeds").unwrap() {
            Json::Arr(a) => assert_eq!(a.len(), 3),
            _ => panic!(),
        }
    }
    let script = get("commands.sh");
    assert_eq!(script.matches("--arm A").count(), 6 * 3, "one search per arm and seed");
    assert_eq!(script.matches("ablation score").count(), 6 * 3);
    assert!(script.contains("ablation prepare") && script.contains("ablation verdict"));
    assert!(script.contains("--generations 300"));

    let finalist = ablation::plan_files(&ablation::PlanInput {
        def: &d,
        data: "p.csv",
        target: "y",
        time: "ts",
        group: None,
        return_col: None,
        spec_columns: &[],
        target_spec: None,
        windows: "w.csv",
        scheme: "expanding",
        out_dir: "out",
        finalist: true,
        search_args: "",
        score_args: "",
        verdict_args: "",
    });
    let s = &finalist.iter().find(|(k, _)| k == "commands.sh").unwrap().1;
    assert_eq!(s.matches("--arm A").count(), 6 * 5, "five seeds for finalists");
}

// --- null columns -------------------------------------------------------------

fn ar1(n_groups: usize, len: usize, rho: f64, seed: u32) -> (Vec<f64>, Vec<u64>) {
    let mut rng = Mt::new(seed);
    let (mut x, mut g) = (Vec::new(), Vec::new());
    for k in 0..n_groups {
        let mut z = gauss(&mut rng);
        for _ in 0..len {
            z = rho * z + (1.0 - rho * rho).sqrt() * gauss(&mut rng);
            x.push(z);
            g.push(k as u64);
        }
    }
    (x, g)
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        c += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    c / (va * vb).sqrt()
}

#[test]
fn a_placebo_matches_prevalence_and_autocorrelation_and_is_unrelated() {
    let (z, g) = ar1(4, 3000, 0.7, 5);
    // A sparse, persistent pattern flag: 1 when the latent state is above its 85th percentile.
    let mut sorted = z.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let cut = sorted[(0.85 * sorted.len() as f64) as usize];
    let mut template: Vec<f64> = z.iter().map(|v| if *v > cut { 1.0 } else { 0.0 }).collect();
    template[0] = f64::NAN;
    template[3000] = f64::NAN;
    let placebo = ablation::placebo_column(&template, &g, 11, 1);
    let prevalence = |v: &[f64]| {
        let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        f.iter().sum::<f64>() / f.len() as f64
    };
    assert_eq!(prevalence(&placebo), prevalence(&template), "the marginal is matched exactly");
    assert_eq!(placebo.iter().filter(|v| v.is_nan()).count(), 2);
    assert!(placebo[0].is_nan() && placebo[3000].is_nan());
    let (ta, pa) = (ablation::lag1_autocorr(&template, &g), ablation::lag1_autocorr(&placebo, &g));
    assert!(ta > 0.3, "the template is persistent: {ta}");
    assert!((ta - pa).abs() < 0.1, "autocorrelation template {ta}, placebo {pa}");
    let pairs: Vec<(f64, f64)> = z.iter().zip(&placebo).filter(|(_, p)| p.is_finite()).map(|(a, b)| (*a, *b)).collect();
    let (a, b): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    assert!(corr(&a, &b).abs() < 0.05, "the placebo is unrelated to the latent state");
    let again = ablation::placebo_column(&template, &g, 11, 1);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&placebo), bits(&again), "reproducible");
    assert_ne!(placebo, ablation::placebo_column(&template, &g, 11, 2), "each placebo is its own draw");
}

#[test]
fn block_permutation_keeps_values_and_pairs_but_breaks_alignment() {
    let (x, g) = ar1(3, 2000, 0.5, 8);
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let out = ablation::permute_columns(&[&x, &y], &g, 96, 4);
    for k in 0..3 {
        let range = k * 2000..(k + 1) * 2000;
        let mut a: Vec<f64> = x[range.clone()].to_vec();
        let mut b: Vec<f64> = out[0][range.clone()].to_vec();
        a.sort_by(|p, q| p.total_cmp(q));
        b.sort_by(|p, q| p.total_cmp(q));
        assert_eq!(a, b, "group {k} keeps its multiset");
    }
    for i in 0..x.len() {
        assert_eq!(out[1][i], 2.0 * out[0][i] + 1.0, "columns of one family stay paired");
    }
    assert!(corr(&x, &out[0]).abs() < 0.2, "alignment with the original is gone: {}", corr(&x, &out[0]));
    assert_eq!(out, ablation::permute_columns(&[&x, &y], &g, 96, 4));
    let mut rng = Mt::new(1);
    let order = ablation::block_permutation(1000, 96, &mut rng);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
}

#[test]
fn prepare_writes_placebo_and_permuted_columns_next_to_the_real_ones() {
    let text = r#"{"schema":"rcsr-ablation/1","b0":["b"],"groups":{"fam":["f1","f2"],"pat":["p1"]},
"placebo_template_group":"pat","generation_budget":10,
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["fam","pat"],"placebo":2},{"name":"A5","add":[],"permute":["fam"]}],
"rule":{"families":[{"name":"fam","treatment":"A1","controls":["A0"],"null_arm":"A5","null_control":"A0"}]}}"#;
    let d = ablation::parse_definition(text).unwrap();
    let mut csv = String::from("ts,sym,y,b,f1,f2,p1\n");
    let mut rng = Mt::new(2);
    for s in ["b", "a"] {
        for t in (0..300).rev() {
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", 1_000_000 + t * 60, s, gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), (t % 7 == 0) as i32));
        }
    }
    let table = data::parse_table(&csv).unwrap();
    let out = ablation::prepare_csv(&ablation::PrepareInput { table: &table, def: &d, time: "ts", group: "sym", keep: &["y".to_string()], seed: 3, locked: Vec::new(), purge_seconds: 0, embargo_seconds: 0 }).unwrap();
    let prepared = data::parse_table(&out).unwrap();
    assert_eq!(prepared.names, vec!["ts", "sym", "y", "b", "f1", "f2", "p1", "placebo_1", "placebo_2", "perm_f1", "perm_f2"]);
    assert_eq!(prepared.len(), 600);
    let ts = prepared.col(0);
    assert!(ts[..300].windows(2).all(|w| w[0] <= w[1]), "rows are sorted by group and time");
    let p1 = prepared.col(prepared.index_of("placebo_1").unwrap());
    let pat = prepared.col(prepared.index_of("p1").unwrap());
    assert_eq!(p1.iter().sum::<f64>(), pat.iter().sum::<f64>(), "placebo_1 copies the prevalence of its template");
}

const NULL_DEF: &str = r#"{"schema":"rcsr-ablation/1","b0":["b"],"groups":{"fam":["f1","f2"],"pat":["p1"]},
"placebo_template_group":"pat","generation_budget":10,
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["fam","pat"],"placebo":2},{"name":"A5","add":[],"permute":["fam"]}],
"rule":{"families":[{"name":"fam","treatment":"A1","controls":["A0"],"null_arm":"A5","null_control":"A0"}]}}"#;

/// Two symbols, 300 rows each; every value from row `flip_from` on is redrawn
/// when `flipped`, so two stores differ only from that row on.
fn two_stores_csv(flipped: bool, flip_from: i64) -> String {
    let mut csv = String::from("ts,sym,y,b,f1,f2,p1\n");
    for (si, s) in ["a", "b"].iter().enumerate() {
        let mut rng = Mt::new(10 + si as u32);
        let mut alt = Mt::new(900 + si as u32);
        for k in 0..300i64 {
            let v: Vec<f64> = (0..4).map(|_| gauss(&mut rng)).collect();
            let w: Vec<f64> = (0..4).map(|_| gauss(&mut alt)).collect();
            let use_alt = flipped && k >= flip_from;
            let u = if use_alt { &w } else { &v };
            let p = if use_alt { (k % 3 == 0) as i32 } else { (k % 7 == 0) as i32 };
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", 1_000_000 + k * 60, s, u[0], u[1], u[2], u[3], p));
        }
    }
    csv
}

fn prepare_segments(csv: &str, locked: &[rcsr::lockbox::Lockbox], purge_seconds: i64) -> (Vec<String>, Vec<ablation::SegmentRows>) {
    let d = ablation::parse_definition(NULL_DEF).unwrap();
    let table = data::parse_table(csv).unwrap();
    let (out, seg) = ablation::prepare_csv_segments(&ablation::PrepareInput {
        table: &table,
        def: &d,
        time: "ts",
        group: "sym",
        keep: &["y".to_string()],
        seed: 3,
        locked: locked.to_vec(),
        purge_seconds,
        embargo_seconds: 0,
    })
    .unwrap();
    (out.lines().map(|l| l.to_string()).collect(), seg)
}

fn prepare_rows(csv: &str, locked: Option<rcsr::lockbox::Lockbox>) -> (Vec<String>, Vec<usize>) {
    let (rows, seg) = prepare_segments(csv, locked.as_slice(), 0);
    (rows, seg.iter().map(|s| s.rows).collect())
}

#[test]
fn null_columns_on_development_rows_ignore_the_locked_range() {
    let start = 1_000_000 + 200 * 60;
    let lb = rcsr::lockbox::Lockbox { start, end: start + 100 * 60 };
    let (a, seg_a) = prepare_rows(&two_stores_csv(false, 200), Some(lb));
    let (b, seg_b) = prepare_rows(&two_stores_csv(true, 200), Some(lb));
    assert_eq!(a[0], b[0]);
    let dev = |rows: &[String]| rows[1..].iter().filter(|l| l.split(',').next().unwrap().parse::<i64>().unwrap() < start).cloned().collect::<Vec<_>>();
    assert_eq!(dev(&a).len(), 400);
    assert_eq!(dev(&a), dev(&b), "development rows, including placebo_* and perm_*, are byte-identical");
    assert_eq!(seg_a, [400, 200, 0]);
    assert_eq!(seg_a, seg_b);
    assert_ne!(a, b, "the locked rows differ");

    let (c, _) = prepare_rows(&two_stores_csv(false, 200), None);
    let (d, _) = prepare_rows(&two_stores_csv(true, 200), None);
    assert_ne!(dev(&c), dev(&d), "without a declaration locked values still reach development rows");
}

#[test]
fn a_range_outside_the_data_changes_nothing() {
    use rcsr::lockbox::Lockbox;
    let csv = two_stores_csv(false, 200);
    let (plain, seg) = prepare_rows(&csv, None);
    assert_eq!(seg, [600]);
    let before = Lockbox { start: 100, end: 200 };
    let after = Lockbox { start: 2_000_000_000, end: 2_100_000_000 };
    for ranges in [vec![before], vec![after], vec![before, after]] {
        for purge in [0, 219_600] {
            let (outside, seg) = prepare_segments(&csv, &ranges, purge);
            let counts: Vec<usize> = seg.iter().map(|s| s.rows).collect();
            assert_eq!(counts.iter().sum::<usize>(), 600);
            assert_eq!(counts.iter().filter(|&&c| c > 0).count(), 1, "one non-empty segment");
            assert_eq!(plain, outside, "{:?} with purge {}", ranges, purge);
        }
    }
}

const HOUR: i64 = 3600;
const H0: i64 = 1_704_067_200;
const ROWS: i64 = 5280;
const PURGE: i64 = 219_600;

fn range_of(from: &str, to: &str) -> rcsr::lockbox::Lockbox {
    rcsr::lockbox::Lockbox::parse(&format!("{},{}", from, to)).unwrap()
}

/// 2 symbols x 5,280 hourly rows from 2024-01-01. Rows inside any `redraw`
/// interval hold other values, so two files differ only inside those intervals.
fn hourly_csv(redraw: &[(i64, i64)]) -> String {
    let mut csv = String::from("ts,sym,y,b,f1,f2,p1\n");
    for (si, s) in ["a", "b"].iter().enumerate() {
        let mut base = Mt::new(30 + si as u32);
        let mut alt = Mt::new(700 + si as u32);
        for k in 0..ROWS {
            let t = H0 + k * HOUR;
            let v: Vec<f64> = (0..4).map(|_| gauss(&mut base)).collect();
            let w: Vec<f64> = (0..4).map(|_| gauss(&mut alt)).collect();
            let use_alt = redraw.iter().any(|&(lo, hi)| t >= lo && t < hi);
            let (u, p) = if use_alt { (&w, (k % 3 == 0) as i32) } else { (&v, (k % 7 == 0) as i32) };
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", t, s, u[0], u[1], u[2], u[3], p));
        }
    }
    csv
}

fn row_time(line: &str) -> i64 {
    line.split(',').next().unwrap().parse().unwrap()
}

fn rows_where(rows: &[String], keep: impl Fn(i64) -> bool) -> Vec<String> {
    rows[1..].iter().filter(|l| keep(row_time(l))).cloned().collect()
}

#[test]
fn null_columns_ignore_both_declared_ranges() {
    let (r1, r2) = (range_of("2024-03-01", "2024-04-01"), range_of("2024-06-01", "2024-07-01"));
    let base = hourly_csv(&[]);
    for purge in [PURGE, 0] {
        let (a, _) = prepare_segments(&base, &[r1, r2], purge);
        for (inside, label) in [(r1, "R1"), (r2, "R2")] {
            let (b, _) = prepare_segments(&hourly_csv(&[(inside.start, inside.end)]), &[r1, r2], purge);
            assert_ne!(a, b, "{} rows differ", label);
            let dev = |t: i64| t < r1.start - purge || (t >= r1.end && t < r2.start - purge) || t >= r2.end;
            let (da, db) = (rows_where(&a, dev), rows_where(&b, dev));
            assert!(da.len() > 7000, "development rows: {}", da.len());
            assert_eq!(da, db, "purge {}: development rows ignore values inside {}", purge, label);
        }
    }
}

#[test]
fn purge_band_is_its_own_segment() {
    let s = range_of("2024-06-01", "2024-07-01");
    let a = hourly_csv(&[]);
    let b = hourly_csv(&[(s.start - PURGE, s.start)]);
    let (ra, seg) = prepare_segments(&a, &[s], PURGE);
    let (rb, _) = prepare_segments(&b, &[s], PURGE);
    let kinds: Vec<_> = seg.iter().map(|x| x.kind).collect();
    assert_eq!(kinds, [ablation::SegmentKind::Development, ablation::SegmentKind::Purge, ablation::SegmentKind::Locked, ablation::SegmentKind::Development]);
    assert_eq!(rows_where(&ra, |t| t < s.start - PURGE), rows_where(&rb, |t| t < s.start - PURGE));
    assert_eq!(rows_where(&ra, |t| t >= s.start), rows_where(&rb, |t| t >= s.start));
    assert_ne!(rows_where(&ra, |t| t >= s.start - PURGE && t < s.start), rows_where(&rb, |t| t >= s.start - PURGE && t < s.start));
}

#[test]
fn overlapping_and_adjacent_ranges_split_into_elementary_segments() {
    use ablation::SegmentKind::{Development as D, Locked as L, Purge as P};
    use rcsr::lockbox::Lockbox;
    let lb = |start, end| Lockbox { start, end };
    let layout = |ranges: &[Lockbox], purge| {
        let s = ablation::Segments::new(ranges, purge);
        (0..s.len()).map(|r| (s.kind(r), s.span(r))).collect::<Vec<_>>()
    };
    let span = |a, b| (Some(a), Some(b));

    let overlap = layout(&[lb(100, 200), lb(150, 300)], 10);
    assert_eq!(
        overlap,
        [(D, (None, Some(90))), (P, span(90, 100)), (L, span(100, 140)), (L, span(140, 150)), (L, span(150, 200)), (L, span(200, 300)), (D, (Some(300), None))]
    );
    let nested = layout(&[lb(100, 300), lb(150, 200)], 0);
    assert_eq!(nested, [(D, (None, Some(100))), (L, span(100, 150)), (L, span(150, 200)), (L, span(200, 300)), (D, (Some(300), None))]);
    let adjacent = layout(&[lb(100, 200), lb(200, 300)], 10);
    assert_eq!(adjacent, [(D, (None, Some(90))), (P, span(90, 100)), (L, span(100, 190)), (L, span(190, 200)), (L, span(200, 300)), (D, (Some(300), None))]);
    let touching = layout(&[lb(100, 200), lb(200, 300)], 0);
    assert_eq!(touching, [(D, (None, Some(100))), (L, span(100, 200)), (L, span(200, 300)), (D, (Some(300), None))]);
    let gap = layout(&[lb(100, 200), lb(400, 500)], 50);
    assert_eq!(gap.iter().map(|x| x.0).collect::<Vec<_>>(), [D, P, L, D, P, L, D]);

    let s = ablation::Segments::new(&[lb(100, 200)], 10);
    assert_eq!((s.raw(89.9), s.raw(90.0), s.raw(99.0), s.raw(100.0), s.raw(199.0), s.raw(200.0)), (0, 1, 1, 2, 2, 3));
}

#[test]
fn one_range_without_purge_is_unchanged() {
    use rcsr::lockbox::{self, Lockbox};
    let start = 1_000_000 + 200 * 60;
    let lb = Lockbox { start, end: start + 100 * 60 };
    let dir = scratch("one-range");
    let raw = dir.join("raw.csv");
    std::fs::write(&raw, two_stores_csv(false, 200)).unwrap();
    lockbox::write_sidecar(raw.to_str().unwrap(), &lb, "test").unwrap();
    let def = dir.join("def.json");
    std::fs::write(&def, NULL_DEF).unwrap();
    let out = dir.join("prepared.csv");
    let run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "prepare", "--definition", def.to_str().unwrap(), "--data", raw.to_str().unwrap(), "--time", "ts", "--group", "sym"])
        .args(["--keep", "y", "--seed", "3", "--purge-seconds", "0", "--out", out.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    let sha = rcsr::manifest::sha256_hex(&std::fs::read(&out).unwrap());
    assert_eq!(sha, PINNED_ONE_RANGE_SHA256);
}

const PINNED_ONE_RANGE_SHA256: &str = "bcd7b921418ae0fe5339195fa4b3aa4c158cf3c808547a9b7c17336cca96a023";

fn prepare_cli(dir: &std::path::Path, data: &str, out: &str, extra: &[&str]) -> std::process::Output {
    let def = dir.join("def.json");
    std::fs::write(&def, NULL_DEF).unwrap();
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "prepare", "--definition", def.to_str().unwrap(), "--data", data, "--time", "ts", "--group", "sym"])
        .args(["--keep", "y", "--seed", "3", "--out", out])
        .args(extra)
        .output()
        .unwrap()
}

fn sidecar_rows(out: &str) -> Vec<usize> {
    let text = std::fs::read_to_string(rcsr::lockbox::sidecar_path(out)).unwrap();
    let list = &text[text.find("\"segments\":[").expect("segments list") + 12..];
    list[..list.find(']').unwrap()].split("\"rows\":").skip(1).map(|p| p.split('}').next().unwrap().parse().unwrap()).collect()
}

#[test]
fn a_flag_range_disjoint_from_the_protocol_default_is_segmented() {
    let dir = scratch("flag-range");
    let (a, b) = (dir.join("A.csv"), dir.join("B.csv"));
    let f = range_of("2024-06-01", "2024-07-01");
    assert_eq!((f.start, f.end), (1_717_200_000, 1_719_792_000));
    std::fs::write(&a, hourly_csv(&[])).unwrap();
    std::fs::write(&b, hourly_csv(&[(f.start, f.end)])).unwrap();
    let flag = format!("{},{}", f.start, f.end);
    let (oa, ob) = (dir.join("oa.csv"), dir.join("ob.csv"));
    for (src, out) in [(&a, &oa), (&b, &ob)] {
        let run = prepare_cli(&dir, src.to_str().unwrap(), out.to_str().unwrap(), &["--lockbox", &flag]);
        assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    }
    assert_eq!(sidecar_rows(oa.to_str().unwrap()), [7174, 122, 1440, 1824, 0, 0, 0, 0, 0]);
    let side = std::fs::read_to_string(rcsr::lockbox::sidecar_path(oa.to_str().unwrap())).unwrap();
    assert!(side.contains("\"purge_seconds\":219600,\"purge_source\":\"protocol_default\""), "{}", side);
    let (ta, tb) = (std::fs::read_to_string(&oa).unwrap(), std::fs::read_to_string(&ob).unwrap());
    let rows = |t: &str| t.lines().map(str::to_string).collect::<Vec<_>>();
    let (ra, rb) = (rows(&ta), rows(&tb));
    let before = |t: i64| t < f.start - PURGE;
    let after = |t: i64| t >= f.end;
    assert_eq!(rows_where(&ra, before).len(), 7174);
    assert_eq!(rows_where(&ra, before), rows_where(&rb, before));
    assert_eq!(rows_where(&ra, after).len(), 1824);
    assert_eq!(rows_where(&ra, after), rows_where(&rb, after));
}

#[test]
fn a_second_prepare_keeps_the_range_roles() {
    use rcsr::lockbox;
    let dir = scratch("roles");
    let a = dir.join("A.csv");
    std::fs::write(&a, hourly_csv(&[])).unwrap();
    let f = range_of("2024-06-01", "2024-07-01");
    let flag = format!("{},{}", f.start, f.end);
    let (o1, o2) = (dir.join("o1.csv"), dir.join("o2.csv"));
    let run = prepare_cli(&dir, a.to_str().unwrap(), o1.to_str().unwrap(), &["--lockbox", &flag]);
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    let run = prepare_cli(&dir, o1.to_str().unwrap(), o2.to_str().unwrap(), &[]);
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    let roles = |p: &std::path::Path| lockbox::stored_all(p.to_str().unwrap()).unwrap().map(|s| (s.range, s.also));
    assert_eq!(roles(&o1), Some((lockbox::protocol_default(), Some(f))));
    assert_eq!(roles(&o2), roles(&o1));
    assert_eq!(sidecar_rows(o2.to_str().unwrap()).iter().sum::<usize>(), 10560);
}

#[test]
fn a_two_range_sidecar_segments_both_ranges_through_the_cli() {
    use rcsr::lockbox;
    let dir = scratch("two-range-sidecar");
    let (r1, f) = (range_of("2024-03-01", "2024-04-01"), range_of("2024-06-01", "2024-07-01"));
    let write = |name: &str, redraw: &[(i64, i64)]| {
        let p = dir.join(name);
        std::fs::write(&p, hourly_csv(redraw)).unwrap();
        let text = format!("{{\"locked_test\":{},\"locked_test_also\":{}}}\n", r1.declaration_json("test"), f.declaration_json("test"));
        std::fs::write(lockbox::sidecar_path(p.to_str().unwrap()), text).unwrap();
        p
    };
    let (a, c) = (write("A.csv", &[]), write("C.csv", &[(r1.start, r1.end)]));
    let (oa, oc) = (dir.join("oa.csv"), dir.join("oc.csv"));
    for (src, out) in [(&a, &oa), (&c, &oc)] {
        let run = prepare_cli(&dir, src.to_str().unwrap(), out.to_str().unwrap(), &[]);
        assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    }
    let rows = |p: &std::path::Path| std::fs::read_to_string(p).unwrap().lines().map(str::to_string).collect::<Vec<_>>();
    let (ra, rc) = (rows(&oa), rows(&oc));
    let dev = |t: i64| t < r1.start - PURGE || (t >= r1.end && t < f.start - PURGE) || t >= f.end;
    assert_eq!(rows_where(&ra, dev), rows_where(&rc, dev), "values inside the first range stay out of development rows");
    let kept = lockbox::stored_all(oa.to_str().unwrap()).unwrap().unwrap();
    assert_eq!((kept.range, kept.also), (r1, Some(f)));
}

#[test]
fn prepare_purge_source() {
    let dir = scratch("purge-source");
    let a = dir.join("A.csv");
    std::fs::write(&a, hourly_csv(&[])).unwrap();
    let f = range_of("2024-06-01", "2024-07-01");
    let flag = format!("{},{}", f.start, f.end);
    let one = std::fs::read_to_string(write_target_spec_one_bar(&dir)).unwrap();
    let entry = &one[one.find("[{").unwrap() + 1..one.rfind("}]").unwrap() + 1];
    let second = entry.replace("\"y\"", "\"y2\"").replace("raw_target", "second_target").replace("\"r\"", "\"r2\"").replace("\"horizon_seconds\":900", "\"horizon_seconds\":7200");
    let spec2 = dir.join("two.json");
    std::fs::write(&spec2, one.replace("}]", &format!("}},{}]", second))).unwrap();
    let side = |name: &str, extra: &[&str]| {
        let out = dir.join(name);
        let mut args = vec!["--lockbox", flag.as_str()];
        args.extend_from_slice(extra);
        let run = prepare_cli(&dir, a.to_str().unwrap(), out.to_str().unwrap(), &args);
        (run.status.code(), std::fs::read_to_string(rcsr::lockbox::sidecar_path(out.to_str().unwrap())).unwrap_or_default(), String::from_utf8_lossy(&run.stderr).into_owned())
    };
    let (code, text, err) = side("flag.csv", &["--purge-seconds", "3600"]);
    assert_eq!(code, Some(0), "{}", err);
    assert!(text.contains("\"purge_seconds\":3600,\"purge_source\":\"flag\""), "{}", text);
    let (code, text, err) = side("spec.csv", &["--target-spec", spec2.to_str().unwrap()]);
    assert_eq!(code, Some(0), "{}", err);
    assert!(text.contains("\"purge_seconds\":7200,\"purge_source\":\"target_spec\""), "{}", text);
    let (code, text, err) = side("default.csv", &[]);
    assert_eq!(code, Some(0), "{}", err);
    assert!(text.contains("\"purge_seconds\":219600,\"purge_source\":\"protocol_default\""), "{}", text);
    for bad in ["-1", "NaN", "inf"] {
        let (code, _, err) = side("bad.csv", &["--purge-seconds", bad]);
        assert_eq!(code, Some(2), "{}: {}", bad, err);
    }
}

#[test]
fn permutation_stays_inside_its_segment() {
    let n = 300usize;
    let groups: Vec<u64> = (0..n).map(|i| (i / 150) as u64).collect();
    let segs: Vec<u8> = (0..n).map(|i| if i % 150 < 100 { 0 } else { 1 }).collect();
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let keys = ablation::segment_keys(&groups, &segs);
    let out = ablation::permute_columns(&[&x], &keys, 16, 5);
    for i in 0..n {
        let seg_of = |v: f64| (groups[v as usize], segs[v as usize]);
        assert_eq!(seg_of(out[0][i]), (groups[i], segs[i]), "row {} holds a value from another symbol or segment", i);
    }
    assert_ne!(out[0], x);
    assert_eq!(ablation::segment_keys(&groups, &vec![0u8; n]), groups, "segment 0 keeps the plain group id");
}

#[test]
fn placebo_marginal_stays_inside_its_segment() {
    let n = 400usize;
    let groups = vec![0u64; n];
    let segs: Vec<u8> = (0..n).map(|i| if i < 250 { 0 } else { 1 }).collect();
    let template: Vec<f64> = (0..n).map(|i| if i < 250 { (i % 5) as f64 } else { 100.0 + (i % 3) as f64 }).collect();
    let out = ablation::placebo_column_segmented(&template, &groups, &segs, 7, 1);
    assert!(out[..250].iter().all(|v| *v < 5.0), "segment 0 holds only its own template values");
    assert!(out[250..].iter().all(|v| *v >= 100.0));
    let mut a = out[..250].to_vec();
    let mut t = template[..250].to_vec();
    a.sort_by(|p, q| p.total_cmp(q));
    t.sort_by(|p, q| p.total_cmp(q));
    assert_eq!(a, t, "each segment keeps its exact marginal");
}

// --- scoring a run --------------------------------------------------------------

const T0: f64 = 1_600_000_000.0;

fn objective_csv(dir: &std::path::Path) -> String {
    let n = 400usize;
    let frac = |v: f64| v - v.floor();
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..n {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", (T0 as i64) + 900 * i as i64, x1, x2, r, r));
    }
    let path = dir.join("d.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_target_spec(dir: &std::path::Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
\"horizon_seconds\":900,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
\"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
\"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("spec.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// As [`write_target_spec`], with a horizon of one 900-second bar so that the rows do not overlap.
fn write_target_spec_one_bar(dir: &std::path::Path) -> String {
    let path = dir.join("spec900.json");
    let text = std::fs::read_to_string(write_target_spec(dir)).unwrap();
    std::fs::write(&path, text.replace("\"horizon_seconds\":3600", "\"horizon_seconds\":900")).unwrap();
    path.to_str().unwrap().to_string()
}

fn search(data: &str, spec: &str, extra: &[&str]) -> std::process::Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal", "--target-spec", spec, "--cpu",
        "--no-dev-log", "--seed", "3", "--generations", "15", "--pop", "32", "--metric", "squared", "--quiet",
    ];
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(&args).output().expect("run rcsr")
}

#[test]
fn run_json_records_arm_feature_hash_and_generation_budget() {
    let dir = scratch("runjson");
    let data = objective_csv(&dir);
    let spec = write_target_spec(&dir);
    let out = search(&data, &spec, &["--json", "--arm", "A7"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let j = Json::parse(String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    let a = j.get("ablation").expect("ablation object");
    assert_eq!(a.get("arm").unwrap().as_str(), Some("A7"));
    assert_eq!(a.get("feature_list_sha256").unwrap().as_str(), Some(ablation::feature_list_sha256(&["x1".to_string(), "x2".to_string()]).as_str()));
    assert_eq!(a.get("generation_budget").unwrap().as_f64(), Some(15.0));
    let reached = a.get("generations_run").unwrap().as_f64().unwrap();
    assert!(reached >= 1.0 && reached <= 15.0);
    let plain = search(&data, &spec, &["--json"]);
    assert!(Json::parse(String::from_utf8_lossy(&plain.stdout).trim()).unwrap().get("ablation").is_none(), "no --arm, no ablation object");
}

#[test]
fn a_real_run_is_scored_on_the_folds_without_the_test_block_in_selection() {
    let dir = scratch("score");
    let data = objective_csv(&dir);
    let spec = write_target_spec(&dir);
    let run_json = dir.join("A0-s3.run.json");
    let out = search(&data, &spec, &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--arm", "A0", "--json-out", run_json.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(300),
            iso(400)
        ),
    )
    .unwrap();
    let res = dir.join("A0-s3.result.json");
    let score = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r", "--time", "t",
            "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--out", res.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(score.status.code(), Some(0), "{}", String::from_utf8_lossy(&score.stderr));
    let r = SeedResult::parse(&std::fs::read_to_string(&res).unwrap()).unwrap();
    assert_eq!(r.arm, "A0");
    assert_eq!(r.seed, 3);
    assert_eq!(r.generation_budget, 15);
    assert_eq!(r.features, vec!["x1", "x2"]);
    assert_eq!(r.folds, vec!["w0"]);
    assert_eq!(r.thresholds, vec![0.6, 0.7, 0.8]);
    assert_eq!(r.net_by_fold_threshold.len(), 1);
    assert_eq!(r.net_by_fold_threshold[0].len(), 3);
    assert_eq!(r.net_period.len(), 100);
    assert!(r.return_mse.is_finite());
    assert!((r.net_by_fold_threshold[0][1] - r.net_period.iter().sum::<f64>()).abs() < 1e-12, "headline threshold is the series");
    assert!(r.net_period.iter().sum::<f64>() > 0.0, "the long book wins on a signal-driven return");
    assert!(std::fs::read_to_string(&res).unwrap().contains("\"min_group_step_seconds\":900"), "the result carries the measured step");
    let overlap = dir.join("spec2700.json");
    std::fs::write(&overlap, std::fs::read_to_string(&spec).unwrap().replace("\"horizon_seconds\":900", "\"horizon_seconds\":2700")).unwrap();
    let refused = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r", "--time", "t",
            "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--target-spec", overlap.to_str().unwrap(),
            "--out", dir.join("overlap.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(refused.status.code(), Some(2));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("overlapping returns") && err.contains("2700-second label horizon"), "{err}");
    assert_eq!(r.feature_list_sha256, ablation::feature_list_sha256(&r.features));

    let plain = dir.join("plain.run.json");
    let out = search(&data, &spec, &["--json-out", plain.to_str().unwrap(), "--arm", "A0"]);
    assert_eq!(out.status.code(), Some(0));
    let bad = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "score", "--run", plain.to_str().unwrap(), "--data", &data, "--target", "y", "--time", "t", "--windows", windows.to_str().unwrap(), "--out", dir.join("x.json").to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("objective.finalist"), "a run without a validation-chosen finalist is refused");
}

#[test]
fn parse_run_refuses_a_hash_that_does_not_match_the_variables() {
    let text = r#"{"seed":1,"variables":["a","b"],"ablation":{"arm":"A0","feature_list_sha256":"00","generation_budget":5,"generations_run":5},
"objective":{"finalist":{"expression":"a"}}}"#;
    assert!(ablation::parse_run(text).unwrap_err().contains("does not match"));
}

#[test]
fn score_run_reads_the_expression_and_its_inputs() {
    let mut rng = Mt::new(4);
    let (mut time, mut group, mut s, mut n, mut ret) = (vec![], vec![], vec![], vec![], vec![]);
    for g in 0..4u64 {
        for t in 0..300 {
            let sig = gauss(&mut rng);
            time.push(T0 + t as f64 * 3600.0);
            group.push(g);
            s.push(sig);
            n.push(gauss(&mut rng));
            ret.push(0.004 * sig + 0.002 * gauss(&mut rng));
        }
    }
    let panel = Panel::new(vec!["s".into(), "n".into()], vec![s, n], time, group, ret.clone(), ret, vec![]).unwrap();
    let rows = vec![WindowRow {
        scheme: "expanding".into(),
        window: Window { label: "w".into(), train_start: T0, train_end: T0 + 200.0 * 3600.0, test_start: T0 + 205.0 * 3600.0, test_end: T0 + 300.0 * 3600.0 },
        validation_start: T0 + 120.0 * 3600.0,
    }];
    let folds: Vec<Fold> = ladder::make_folds(&panel, &rows, 3600.0, &[], 0.0).unwrap();
    let facts = ablation::RunFacts {
        arm: "A1".into(),
        feature_list_sha256: ablation::feature_list_sha256(&["s".into(), "n".into()]),
        generation_budget: 10,
        generations_run: 7,
        variables: vec!["s".into(), "n".into()],
        expression: "s".into(),
        seed: 2,
    };
    let r = ablation::score_run(&facts, &panel, &folds, &[0.5, 0.7], 1, &Options::default()).unwrap();
    assert_eq!(r.features_used, vec!["s"]);
    assert_eq!(r.generations_run, 7);
    assert_eq!(r.net_period.len(), 95);
    assert!(r.net_by_fold_threshold[0][1] > 0.0);
}

// --- verdict ------------------------------------------------------------------------

struct Scenario {
    mse: BTreeMap<&'static str, f64>,
    /// Mean per-period net per arm.
    net: BTreeMap<&'static str, f64>,
    mse_noise: f64,
    /// Arm and fold whose net difference is pushed negative.
    flip: Option<(&'static str, usize)>,
}

fn results(d: &Definition, sc: &Scenario, seeds: u32) -> Vec<SeedResult> {
    let mut out = Vec::new();
    for (ai, arm) in d.arms.iter().enumerate() {
        for s in 1..=seeds {
            let mut rng = Mt::new(1000 + s + 97 * ai as u32);
            let mut period = Vec::new();
            for _ in 0..360 {
                period.push(sc.net[arm.name.as_str()] + 0.002 * gauss(&mut rng));
            }
            // The first 100 periods belong to fold 0, and so on.
            let folds = 3usize;
            let thresholds = vec![0.6, 0.7, 0.8];
            let mut by = Vec::new();
            for f in 0..folds {
                let base: f64 = period[f * 120..(f + 1) * 120].iter().sum();
                let mut row = Vec::new();
                for q in 0..3 {
                    let mut v = base * (1.0 + 0.05 * q as f64);
                    if let Some((a, ff)) = sc.flip {
                        if a == arm.name && ff == f {
                            v -= 1.0;
                        }
                    }
                    row.push(v);
                }
                by.push(row);
            }
            let feats = d.features(arm);
            out.push(SeedResult {
                arm: arm.name.clone(),
                seed: s,
                feature_list_sha256: ablation::feature_list_sha256(&feats),
                generation_budget: d.generation_budget,
                generations_run: d.generation_budget,
                features: feats.clone(),
                features_used: feats[..2].to_vec(),
                return_mse: sc.mse[arm.name.as_str()] + sc.mse_noise * gauss(&mut rng),
                folds: vec!["f0".into(), "f1".into(), "f2".into()],
                thresholds,
                net_by_fold_threshold: by,
                net_period: period,
            });
        }
    }
    out
}

fn base_scenario() -> Scenario {
    let mse: BTreeMap<_, _> = [("A0", 1.0), ("A1", 0.90), ("A2", 0.95), ("A3", 0.95), ("A4", 0.95), ("A5", 1.0)].into_iter().collect();
    let net: BTreeMap<_, _> = [("A0", 0.0), ("A1", 0.0015), ("A2", 0.0005), ("A3", 0.0005), ("A4", 0.0005), ("A5", 0.0)].into_iter().collect();
    Scenario { mse, net, mse_noise: 0.004, flip: None }
}

fn verdict_of(sc: &Scenario, seeds: u32) -> Result<ablation::Verdict, String> {
    let d = definition();
    ablation::verdict(&d, &results(&d, sc, seeds), &BTreeMap::new())
}

fn family<'a>(v: &'a ablation::Verdict, name: &str) -> &'a ablation::FamilyVerdict {
    v.families.iter().find(|f| f.name == name).unwrap()
}

#[test]
fn a_real_family_ships_and_one_without_a_gain_over_its_control_is_dropped() {
    let v = verdict_of(&base_scenario(), 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "ship", "failed: {:?}", candle.failed);
    assert!(candle.comparisons[0].gain_mse > candle.comparisons[0].seed_range);
    assert!(candle.comparisons[0].spa.as_ref().unwrap().p_consistent < 0.05);
    let multi = family(&v, "multi_bar");
    assert_eq!(multi.verdict, "drop", "A2 equals A1: no gain");
    assert!(multi.failed.iter().any(|f| f.starts_with("gain_exceeds_seed_range")));
    let patterns = family(&v, "patterns_trend");
    assert_eq!(patterns.verdict, "drop", "A3 does not beat the placebo arm A4");
    assert!(patterns.failed.iter().any(|f| f.contains("vs A4")), "{:?}", patterns.failed);
    let js = v.json();
    for needle in ["\"verdict\":\"ship\"", "\"verdict\":\"drop\"", "\"n_trials\"", "\"dsr\"", "\"pbo\"", "\"bh_adjusted\"", "\"null_band\"", "\"generation_budget\"", "\"feature_list_sha256\""] {
        assert!(js.contains(needle), "verdict json lacks {needle}");
    }
    assert!(v.markdown().contains("| candle | A1 | ship |"));
    let d = definition();
    assert_eq!(v.n_trials, d.n_trials(3, 3));
    assert!(v.notes.iter().any(|n| n.contains("selection on training data repeated inside every fold")));
}

#[test]
fn a_gain_inside_the_null_band_is_dropped() {
    let mut sc = base_scenario();
    sc.mse.insert("A5", 0.90);
    sc.net.insert("A5", 0.0015);
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("outside_null_band")), "{:?}", candle.failed);
    let (lo, hi) = candle.null_band.unwrap();
    assert!(lo <= hi && hi > 0.05, "the permuted arm reaches the family's gain: {lo} {hi}");
}

#[test]
fn a_gain_smaller_than_the_seed_range_is_dropped() {
    let mut sc = base_scenario();
    sc.mse_noise = 0.2;
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("gain_exceeds_seed_range")));
}

#[test]
fn a_sign_flip_across_quarters_drops_the_family() {
    let mut sc = base_scenario();
    sc.flip = Some(("A1", 2));
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("no_sign_flip")), "{:?}", candle.failed);
    assert!(candle.comparisons[0].sign_by_fold.iter().any(|x| *x < 0.0));
    assert!(candle.comparisons[0].sign_by_fold.iter().any(|x| *x > 0.0));
}

#[test]
fn a_family_that_does_not_survive_fdr_or_the_deflated_sharpe_is_dropped() {
    let mut sc = base_scenario();
    sc.net.insert("A1", 0.0001);
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("spa_p_below_alpha") || f.starts_with("fdr_survives") || f.starts_with("dsr_above_min")), "{:?}", candle.failed);
}

#[test]
fn inconsistent_inputs_are_refused_not_averaged() {
    let d = definition();
    let sc = base_scenario();
    let all = results(&d, &sc, 3);
    // too few seeds
    let two: Vec<SeedResult> = all.iter().filter(|r| r.seed <= 2).cloned().collect();
    assert!(ablation::verdict(&d, &two, &BTreeMap::new()).unwrap_err().contains("seeds"));
    // a different generation budget in one run
    let mut budget = all.clone();
    budget[0].generation_budget = 299;
    assert!(ablation::verdict(&d, &budget, &BTreeMap::new()).unwrap_err().contains("generation budget"));
    // a feature list that is not the arm's
    let mut hash = all.clone();
    hash[4].feature_list_sha256 = "deadbeef".into();
    assert!(ablation::verdict(&d, &hash, &BTreeMap::new()).unwrap_err().contains("feature_list_sha256"));
    // a hash that differs from the arm manifest
    let mut expected = BTreeMap::new();
    expected.insert("A0".to_string(), "other".to_string());
    assert!(ablation::verdict(&d, &all, &expected).unwrap_err().contains("arm manifest"));
    // different seeds across the compared arms
    let mut seeds = all.clone();
    for r in seeds.iter_mut().filter(|r| r.arm == "A1" && r.seed == 3) {
        r.seed = 9;
    }
    assert!(ablation::verdict(&d, &seeds, &BTreeMap::new()).unwrap_err().contains("different seeds"));
}

#[test]
fn the_cli_verdict_reads_a_directory_of_result_files() {
    let dir = scratch("verdict");
    let d = definition();
    for r in results(&d, &base_scenario(), 3) {
        std::fs::write(dir.join(format!("{}-s{}.result.json", r.arm, r.seed)), r.json()).unwrap();
    }
    let def_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json");
    let out = dir.join("verdict.json");
    let md = dir.join("verdict.md");
    let run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "verdict", "--definition", def_path.to_str().unwrap(), "--results", dir.to_str().unwrap(), "--out", out.to_str().unwrap(), "--markdown", md.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    let j = Json::parse(std::fs::read_to_string(&out).unwrap().trim()).unwrap();
    assert_eq!(j.get("schema").unwrap().as_str(), Some("rcsr-ablation-verdict/1"));
    assert!(std::fs::read_to_string(&md).unwrap().contains("ship"));
}

#[test]
fn the_cli_plan_writes_manifests_to_a_directory() {
    let dir = scratch("plan");
    let data = dir.join("panel.csv");
    let d = definition();
    let mut header = vec!["ts".to_string(), "sym".to_string(), "y".to_string()];
    header.extend(d.real_columns());
    std::fs::write(&data, header.join(",") + "\n").unwrap();
    let def_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json");
    let run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "plan", "--definition", def_path.to_str().unwrap(), "--data", data.to_str().unwrap(), "--target", "y", "--time", "ts",
            "--group", "sym", "--windows", "protocol/windows.csv", "--out", dir.join("out").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    for arm in ["A0", "A1", "A2", "A3", "A4", "A5"] {
        let m = std::fs::read_to_string(dir.join("out").join(format!("{arm}.arm.json"))).unwrap();
        assert!(m.contains(&format!("\"arm\":\"{arm}\"")) && m.contains("feature_list_sha256") && m.contains("\"generation_budget\":300"));
    }
    assert!(dir.join("out/commands.sh").exists() && dir.join("out/definition.json").exists());
    let missing = std::fs::write(&data, "ts,sym,y\n").and_then(|_| {
        Ok(Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "ablation", "plan", "--definition", def_path.to_str().unwrap(), "--data", data.to_str().unwrap(), "--target", "y", "--time", "ts",
                "--windows", "w.csv", "--out", dir.join("out2").to_str().unwrap(),
            ])
            .output()
            .unwrap())
    });
    let missing = missing.unwrap();
    assert_eq!(missing.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("missing columns"));
}

#[test]
fn a_family_that_loses_only_on_pbo_is_dropped_with_reason_pbo() {
    let sc = base_scenario();
    let d = definition();
    let rs = results(&d, &sc, 3);
    let ok = ablation::verdict(&d, &rs, &BTreeMap::new()).unwrap();
    let pbo = ok.pbo.as_ref().expect("pbo computed").pbo;
    assert!(pbo < d.rule.max_pbo, "pbo {pbo}");
    assert_eq!(d.rule.max_pbo, 0.5);
    assert_eq!(family(&ok, "candle").verdict, "ship");

    let mut strict = d.clone();
    strict.rule.max_pbo = pbo / 2.0;
    let v = ablation::verdict(&strict, &rs, &BTreeMap::new()).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert_eq!(candle.failed.len(), 1, "{:?}", candle.failed);
    assert!(candle.failed[0].starts_with("pbo:"), "{:?}", candle.failed);
}

#[test]
fn a_declared_locked_test_is_refused_by_score_and_carried_by_prepare() {
    use rcsr::lockbox::{self, Lockbox};

    let dir = scratch("locked-test");
    let data = objective_csv(&dir);
    let spec = write_target_spec_one_bar(&dir);
    let run_json = dir.join("A0-s3.run.json");
    let out = search(&data, &spec, &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--arm", "A0", "--json-out", run_json.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(300),
            iso(400)
        ),
    )
    .unwrap();
    let lb = Lockbox { start: T0 as i64 + 900 * 300, end: T0 as i64 + 900 * 400 };
    lockbox::write_sidecar(&data, &lb, "test").unwrap();
    let score = |res: &str| {
        Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r",
                "--time", "t", "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--out", res,
            ])
            .output()
            .unwrap()
    };
    let refused = score(dir.join("refused.json").to_str().unwrap());
    assert_eq!(refused.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("declared locked test"), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(!dir.join("refused.json").exists());

    // prepare copies the declaration to a sidecar and a search on the CSV refuses the rows
    let def = dir.join("def.json");
    std::fs::write(
        &def,
        r#"{"schema":"rcsr-ablation/1","b0":["b"],"groups":{"fam":["f1","f2"],"pat":["p1"]},
"placebo_template_group":"pat","generation_budget":10,
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["fam"]},{"name":"A5","add":[],"permute":["fam"]}],
"rule":{"families":[{"name":"fam","treatment":"A1","controls":["A0"],"null_arm":"A5","null_control":"A0"}]}}"#,
    )
    .unwrap();
    let mut csv = String::from("ts,sym,y,b,f1,f2,p1\n");
    let mut rng = Mt::new(2);
    for s in ["a", "b"] {
        for k in 0..400i64 {
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", T0 as i64 + 900 * k, s, gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), (k % 7 == 0) as i32));
        }
    }
    let raw = dir.join("raw.csv");
    std::fs::write(&raw, csv).unwrap();
    lockbox::write_sidecar(raw.to_str().unwrap(), &lb, "test").unwrap();
    let prepared = dir.join("prepared.csv");
    let prep = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "prepare", "--definition", def.to_str().unwrap(), "--data", raw.to_str().unwrap(), "--time", "ts", "--group", "sym",
            "--keep", "y", "--out", prepared.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(prep.status.code(), Some(0), "{}", String::from_utf8_lossy(&prep.stderr));
    assert_eq!(lockbox::stored(prepared.to_str().unwrap()).unwrap().map(|(_, l)| l), Some(lb));
    let side = std::fs::read_to_string(lockbox::sidecar_path(prepared.to_str().unwrap())).unwrap();
    assert_eq!(sidecar_rows(prepared.to_str().unwrap()), [112, 488, 200, 0, 0], "{}", side);
    assert!(side.contains("\"purge_source\":\"protocol_default\""), "{}", side);
    let found = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", prepared.to_str().unwrap(), "--target", "y", "--vars", "b,f1,f2", "--time", "ts", "--split", "temporal", "--cpu",
            "--no-dev-log", "--seed", "3", "--generations", "3", "--pop", "16", "--quiet",
        ])
        .output()
        .unwrap();
    assert_eq!(found.status.code(), Some(3), "{}", String::from_utf8_lossy(&found.stderr));
    assert!(String::from_utf8_lossy(&found.stderr).contains("declared locked test"));

    // undeclared data and a flag range disjoint from the protocol default: the output declares both
    let raw2 = dir.join("raw2.csv");
    std::fs::copy(&raw, &raw2).unwrap();
    let prepared2 = dir.join("prepared2.csv");
    let range = format!("{},{}", lb.start, lb.end);
    let prep2 = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "prepare", "--definition", def.to_str().unwrap(), "--data", raw2.to_str().unwrap(), "--time", "ts", "--group", "sym",
            "--keep", "y", "--out", prepared2.to_str().unwrap(), "--lockbox", &range,
        ])
        .output()
        .unwrap();
    assert_eq!(prep2.status.code(), Some(0), "{}", String::from_utf8_lossy(&prep2.stderr));
    let stored = lockbox::stored_all(prepared2.to_str().unwrap()).unwrap().unwrap();
    assert_eq!((stored.range, stored.also), (lockbox::protocol_default(), Some(lb)));
    let resolved = lockbox::resolve(None, prepared2.to_str().unwrap()).unwrap();
    assert_eq!(resolved.ranges(), vec![lockbox::protocol_default(), lb]);
    let found = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", prepared2.to_str().unwrap(), "--target", "y", "--vars", "b,f1,f2", "--time", "ts", "--split", "temporal", "--cpu",
            "--no-dev-log", "--seed", "3", "--generations", "3", "--pop", "16", "--quiet",
        ])
        .output()
        .unwrap();
    assert_eq!(found.status.code(), Some(3), "{}", String::from_utf8_lossy(&found.stderr));
    assert!(String::from_utf8_lossy(&found.stderr).contains("declared locked test"));
}

#[test]
fn stale_sidecar_is_refused() {
    use rcsr::lockbox::{self, Lockbox};

    let dir = scratch("stale-sidecar");
    let data = objective_csv(&dir);
    let spec = write_target_spec_one_bar(&dir);
    let run_json = dir.join("A0.run.json");
    let out = search(&data, &spec, &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--arm", "A0", "--json-out", run_json.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let windows = dir.join("w.csv");
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(290),
            iso(400)
        ),
    )
    .unwrap();
    let score = || {
        Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r",
                "--time", "t", "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--out", dir.join("o.json").to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };

    let lb = Lockbox { start: T0 as i64 + 900 * 300, end: T0 as i64 + 900 * 400 };
    lockbox::write_sidecar(&data, &lb, "test").unwrap();
    let text = std::fs::read_to_string(lockbox::sidecar_path(&data)).unwrap();
    assert!(text.contains("\"data_bytes\":") && text.contains("\"data_sha256\":"), "{text}");
    assert!(lockbox::resolve(None, &data).is_ok());

    // same length, one digit changed: the SHA-256 differs
    let original = std::fs::read(&data).unwrap();
    let mut changed = original.clone();
    let i = changed.iter().rposition(|c| c.is_ascii_digit()).unwrap();
    changed[i] = if changed[i] == b'0' { b'1' } else { b'0' };
    std::fs::write(&data, &changed).unwrap();
    let e = lockbox::resolve(None, &data).unwrap_err();
    assert_eq!(e.exit, 3, "{}", e.message);
    assert!(e.message.contains("stale") && e.message.contains("sha256"), "{}", e.message);
    let found = search(&data, &spec, &[]);
    assert_eq!(found.status.code(), Some(3), "{}", String::from_utf8_lossy(&found.stderr));
    assert!(String::from_utf8_lossy(&found.stderr).contains("stale"), "{}", String::from_utf8_lossy(&found.stderr));
    let refused = score();
    assert_eq!(refused.status.code(), Some(3), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("stale"), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(!dir.join("o.json").exists());

    // another length is refused by the byte count
    let mut longer = original.clone();
    longer.push(b'\n');
    std::fs::write(&data, &longer).unwrap();
    let e = lockbox::resolve(None, &data).unwrap_err();
    assert_eq!(e.exit, 3, "{}", e.message);
    assert!(e.message.contains("bytes"), "{}", e.message);

    // the original bytes are accepted again
    std::fs::write(&data, &original).unwrap();
    assert!(lockbox::resolve(None, &data).is_ok());
}

const PLAN_DEF: &str = r#"{"schema":"rcsr-ablation/1","b0":["b"],"groups":{"fam":["f1"],"pat":["p1"]},
"placebo_template_group":"pat","generation_budget":6,
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["fam","pat"],"placebo":1},{"name":"A5","add":[],"permute":["fam"]}],
"rule":{"families":[{"name":"fam","treatment":"A1","controls":["A0"],"null_arm":"A5","null_control":"A0"}]}}"#;

fn plan_store(dir: &std::path::Path) -> String {
    let n = 400usize;
    let frac = |v: f64| v - v.floor();
    let mut text = String::from("ts,sym,b,f1,p1,lbl_y,lbl_fwd,vol96\n");
    for i in 0..n {
        let f = i as f64;
        let b = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let f1 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let r = 0.01 * b + 0.0005 * (frac(f * 0.324_717_957_244_746) - 0.5);
        text.push_str(&format!("{},a,{},{},{},{},{},1\n", (T0 as i64) + 900 * i as i64, b, f1, (i % 5 == 0) as i32, r, r));
    }
    let path = dir.join("panel.csv");
    std::fs::write(&path, text).unwrap();
    let spec = std::fs::read_to_string(write_target_spec(dir)).unwrap();
    let spec = spec
        .replace("\"column\":\"y\"", "\"column\":\"lbl_y\"")
        .replace("\"return_column\":\"r\"", "\"return_column\":\"lbl_fwd\"")
        .replace("\"scale\":{\"kind\":\"constant\",\"value\":1.0}", "\"scale\":{\"kind\":\"column\",\"column\":\"vol96\",\"multiplier\":1,\"window_seconds\":900,\"ends_at_offset_seconds\":0}")
        .replace("\"construction_columns\":[]", "\"construction_columns\":[\"lbl_fwd\",\"vol96\"]");
    std::fs::write(dir.join("panel.targets.json"), spec).unwrap();
    path.to_string_lossy().into_owned()
}

fn plan_cli(dir: &std::path::Path, data: &str, extra: &[&str]) -> String {
    let def = dir.join("def.json");
    std::fs::write(&def, PLAN_DEF).unwrap();
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(300),
            iso(400)
        ),
    )
    .unwrap();
    let out = dir.join("plan");
    let o = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "plan", "--definition", def.to_str().unwrap(), "--data", data, "--target", "lbl_y", "--time", "ts", "--group", "sym"])
        .args(["--windows", windows.to_str().unwrap(), "--out", out.to_str().unwrap()])
        .args(extra)
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(0), "{}", String::from_utf8_lossy(&o.stderr));
    std::fs::read_to_string(out.join("commands.sh")).unwrap()
}

fn keep_of(script: &str) -> String {
    let prepare = script.lines().find(|l| l.starts_with("rcsr ablation prepare")).unwrap();
    let words: Vec<&str> = prepare.split_whitespace().collect();
    words[words.iter().position(|w| *w == "--keep").unwrap() + 1].to_string()
}

#[test]
fn plan_keeps_the_return_and_construction_columns() {
    let dir = scratch("plan-keep");
    let data = plan_store(&dir);
    let script = plan_cli(&dir, &data, &["--return-col", "lbl_fwd"]);
    assert_eq!(keep_of(&script), "lbl_y,lbl_fwd,vol96");
    assert!(script.lines().filter(|l| l.contains("--arm A")).all(|l| l.contains("--target-spec")), "{script}");

    let no_spec = scratch("plan-keep-nospec");
    let data = plan_store(&no_spec);
    std::fs::remove_file(no_spec.join("panel.targets.json")).unwrap();
    assert_eq!(keep_of(&plan_cli(&no_spec, &data, &[])), "lbl_y");
}

#[test]
fn the_planned_lines_for_one_arm_and_seed_run_as_written() {
    let dir = scratch("plan-run");
    let data = plan_store(&dir);
    let script = plan_cli(&dir, &data, &["--return-col", "lbl_fwd", "--search-args", "--objective pnl --cost-bps 5 --validation-frac 0.3 --split temporal --time ts --cpu --no-dev-log --pop 16 --quiet", "--score-args", "--purge-seconds 900"]);
    let lines: Vec<&str> = script
        .lines()
        .filter(|l| l.starts_with("rcsr ablation prepare") || (l.contains("A0-s1.") && l.starts_with("rcsr")))
        .collect();
    assert_eq!(lines.len(), 3, "{script}");
    for line in lines {
        let mut words = line.split_whitespace();
        assert_eq!(words.next(), Some("rcsr"));
        let o = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(words).output().unwrap();
        assert_eq!(o.status.code(), Some(0), "{line}\n{}", String::from_utf8_lossy(&o.stderr));
    }
}
