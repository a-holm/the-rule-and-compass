//! Train/test splitting for i.i.d., time-series and grouped (panel) data.

use rcsr::data::{self, SplitMode, SplitSpec};
use rcsr::intervals::{Declared as IvDeclared, EmbargoSource, IntervalSplit, Role};
use rcsr::lockbox::{Declared as LockDeclared, Embargo, Lockbox};
use rcsr::rng::Mt;

fn spec(mode: SplitMode) -> SplitSpec {
    SplitSpec {
        mode,
        test_frac: 0.3,
        ..Default::default()
    }
}

/// Rows `1..=n`, target = the row index, one input column `x`.
fn series_csv(n: usize) -> data::Table {
    let mut s = String::from("t,x,y\n");
    for i in 1..=n {
        s.push_str(&format!("{},{},{}\n", i, i, i));
    }
    data::parse_table(&s).unwrap()
}

#[test]
fn label_columns_are_kept_as_text() {
    let t = data::parse_table("date,stock,close,y\n2024-01-01,AAPL,1.5,1\n2024-01-02,MSFT,2.5,2\n")
        .unwrap();
    assert_eq!(t.len(), 2);
    assert!(t.numeric[2] && t.numeric[3]);
    assert!(!t.numeric[0] && !t.numeric[1], "date/stock must not be numeric");
    assert!(t.value(0, 0).is_nan());
    assert_eq!(t.text(0, 1), Some("AAPL"));
    assert_eq!(t.distinct_count(1), 2);
}

#[test]
fn temporal_split_is_chronological_and_honours_the_gap() {
    let t = series_csv(10);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0); // column `t`
    s.gap = 2;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();

    let train_y = &p.train.y;
    let test_y = &p.test.y;
    assert_eq!(train_y, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    // Rows 8 and 9 are the embargo, so only row 10 is tested.
    assert_eq!(test_y, &vec![10.0]);
    let max_train = train_y.iter().cloned().fold(f64::MIN, f64::max);
    let min_test = test_y.iter().cloned().fold(f64::MAX, f64::min);
    assert!(max_train < min_test, "train must precede test in time");
    assert_eq!(p.report.gap, 2);
    assert_eq!(p.report.mode, "temporal");
}

#[test]
fn normalisation_statistics_come_from_training_rows_only() {
    let t = series_csv(10);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0);
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();

    // Training rows are 1..=7, whose mean is 4. Using all ten rows would give
    // 5.5 — that leak is exactly what this test guards against.
    assert!((p.means[0] - 4.0).abs() < 1e-12, "mean {:?}", p.means);
    // And the training inputs must be z-scored with those statistics.
    let m: f64 = p.train.x.iter().map(|r| r[0]).sum::<f64>() / p.train.x.len() as f64;
    assert!(m.abs() < 1e-12);
}

#[test]
fn group_split_never_puts_a_group_on_both_sides() {
    // Three stocks, four rows each; the target encodes the stock.
    let mut s = String::from("stock,x,y\n");
    for (g, name) in ["AAPL", "MSFT", "NVDA"].iter().enumerate() {
        for i in 0..4 {
            s.push_str(&format!("{},{},{}\n", name, i, g));
        }
    }
    let t = data::parse_table(&s).unwrap();
    let mut sp = SplitSpec {
        mode: SplitMode::Group,
        group_col: Some(0),
        test_frac: 0.34,
        ..Default::default()
    };
    sp.folds = 1;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &sp, &mut rng).unwrap();

    let train_groups: Vec<f64> = p.train.y.clone();
    let test_groups: Vec<f64> = p.test.y.clone();
    assert!(!test_groups.is_empty() && !train_groups.is_empty());
    for g in &test_groups {
        assert!(
            !train_groups.contains(g),
            "group {} appears on both sides",
            g
        );
    }
    assert_eq!(p.report.groups_total, 3);
    assert_eq!(p.report.groups_train + p.report.groups_test, 3);
    assert_eq!(p.report.groups_test, 1, "one whole stock should be held out");
}

#[test]
fn temporal_group_tests_every_series_and_uses_whole_series_history() {
    let mut s = String::from("t,stock,x,y\n");
    for (g, name) in ["A", "B"].iter().enumerate() {
        for i in 1..=10 {
            s.push_str(&format!("{},{},{},{}\n", i, name, i, g));
        }
    }
    let t = data::parse_table(&s).unwrap();
    let sp = SplitSpec {
        mode: SplitMode::TemporalGroup,
        time_col: Some(0),
        group_col: Some(1),
        test_frac: 0.3,
        ..Default::default()
    };
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &sp, &mut rng).unwrap();

    // Every group contributes to both sides.
    let mut train_groups: Vec<f64> = p.train.y.clone();
    train_groups.sort_by(|a, b| a.partial_cmp(b).unwrap());
    train_groups.dedup();
    let mut test_groups: Vec<f64> = p.test.y.clone();
    test_groups.sort_by(|a, b| a.partial_cmp(b).unwrap());
    test_groups.dedup();
    assert_eq!(train_groups, vec![0.0, 1.0]);
    assert_eq!(test_groups, vec![0.0, 1.0]);
    // 3 of each series is tested (rows 8..=10), the rest trained.
    assert_eq!(p.test.len(), 6);
    assert_eq!(p.train.len(), 14);
}

#[test]
fn a_single_fold_uses_the_pooled_test_block_without_a_second_copy() {
    let t = series_csv(12);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0);
    s.folds = 1;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();
    // #133: the pooled test block is the single fold; `fold_samples` only
    // materialises per-fold copies when there is more than one fold.
    assert!(p.fold_samples.is_empty());
    assert!(!p.test.is_empty());
}

#[test]
fn folds_partition_the_test_region() {
    let t = series_csv(12);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0);
    s.folds = 3;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();

    assert_eq!(p.fold_samples.len(), 3);
    let total: usize = p.fold_samples.iter().map(|f| f.len()).sum();
    assert_eq!(total, p.test.len());
    assert_eq!(total, 4); // 30% of 12, rounded
    for f in &p.fold_samples {
        assert!(!f.is_empty());
    }
    // Blocks are contiguous and non-overlapping in time, in order.
    let bounds: Vec<(f64, f64)> = p
        .fold_samples
        .iter()
        .map(|f| {
            (
                f.y.iter().cloned().fold(f64::MAX, f64::min),
                f.y.iter().cloned().fold(f64::MIN, f64::max),
            )
        })
        .collect();
    for w in bounds.windows(2) {
        assert!(w[0].1 < w[1].0, "folds must not overlap: {:?}", bounds);
    }
}

#[test]
fn auto_detection_finds_time_and_group_columns() {
    let mut s = String::from("date,stock,x,y\n");
    for name in ["A", "B"] {
        for d in 1..=6 {
            s.push_str(&format!("2024-01-0{},{},{},{}\n", d, name, d, d));
        }
    }
    let t = data::parse_table(&s).unwrap();
    let (mode, tc, gc, _notes) = data::suggest_split(&t, 3, &[2]);
    assert_eq!(mode, SplitMode::TemporalGroup);
    assert_eq!(t.names[tc.unwrap()], "date");
    assert_eq!(t.names[gc.unwrap()], "stock");

    // Exchangeable rows stay random.
    let t2 = data::parse_table("a,b\n0.1,1\n0.7,2\n0.3,3\n0.9,4\n0.2,5\n").unwrap();
    let (m2, _, _, _) = data::suggest_split(&t2, 1, &[0]);
    assert_eq!(m2, SplitMode::Random);
}

#[test]
fn dates_parse_and_order_correctly() {
    let a = data::parse_time("2024-01-31").unwrap();
    let b = data::parse_time("2024-02-01").unwrap();
    let c = data::parse_time("2024-01-31T09:30:00").unwrap();
    assert!(a < b && a < c && c - a == 9.0 * 3600.0 + 30.0 * 60.0);
    assert_eq!(data::parse_time("2024/02/01"), Some(b));
    assert_eq!(data::parse_time("42.5"), Some(42.5));
    assert!(data::parse_time("not-a-date").is_none());
}

#[test]
fn delayed_columns_look_back_one_row() {
    let t = data::parse_table("t,x,y\n0,10,1\n1,20,2\n2,30,3\n3,40,4\n").unwrap();
    let (t2, vars, rep) = data::add_lags(
        &t,
        &["x".into()],
        &data::LagSpec {
            lags: 1,
            ..Default::default()
        },
        Some(0),
        None,
    )
    .unwrap();
    assert_eq!(vars, vec!["x".to_string(), "x_lag1".to_string()]);
    assert_eq!(rep.rows_kept, 3); // the first row is history only
    let x = t2.index_of("x").unwrap();
    let lag = t2.index_of("x_lag1").unwrap();
    assert_eq!(t2.value(0, x), 20.0);
    assert_eq!(t2.value(0, lag), 10.0);
    assert_eq!(t2.value(1, lag), 20.0);
    assert_eq!(t2.value(2, lag), 30.0);
}

#[test]
fn delays_do_not_cross_group_boundaries() {
    let t = data::parse_table("g,x\nA,1\nA,2\nB,100\nB,200\n").unwrap();
    let (t2, _v, rep) = data::add_lags(
        &t,
        &["x".into()],
        &data::LagSpec {
            lags: 1,
            ..Default::default()
        },
        None,
        Some(0),
    )
    .unwrap();
    assert_eq!(rep.series, 2);
    assert_eq!(rep.rows_kept, 2);
    let lag = t2.index_of("x_lag1").unwrap();
    assert_eq!(t2.value(0, lag), 1.0, "A's second row looks back inside A");
    assert_eq!(t2.value(1, lag), 100.0, "B must not see A's last value");
}

#[test]
fn delays_do_not_cross_time_gaps() {
    let t = data::parse_table("t,x\n0,1\n1,2\n100,3\n101,4\n").unwrap();
    let (t2, _v, rep) = data::add_lags(
        &t,
        &["x".into()],
        &data::LagSpec {
            lags: 1,
            gap_break: 2.0,
            ..Default::default()
        },
        Some(0),
        None,
    )
    .unwrap();
    assert_eq!(rep.breaks, 1);
    assert_eq!(rep.series, 2);
    let lag = t2.index_of("x_lag1").unwrap();
    assert_eq!(t2.value(0, lag), 1.0); // t=1 looks back to t=0
    assert_eq!(t2.value(1, lag), 3.0, "the lag must not jump the gap");
}

#[test]
fn history_fraction_drops_leading_rows_but_still_supplies_lags() {
    let mut s = String::from("t,x\n");
    for i in 0..10 {
        s.push_str(&format!("{},{}\n", i, i * 10));
    }
    let t = data::parse_table(&s).unwrap();
    let (t2, _v, rep) = data::add_lags(
        &t,
        &["x".into()],
        &data::LagSpec {
            lags: 1,
            history_frac: 0.3,
            ..Default::default()
        },
        Some(0),
        None,
    )
    .unwrap();
    assert_eq!(rep.history_rows, 3);
    assert_eq!(rep.rows_kept, 7);
    let lag = t2.index_of("x_lag1").unwrap();
    // first scored row is t=3 (x=30); its lag comes from the history row t=2.
    assert_eq!(t2.value(0, lag), 20.0);
}

/// Issue #70: the report says which row selector the split is stored as.
#[test]
fn split_report_names_the_row_selector() {
    let t = data::parse_table(
        "t,x,y\n0,1,1\n1,2,2\n2,3,3\n3,4,4\n4,5,5\n5,6,6\n6,7,7\n7,8,8\n",
    )
    .unwrap();
    let p = data::prepare_full_with(
        &t,
        &["x".into()],
        "y",
        &SplitSpec {
            mode: SplitMode::Temporal,
            time_col: Some(0),
            test_frac: 0.25,
            ..Default::default()
        },
        &mut Mt::new(1),
        &Default::default(),
    )
    .unwrap();
    assert!(p.report.selector.contains("range"), "{}", p.report.selector);

    let p = data::prepare_full_with(
        &t,
        &["x".into()],
        "y",
        &SplitSpec {
            test_frac: 0.25,
            ..Default::default()
        },
        &mut Mt::new(1),
        &Default::default(),
    )
    .unwrap();
    assert!(p.report.selector.contains("set"), "{}", p.report.selector);
}

// ---------------------------------------------------------------------------
// Partition law: every row has at most one role, every other row is discarded.
// ---------------------------------------------------------------------------

const BASE: i64 = 1_000_000_000;
const H: i64 = 3600;
const MODES: [SplitMode; 5] = [
    SplitMode::Random,
    SplitMode::Temporal,
    SplitMode::Group,
    SplitMode::TemporalGroup,
    SplitMode::TemporalIntervals,
];
const CASES: usize = 600;
const OK_FLOOR: usize = 400;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Shape {
    Unique,
    Tied,
    Gapped,
}

#[derive(Clone, Copy, PartialEq)]
enum PurgePolicy {
    Mixed,
    Always,
    Never,
}

/// A table whose target is the row id and whose stamps and groups are known by id.
/// Ids ascend with the stamp, so the id order is the time order.
struct Fixture {
    n: usize,
    stamp: Vec<i64>,
    group: Vec<usize>,
    shape: Shape,
    shuffled: bool,
    table: data::Table,
}

struct Case {
    mode: SplitMode,
    test_frac: f64,
    gap: usize,
    vfrac: f64,
    folds: usize,
    purge: Option<f64>,
    iv: Option<IntervalSplit>,
}

fn layout(rng: &mut Mt, n: usize, shape: Shape, groups: usize, step: i64, gap_unit: i64, gap_max: usize) -> (Vec<i64>, Vec<usize>) {
    let mut stamp = Vec::with_capacity(n);
    let mut group = Vec::with_capacity(n);
    let mut t = BASE;
    match shape {
        Shape::Unique => {
            for _ in 0..n {
                stamp.push(t);
                group.push(rng.usize_below(groups));
                t += step;
            }
        }
        Shape::Gapped => {
            for _ in 0..n {
                stamp.push(t);
                group.push(rng.usize_below(groups));
                t += (1 + rng.usize_below(gap_max)) as i64 * gap_unit;
            }
        }
        Shape::Tied => {
            while stamp.len() < n {
                let k = (1 + rng.usize_below(4)).min(groups).min(n - stamp.len());
                let mut gs: Vec<usize> = (0..groups).collect();
                for j in 0..k {
                    let s = j + rng.usize_below(groups - j);
                    gs.swap(j, s);
                    stamp.push(t);
                    group.push(gs[j]);
                }
                t += step;
            }
        }
    }
    (stamp, group)
}

fn build_table(stamp: &[i64], group: &[usize], rng: &mut Mt, shuffled: bool) -> data::Table {
    let n = stamp.len();
    let mut order: Vec<usize> = (0..n).collect();
    if shuffled {
        for i in (1..n).rev() {
            let j = rng.usize_below(i + 1);
            order.swap(i, j);
        }
    }
    let mut s = String::from("t,g,x,y\n");
    for &id in &order {
        s.push_str(&format!("{},g{},{},{}\n", stamp[id], group[id], (0.37 * id as f64).sin(), id));
    }
    data::parse_table(&s).unwrap()
}

const PURGE_HOURS: [i64; 4] = [1, 5, 24, 72];
const INTERVAL_HOURS: [i64; 3] = [1, 5, 12];

/// Contiguous hourly blocks, each a train, validation, test or unused interval,
/// with at least one train and one test block.
fn random_intervals(rng: &mut Mt) -> (IntervalSplit, i64) {
    let blocks = 3 + rng.usize_below(6);
    let mut roles: Vec<usize> = (0..blocks).map(|_| rng.usize_below(4)).collect(); // 0 validation, 1 test, 2 unused, 3 train
    if !roles.contains(&3) {
        let i = rng.usize_below(blocks);
        roles[i] = 3;
    }
    while !roles.contains(&1) {
        let i = rng.usize_below(blocks);
        if roles[i] == 3 && roles.iter().filter(|&&r| r == 3).count() == 1 {
            continue;
        }
        roles[i] = 1;
    }
    let mut d = IvDeclared { split_id: "law".to_string(), ..IvDeclared::default() };
    let mut t = BASE;
    for r in roles {
        let len = (24 + rng.usize_below(97)) as i64 * H;
        match r {
            0 => d.validation.push((t, t + len)),
            1 => d.test.push((t, t + len)),
            3 => d.train.push((t, t + len)),
            _ => {}
        }
        t += len;
    }
    let purge = *rng.pick(&INTERVAL_HOURS) * H;
    let embargo = *rng.pick(&INTERVAL_HOURS) * H;
    (IntervalSplit::new(d, purge, embargo, EmbargoSource::Flag, None), t - BASE)
}

fn gen_case(rng: &mut Mt, mode: SplitMode, policy: PurgePolicy) -> (Case, Fixture) {
    let shape = *rng.pick(&[Shape::Unique, Shape::Tied, Shape::Gapped]);
    let groups = 1 + rng.usize_below(6);
    let shuffled = rng.bool(0.5);
    let test_frac = *rng.pick(&[0.05, 0.1, 0.25, 0.3, 0.5, 0.9]);
    let gap = *rng.pick(&[0usize, 1, 2, 5, 40]);
    let vfrac = if mode == SplitMode::TemporalIntervals || rng.bool(0.5) { 0.0 } else { *rng.pick(&[0.1, 0.2, 0.5]) };
    let folds = *rng.pick(&[1usize, 2, 3, 7]);
    let hours: &[i64] = if mode == SplitMode::TemporalIntervals { &PURGE_HOURS[..3] } else { &PURGE_HOURS };
    let timed = matches!(mode, SplitMode::Temporal | SplitMode::TemporalGroup | SplitMode::TemporalIntervals);
    let purge = if !timed {
        None
    } else {
        match policy {
            PurgePolicy::Never => None,
            PurgePolicy::Always => Some(*rng.pick(hours) as f64 * 3600.0),
            PurgePolicy::Mixed => rng.bool(0.5).then(|| *rng.pick(hours) as f64 * 3600.0),
        }
    };
    let (n, stamp, group, iv);
    if mode == SplitMode::TemporalIntervals {
        let (split, span) = random_intervals(rng);
        n = 160 + rng.usize_below(81);
        let step = (span / n as i64).max(1);
        let (s, g) = layout(rng, n, shape, groups, step, 1, (2 * step) as usize);
        stamp = s;
        group = g;
        iv = Some(split);
    } else {
        n = if rng.bool(0.1) { 2 + rng.usize_below(5) } else { 2 + rng.usize_below(239) };
        let (s, g) = layout(rng, n, shape, groups, H, H, 200);
        stamp = s;
        group = g;
        iv = None;
    }
    let table = build_table(&stamp, &group, rng, shuffled);
    (
        Case { mode, test_frac, gap, vfrac, folds, purge, iv },
        Fixture { n, stamp, group, shape, shuffled, table },
    )
}

fn spec_of(c: &Case) -> SplitSpec {
    SplitSpec {
        mode: c.mode,
        time_col: Some(0),
        group_col: Some(1),
        test_frac: c.test_frac,
        gap: c.gap,
        validation_frac: c.vfrac,
        folds: c.folds,
        purge_seconds: c.purge,
        intervals: c.iv.clone(),
        ..Default::default()
    }
}

fn prepare(c: &Case, fx: &Fixture) -> Result<data::Prepared, String> {
    data::prepare_full(&fx.table, &["x".to_string()], "y", &spec_of(c), &mut Mt::new(7))
}

fn describe(c: &Case, fx: &Fixture) -> String {
    format!(
        "{:?} n={} shape={:?} shuffled={} test_frac={} gap={} vfrac={} folds={} purge={:?} iv={:?}",
        c.mode,
        fx.n,
        fx.shape,
        fx.shuffled,
        c.test_frac,
        c.gap,
        c.vfrac,
        c.folds,
        c.purge,
        c.iv.as_ref().map(|i| (&i.declared, i.purge, i.embargo))
    )
}

struct Stats {
    ok: usize,
    errors: std::collections::BTreeMap<String, usize>,
}

impl Stats {
    fn has_error(&self, needle: &str) -> bool {
        self.errors.keys().any(|k| k.contains(needle))
    }

    fn assert_floor(&self, mode: SplitMode) {
        assert!(self.ok >= OK_FLOOR, "{:?}: {} of {} cases ok, errors {:?}", mode, self.ok, CASES, self.errors);
    }
}

/// Runs `CASES` generated cases of one mode and calls `check` on every `Ok` result.
fn drive(mode: SplitMode, seed: u32, policy: PurgePolicy, mut check: impl FnMut(&Case, &Fixture, &data::Prepared)) -> Stats {
    let mut rng = Mt::new(seed);
    let mut stats = Stats { ok: 0, errors: Default::default() };
    for _ in 0..CASES {
        let (case, fx) = gen_case(&mut rng, mode, policy);
        match prepare(&case, &fx) {
            Ok(p) => {
                stats.ok += 1;
                check(&case, &fx, &p);
            }
            Err(e) => {
                let key: String = e.chars().map(|c| if c.is_ascii_digit() { '#' } else { c }).take(90).collect();
                *stats.errors.entry(key).or_insert(0) += 1;
            }
        }
    }
    stats
}

struct Roles {
    train: Vec<usize>,
    validation: Vec<usize>,
    test: Vec<usize>,
    folds: Vec<Vec<usize>>,
    discarded: Vec<usize>,
}

fn ids(y: &[f64]) -> Vec<usize> {
    y.iter().map(|&v| v as usize).collect()
}

fn roles(p: &data::Prepared, n: usize) -> Roles {
    let train = ids(&p.train.y);
    let validation = p.validation.as_ref().map(|s| ids(&s.y)).unwrap_or_default();
    let test = ids(&p.test.y);
    let folds: Vec<Vec<usize>> = p.fold_samples.iter().map(|s| ids(&s.y)).collect();
    let mut seen = vec![false; n];
    for &i in train.iter().chain(&validation).chain(&test) {
        if i < n {
            seen[i] = true;
        }
    }
    let discarded = (0..n).filter(|&i| !seen[i]).collect();
    Roles { train, validation, test, folds, discarded }
}

/// Rows between the last training row and the first test row: the documented rule
/// of `docs/data-splitting.md`.
fn embargo(m: usize, test_frac: f64, gap: usize) -> usize {
    if m < 2 {
        return 0;
    }
    let test_n = (((m as f64) * test_frac).round() as usize).clamp(1, m - 1);
    let train_n = m - test_n;
    (train_n + gap).min(m - 1) - train_n
}

fn purge_count(p: &data::Prepared) -> usize {
    p.report.purged_test_boundary_rows + p.report.purged_validation_boundary_rows
}

fn expected_discarded(c: &Case, fx: &Fixture, p: &data::Prepared) -> usize {
    let r = &p.report;
    match c.mode {
        SplitMode::Random | SplitMode::Group => r.purged_validation_rows,
        SplitMode::Temporal => embargo(fx.n, c.test_frac, c.gap) + r.purged_validation_rows + purge_count(p),
        SplitMode::TemporalGroup => {
            let mut counts = std::collections::BTreeMap::new();
            for &g in &fx.group {
                *counts.entry(g).or_insert(0usize) += 1;
            }
            counts.values().map(|&m| embargo(m, c.test_frac, c.gap)).sum::<usize>() + r.purged_validation_rows + purge_count(p)
        }
        SplitMode::TemporalIntervals => r.interval_dropped_rows + r.interval_outside_rows + purge_count(p),
    }
}

#[test]
fn every_split_mode_gives_disjoint_roles_inside_the_table() {
    for (k, &mode) in MODES.iter().enumerate() {
        let stats = drive(mode, 0x1461_0000 + k as u32, PurgePolicy::Mixed, |c, fx, p| {
            let ctx = describe(c, fx);
            let r = roles(p, fx.n);
            let mut count = vec![0u8; fx.n];
            for (name, v) in [("train", &r.train), ("validation", &r.validation), ("test", &r.test)] {
                for &i in v {
                    assert!(i < fx.n, "{ctx}: {name} id {i} outside the table");
                    count[i] += 1;
                    assert!(count[i] == 1, "{ctx}: row {i} in two roles or twice ({name})");
                }
            }
            assert!(!r.train.is_empty(), "{ctx}: no training rows");
            assert_eq!(p.report.rows_total, fx.n, "{ctx}");
            assert_eq!(p.report.train_rows, r.train.len(), "{ctx}");
            assert_eq!(p.report.validation_rows, r.validation.len(), "{ctx}");
            assert_eq!(p.report.test_rows, r.test.len(), "{ctx}");
        });
        stats.assert_floor(mode);
    }
}

#[derive(Default)]
struct Edges {
    n2: usize,
    n3: usize,
    gap_beyond_block: usize,
    single_row_group: usize,
    single_group: usize,
    no_validation_slice: usize,
    tied_validation_cut: usize,
    interval_purged_before_test: usize,
    interval_embargoed_after_test: usize,
    interval_outside: usize,
    shuffled: usize,
    purge_removed_everything: usize,
}

#[test]
fn every_split_mode_accounts_for_every_row() {
    let mut e = Edges::default();
    for (k, &mode) in MODES.iter().enumerate() {
        let stats = drive(mode, 0x1461_1000 + k as u32, PurgePolicy::Mixed, |c, fx, p| {
            let ctx = describe(c, fx);
            let r = roles(p, fx.n);
            let want = expected_discarded(c, fx, p);
            assert_eq!(
                r.train.len() + r.validation.len() + r.test.len() + r.discarded.len(),
                fx.n,
                "{ctx}"
            );
            assert_eq!(r.discarded.len(), want, "{ctx}: discarded rows");

            if mode == SplitMode::Temporal && c.purge.is_none() && fx.shape == Shape::Unique {
                let m = fx.n - (((fx.n as f64) * c.test_frac).round() as usize).clamp(1, fx.n - 1);
                let gv = p.report.purged_validation_rows;
                let fit = r.train.len();
                assert_eq!(m, fit + gv + r.validation.len(), "{ctx}");
                let emb = embargo(fx.n, c.test_frac, c.gap);
                let mut tr = r.train.clone();
                tr.sort_unstable();
                assert_eq!(tr, (0..fit).collect::<Vec<_>>(), "{ctx}: fit rows");
                let mut va = r.validation.clone();
                va.sort_unstable();
                assert_eq!(va, (fit + gv..m).collect::<Vec<_>>(), "{ctx}: validation rows");
                let mut disc: Vec<usize> = (fit..fit + gv).collect();
                disc.extend(m..m + emb);
                assert_eq!(r.discarded, disc, "{ctx}: discarded rows");
                assert_eq!(r.test, (m + emb..fx.n).collect::<Vec<_>>(), "{ctx}: test rows");
            }

            e.n2 += (fx.n == 2) as usize;
            e.n3 += (fx.n == 3) as usize;
            e.shuffled += fx.shuffled as usize;
            let notes = p.report.notes.join("\n");
            if mode == SplitMode::Temporal && fx.n >= 2 {
                let test_n = (((fx.n as f64) * c.test_frac).round() as usize).clamp(1, fx.n - 1);
                let m = fx.n - test_n;
                e.gap_beyond_block += (m + c.gap > fx.n - 1) as usize;
                if c.vfrac > 0.0 && m >= 2 {
                    let want = (((m as f64) * c.vfrac).round() as usize).clamp(1, m - 1);
                    let cut = m - want;
                    e.tied_validation_cut += (cut > 0 && fx.stamp[cut - 1] == fx.stamp[cut]) as usize;
                }
            }
            e.single_row_group += notes.contains("too few rows for a temporal split") as usize;
            e.single_group += notes.contains("only one group") as usize;
            e.no_validation_slice += notes.contains("fewer than 2 rows") as usize;
            if let Some(iv) = &c.iv {
                let d = &iv.declared;
                let inside = |v: &[(i64, i64)], t: i64| v.iter().any(|&(a, b)| a <= t && t < b);
                for &i in &r.discarded {
                    let t = fx.stamp[i];
                    if inside(&d.train, t) {
                        e.interval_purged_before_test += d.test.iter().any(|&(s, _)| t < s && t + iv.purge > s) as usize;
                        e.interval_embargoed_after_test += d.test.iter().any(|&(_, en)| t >= en && t < en + iv.embargo) as usize;
                    } else if !inside(&d.validation, t) {
                        e.interval_outside += 1;
                    }
                }
            }
        });
        stats.assert_floor(mode);
        e.purge_removed_everything += stats.has_error("after purging") as usize;
    }
    assert!(e.n2 > 0 && e.n3 > 0, "n=2: {}, n=3: {}", e.n2, e.n3);
    assert!(e.gap_beyond_block > 0, "no gap beyond the test block");
    assert!(e.single_row_group > 0, "no one-row temporal group");
    assert!(e.single_group > 0, "no one-group table");
    assert!(e.no_validation_slice > 0, "no validation fraction on a one-row training part");
    assert!(e.tied_validation_cut > 0, "no tied stamp at the validation cut");
    assert!(e.interval_purged_before_test > 0, "no interval row purged before a test interval");
    assert!(e.interval_embargoed_after_test > 0, "no interval row embargoed after a test interval");
    assert!(e.interval_outside > 0, "no row outside every interval");
    assert!(e.shuffled > 0, "no shuffled file order");
    assert!(e.purge_removed_everything > 0, "no purge removed every training row");
}

#[test]
fn folds_partition_the_test_rows_in_every_mode() {
    for (k, &mode) in MODES.iter().enumerate() {
        let stats = drive(mode, 0x1461_2000 + k as u32, PurgePolicy::Mixed, |c, fx, p| {
            let ctx = describe(c, fx);
            let r = roles(p, fx.n);
            if c.folds == 1 || r.test.len() < 2 {
                assert!(p.fold_samples.is_empty(), "{ctx}: fold samples without several folds");
                return;
            }
            assert!(p.fold_samples.is_empty() || p.fold_samples.len() >= 2, "{ctx}");
            assert!(p.fold_samples.len() <= c.folds, "{ctx}");
            if mode != SplitMode::TemporalGroup {
                assert_eq!(p.fold_samples.len(), c.folds.min(r.test.len()), "{ctx}: fold count");
            }
            if p.fold_samples.is_empty() {
                return;
            }
            let mut seen = vec![false; fx.n];
            for (j, f) in r.folds.iter().enumerate() {
                assert!(!f.is_empty(), "{ctx}: fold {j} is empty");
                for &i in f {
                    assert!(!seen[i], "{ctx}: row {i} in two folds");
                    seen[i] = true;
                }
            }
            let joined: Vec<usize> = r.folds.iter().flatten().copied().collect();
            assert_eq!(joined, r.test, "{ctx}: folds in order are the test rows");
        });
        stats.assert_floor(mode);
    }
}

fn stamps_of(fx: &Fixture, v: &[usize]) -> Vec<i64> {
    v.iter().map(|&i| fx.stamp[i]).collect()
}

fn max_of(v: &[i64]) -> Option<i64> {
    v.iter().copied().max()
}

fn min_of(v: &[i64]) -> Option<i64> {
    v.iter().copied().min()
}

#[test]
fn temporal_modes_never_train_after_the_test_block() {
    for (k, &mode) in [SplitMode::Temporal, SplitMode::TemporalGroup, SplitMode::Group].iter().enumerate() {
        let stats = drive(mode, 0x1461_3000 + k as u32, PurgePolicy::Mixed, |c, fx, p| {
            let ctx = describe(c, fx);
            let r = roles(p, fx.n);
            match mode {
                SplitMode::Temporal => {
                    let tr = stamps_of(fx, &r.train);
                    let va = stamps_of(fx, &r.validation);
                    let te = stamps_of(fx, &r.test);
                    let (tr_max, te_min) = (max_of(&tr).unwrap(), min_of(&te).unwrap());
                    assert!(tr_max <= te_min, "{ctx}: training after the test block");
                    if fx.shape != Shape::Tied {
                        assert!(tr_max < te_min, "{ctx}: unique stamps must be strictly ordered");
                    }
                    if let Some(va_min) = min_of(&va) {
                        assert!(tr_max < va_min, "{ctx}: a stamp straddles fit and validation");
                        assert!(max_of(&va).unwrap() <= te_min, "{ctx}: validation after the test block");
                    }
                }
                SplitMode::TemporalGroup => {
                    let all = stamps_of(fx, &r.train);
                    let va = stamps_of(fx, &r.validation);
                    if let (Some(a), Some(b)) = (max_of(&all), min_of(&va)) {
                        assert!(a < b, "{ctx}: a stamp straddles fit and validation");
                    }
                    for g in 0..=*fx.group.iter().max().unwrap() {
                        let of = |v: &[usize]| -> Vec<i64> { v.iter().filter(|&&i| fx.group[i] == g).map(|&i| fx.stamp[i]).collect() };
                        let (tr, va, te) = (of(&r.train), of(&r.validation), of(&r.test));
                        if let (Some(a), Some(b)) = (max_of(&tr), min_of(&te)) {
                            assert!(a < b, "{ctx}: group {g} trains after its test block");
                        }
                        if let (Some(a), Some(b)) = (max_of(&va), min_of(&te)) {
                            assert!(a < b, "{ctx}: group {g} validates after its test block");
                        }
                    }
                }
                _ => {
                    let test_groups: std::collections::BTreeSet<usize> = r.test.iter().map(|&i| fx.group[i]).collect();
                    for &i in r.train.iter().chain(&r.validation) {
                        assert!(!test_groups.contains(&fx.group[i]), "{ctx}: group {} on both sides", fx.group[i]);
                    }
                }
            }
        });
        stats.assert_floor(mode);
    }
}

fn distance_to(t: i64, sorted: &[i64]) -> i64 {
    sorted.iter().map(|&s| (s - t).abs()).min().unwrap_or(i64::MAX)
}

#[test]
fn purge_keeps_every_training_row_outside_the_label_distance() {
    let timed = [SplitMode::Temporal, SplitMode::TemporalGroup, SplitMode::TemporalIntervals];
    for (k, &mode) in timed.iter().enumerate() {
        let mut positive = 0usize;
        let stats = drive(mode, 0x1461_4000 + k as u32, PurgePolicy::Always, |c, fx, p| {
            let ctx = describe(c, fx);
            let l = c.purge.unwrap() as i64;
            let r = roles(p, fx.n);
            let tr = stamps_of(fx, &r.train);
            let va = stamps_of(fx, &r.validation);
            let te = stamps_of(fx, &r.test);
            for &t in tr.iter().chain(&va) {
                assert!(distance_to(t, &te) >= l, "{ctx}: stamp {t} within {l} s of a test stamp");
            }
            for &t in &tr {
                assert!(distance_to(t, &va) >= l, "{ctx}: fit stamp {t} within {l} s of a validation stamp");
            }
            if te.is_empty() {
                assert_eq!(p.report.purge_seconds, None, "{ctx}");
            } else {
                let nearest = tr.iter().map(|&t| distance_to(t, &te)).min().unwrap();
                let reported = p.report.purge_seconds.unwrap_or_else(|| panic!("{ctx}: purge_seconds is not recorded"));
                assert!(reported >= l as f64 && reported == nearest as f64, "{ctx}: reported distance {reported}, measured {nearest}");
            }

            let open = Case { mode: c.mode, test_frac: c.test_frac, gap: c.gap, vfrac: c.vfrac, folds: c.folds, purge: None, iv: c.iv.clone() };
            let base = prepare(&open, fx).expect("the same case without a purge is ok");
            let removed = base.train.y.len() + base.validation.as_ref().map_or(0, |s| s.y.len()) - r.train.len() - r.validation.len();
            assert_eq!(removed, purge_count(p), "{ctx}: purge counters");
            positive += (removed > 0) as usize;
        });
        stats.assert_floor(mode);
        assert!(positive > 0, "{:?}: the purge never removed a row", mode);
    }
}

#[test]
fn interval_roles_follow_the_split_table() {
    let mut before = 0usize;
    let mut after = 0usize;
    let mut outside = 0usize;
    let stats = drive(SplitMode::TemporalIntervals, 0x1461_5000, PurgePolicy::Never, |c, fx, p| {
        let ctx = describe(c, fx);
        let iv = c.iv.as_ref().unwrap();
        let d = &iv.declared;
        let r = roles(p, fx.n);
        let mut role_of: Vec<Option<Role>> = vec![None; fx.n];
        for (role, v) in [(Role::Train, &r.train), (Role::Validation, &r.validation), (Role::Test, &r.test)] {
            for &i in v {
                role_of[i] = Some(role);
            }
        }
        let inside = |v: &[(i64, i64)], t: i64| v.iter().any(|&(a, b)| a <= t && t < b);
        let (mut dropped, mut out) = (0usize, 0usize);
        for i in 0..fx.n {
            let t = fx.stamp[i];
            assert_eq!(role_of[i], iv.role(t), "{ctx}: row {i} at {t}");
            if role_of[i].is_none() {
                if inside(&d.train, t) || inside(&d.validation, t) {
                    dropped += 1;
                    if inside(&d.train, t) {
                        before += d.test.iter().any(|&(s, _)| t < s && t + iv.purge > s) as usize;
                        after += d.test.iter().any(|&(_, e)| t >= e && t < e + iv.embargo) as usize;
                    }
                } else {
                    out += 1;
                }
            }
        }
        outside += out;
        assert_eq!(p.report.interval_dropped_rows, dropped, "{ctx}");
        assert_eq!(p.report.interval_outside_rows, out, "{ctx}");
    });
    stats.assert_floor(SplitMode::TemporalIntervals);
    assert!(before > 0 && after > 0 && outside > 0, "purged before {before}, embargoed after {after}, outside {outside}");
}

#[test]
fn temporal_cut_on_a_tied_stamp_is_pinned() {
    let s = BASE;
    let stamps = [s, s + H, s + 2 * H, s + 2 * H, s + 3 * H, s + 4 * H];
    let mut csv = String::from("t,x,y\n");
    for (id, t) in stamps.iter().enumerate() {
        csv.push_str(&format!("{},{},{}\n", t, (0.37 * id as f64).sin(), id));
    }
    let table = data::parse_table(&csv).unwrap();
    let run = |purge: Option<f64>| {
        let mut sp = spec(SplitMode::Temporal);
        sp.time_col = Some(0);
        sp.test_frac = 0.5;
        sp.purge_seconds = purge;
        data::prepare_full(&table, &["x".to_string()], "y", &sp, &mut Mt::new(1)).unwrap()
    };
    let open = run(None);
    assert_eq!(ids(&open.train.y), vec![0, 1, 2], "the stamp s+2h is on both sides of the cut");
    assert_eq!(ids(&open.test.y), vec![3, 4, 5]);
    let purged = run(Some(1.0));
    assert_eq!(ids(&purged.train.y), vec![0, 1]);
    assert_eq!(ids(&purged.test.y), vec![3, 4, 5]);
    assert_eq!(purged.report.purged_test_boundary_rows, 1);
}

#[test]
fn a_kept_row_never_reaches_a_declared_locked_range() {
    let mut rng = Mt::new(0x1461_6000);
    let mut ran = [0usize; 3];
    for case in 0..300 {
        let kind = case % 3;
        let mode = if (case / 3) % 2 == 0 { SplitMode::Temporal } else { SplitMode::TemporalGroup };
        let groups = 1 + rng.usize_below(3);
        let exact = rng.bool(0.5);
        let extra = |rng: &mut Mt| if exact { 0 } else { (1 + rng.usize_below(5)) as i64 * H };
        let mut stamp: Vec<i64>;
        let start: i64;
        let end: i64;
        match kind {
            2 => {
                let m = 30 + rng.usize_below(40);
                let k = 1 + rng.usize_below(3);
                let r = (m + 1) / 2;
                let l = m - k - r;
                let len = (24 + rng.usize_below(100)) as i64 * H;
                start = BASE + (l as i64 + 2) * H;
                let before = extra(&mut rng);
                end = start + len;
                stamp = (0..l as i64).map(|i| start - before - (l as i64 - i) * H).collect::<Vec<_>>();
                stamp.extend((0..k as i64).map(|j| start - 900 * (k as i64 - j)));
                let first = end + H + extra(&mut rng);
                stamp.extend((0..r as i64).map(|j| first + j * H));
            }
            _ => {
                let l = 6 + rng.usize_below(25);
                let r = 6 + rng.usize_below(25);
                let len = (24 + rng.usize_below(100)) as i64 * H;
                start = BASE + (l as i64 + 2) * H;
                let before = extra(&mut rng);
                end = start + len;
                stamp = (0..l as i64).map(|i| start - before - (l as i64 - i) * H).collect::<Vec<_>>();
                if kind == 1 {
                    let k = 1 + rng.usize_below(4) as i64;
                    stamp.extend((1..=k).map(|j| start + j * len / (k + 1)));
                }
                let first = end + H + extra(&mut rng);
                stamp.extend((0..r as i64).map(|j| first + j * H));
            }
        }
        let group: Vec<usize> = (0..stamp.len()).map(|i| i % groups).collect();
        let shuffled = rng.bool(0.5);
        let table = build_table(&stamp, &group, &mut rng, shuffled);
        let mut sp = spec(mode);
        sp.time_col = Some(0);
        sp.group_col = Some(1);
        sp.boundary_purge_seconds = Some(3600.0);
        sp.declared_lockbox = Some(LockDeclared {
            lockbox: Lockbox { start, end },
            source: "flag",
            also: None,
            embargo: Embargo::none(),
        });
        let ctx = format!("kind {kind} {mode:?} start {start} end {end} stamps {stamp:?}");
        let res = data::prepare_full(&table, &["x".to_string()], "y", &sp, &mut Mt::new(3));
        match (kind, res) {
            (0, Ok(p)) => {
                let k = p.keys.as_ref().expect("time keys");
                let lb = Lockbox { start, end };
                for &t in k.train.iter().chain(k.validation.iter().flatten()).chain(&k.test) {
                    assert!(!lb.reaches(t as i64, 3600, 3600), "{ctx}: kept stamp {t} reaches the range");
                }
            }
            (0, Err(e)) => panic!("{ctx}: refused: {e}"),
            (1, Err(e)) => assert!(e.contains("lock"), "{ctx}: {e}"),
            (2, Err(e)) => {
                assert!(e.contains("declared locked test"), "{ctx}: {e}");
                assert!(e.starts_with("training split has"), "{ctx}: {e}");
            }
            (_, Ok(_)) => panic!("{ctx}: a row that reaches the range was kept"),
            _ => unreachable!(),
        }
        ran[kind] += 1;
    }
    assert_eq!(ran, [100, 100, 100]);
}
