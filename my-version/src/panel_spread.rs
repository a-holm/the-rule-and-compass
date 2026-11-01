//! #261: `rcsr panel spread`, the market-neutral bucket spread of a feature.
//!
//! Rows of a half-open `--range` are bucketed by a feature (pooled over the
//! window or per timestamp), the forward label is optionally demeaned within
//! each timestamp, and the spread is the mean label of the lowest bucket minus
//! the mean label of the highest bucket. The interval is a circular block
//! bootstrap over timestamps. The command refuses a range whose label windows
//! reach a declared locked test.

use crate::devlog;
use crate::inference;
use crate::lockbox::{self, Lockbox};
use crate::rng::Mt;
use crate::store::FileStore;
use crate::target;

#[derive(Clone)]
pub struct SpreadOptions {
    pub buckets: usize,
    pub per_timestamp_buckets: bool,
    pub demean: bool,
    pub min_members: usize,
    /// Block length in timestamps; `None` takes the label horizon or `ceil(T^(1/3))`.
    pub block: Option<usize>,
    pub horizon_periods: usize,
    pub draws: usize,
    pub seed: u32,
    pub level: f64,
}

#[derive(Debug)]
pub struct SpreadResult {
    pub timestamps: usize,
    pub thin_timestamps: usize,
    pub rows: usize,
    pub spread: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
    pub se: f64,
    pub block: usize,
    pub bucket_means: Vec<f64>,
    pub bucket_rows: Vec<usize>,
    /// Per-timestamp spread of the lowest minus the highest bucket, `NaN` when a bucket is empty.
    pub hourly_spread: Vec<f64>,
}


fn bucket_of(edges: &[f64], v: f64) -> usize {
    edges.partition_point(|e| *e <= v)
}

/// `k - 1` cut points at the sorted positions `j * n / k`; a value goes to the
/// number of cut points at or below it, so tied values share a bucket.
fn edges_of(sorted: &[f64], k: usize) -> Vec<f64> {
    (1..k).map(|j| sorted[j * sorted.len() / k]).collect()
}

/// Spread of the lowest minus the highest bucket of `feature` over the rows,
/// in the units of `label`. Rows with a non-finite feature or label are
/// dropped, then timestamps with fewer than `max(min_members, 2)` rows.
pub fn spread(feature: &[f64], label: &[f64], time: &[i64], o: &SpreadOptions) -> Result<SpreadResult, String> {
    if o.buckets < 2 {
        return Err("--buckets must be at least 2".to_string());
    }
    let n = feature.len().min(label.len()).min(time.len());
    let mut order: Vec<usize> = (0..n).filter(|&i| feature[i].is_finite() && label[i].is_finite()).collect();
    order.sort_by_key(|&i| (time[i], i));
    let need = o.min_members.max(2);
    let mut groups: Vec<&[usize]> = Vec::new();
    let mut thin = 0usize;
    let mut s = 0usize;
    while s < order.len() {
        let mut e = s + 1;
        while e < order.len() && time[order[e]] == time[order[s]] {
            e += 1;
        }
        if e - s < need {
            thin += 1;
        } else {
            groups.push(&order[s..e]);
        }
        s = e;
    }
    let t = groups.len();
    if t == 0 {
        return Err("no timestamp has enough finite rows".to_string());
    }
    let k = o.buckets;
    let pooled_edges = if o.per_timestamp_buckets {
        Vec::new()
    } else {
        let mut all: Vec<f64> = groups.iter().flat_map(|g| g.iter().map(|&i| feature[i])).collect();
        all.sort_by(|a, b| a.total_cmp(b));
        edges_of(&all, k)
    };
    let mut sum = vec![vec![0.0f64; k]; t];
    let mut cnt = vec![vec![0usize; k]; t];
    for (g, rows) in groups.iter().enumerate() {
        let m = if o.demean { rows.iter().map(|&i| label[i]).sum::<f64>() / rows.len() as f64 } else { 0.0 };
        let own;
        let edges: &[f64] = if o.per_timestamp_buckets {
            let mut v: Vec<f64> = rows.iter().map(|&i| feature[i]).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            own = edges_of(&v, k);
            &own
        } else {
            &pooled_edges
        };
        for &i in rows.iter() {
            let b = bucket_of(edges, feature[i]);
            sum[g][b] += label[i] - m;
            cnt[g][b] += 1;
        }
    }
    let bucket_rows: Vec<usize> = (0..k).map(|b| cnt.iter().map(|c| c[b]).sum()).collect();
    if bucket_rows[0] == 0 || bucket_rows[k - 1] == 0 {
        return Err("the lowest or highest bucket is empty: the feature has too many tied values for --buckets".to_string());
    }
    let bucket_means: Vec<f64> = (0..k)
        .map(|b| if bucket_rows[b] == 0 { f64::NAN } else { sum.iter().map(|s| s[b]).sum::<f64>() / bucket_rows[b] as f64 })
        .collect();
    let point = bucket_means[0] - bucket_means[k - 1];
    let hourly_spread: Vec<f64> = (0..t)
        .map(|g| {
            if cnt[g][0] == 0 || cnt[g][k - 1] == 0 {
                f64::NAN
            } else {
                sum[g][0] / cnt[g][0] as f64 - sum[g][k - 1] / cnt[g][k - 1] as f64
            }
        })
        .collect();

    let block = o.block.unwrap_or_else(|| inference::auto_block_len(t, o.horizon_periods, 1)).clamp(1, t);
    let mut rng = Mt::new(o.seed);
    let mut draws: Vec<f64> = Vec::with_capacity(o.draws);
    for _ in 0..o.draws {
        let (mut s0, mut n0, mut s1, mut n1) = (0.0f64, 0usize, 0.0f64, 0usize);
        let mut remaining = t;
        while remaining > 0 {
            let len = block.min(remaining);
            let start = rng.usize_below(t);
            for j in 0..len {
                let g = (start + j) % t;
                s0 += sum[g][0];
                n0 += cnt[g][0];
                s1 += sum[g][k - 1];
                n1 += cnt[g][k - 1];
            }
            remaining -= len;
        }
        if n0 > 0 && n1 > 0 {
            draws.push(s0 / n0 as f64 - s1 / n1 as f64);
        }
    }
    draws.sort_by(|a, b| a.total_cmp(b));
    let tail = (1.0 - o.level) / 2.0;
    let (ci_lo, ci_hi, se) = if draws.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (inference::quantile_sorted(&draws, tail), inference::quantile_sorted(&draws, 1.0 - tail), inference::sample_sd(&draws))
    };
    Ok(SpreadResult {
        timestamps: t,
        thin_timestamps: thin,
        rows: bucket_rows.iter().sum(),
        spread: point,
        ci_lo,
        ci_hi,
        se,
        block,
        bucket_means,
        bucket_rows,
        hourly_spread,
    })
}

const BP: f64 = 1e4;

pub struct Args {
    pub data: String,
    pub feature: String,
    pub label: String,
    pub range: (i64, i64),
    pub opts: SpreadOptions,
    pub target_spec: Option<String>,
    pub purge_seconds: Option<i64>,
    pub lockbox: Option<String>,
}

/// An error with its exit code: 2 usage, 3 input or lockbox refusal.
pub type Refusal = (i32, String);

fn parse_range(s: &str) -> Result<(i64, i64), String> {
    let (a, b) = s.split_once("..").ok_or_else(|| format!("--range '{}' must be FROM..TO", s))?;
    let from = lockbox::parse_time(a.trim()).ok_or_else(|| format!("--range: bad start '{}'", a))?;
    let to = lockbox::parse_time(b.trim()).ok_or_else(|| format!("--range: bad end '{}'", b))?;
    if to <= from {
        return Err("--range end must be after start".to_string());
    }
    Ok((from, to))
}

fn refuse_range(range: (i64, i64), locks: &[Lockbox], purge: i64) -> Result<(), String> {
    for l in locks {
        let (from, to) = range;
        if from < l.end && to > l.start {
            return Err(format!(
                "--range {} overlaps the declared locked test {}; read it only with rcsr walkforward --locked-test or rcsr ladder --locked-test",
                lockbox::span(&Lockbox { start: from, end: to }),
                lockbox::span(l)
            ));
        }
        if to > l.start - purge && from < l.end {
            return Err(format!(
                "--range {} has rows whose label window ({} s) reaches the declared locked test {}; end the range at {} or earlier",
                lockbox::span(&Lockbox { start: from, end: to }),
                purge,
                lockbox::span(l),
                crate::protocol::iso((l.start - purge) as f64)
            ));
        }
    }
    Ok(())
}

pub fn run(a: &Args) -> Result<String, Refusal> {
    let resolved = lockbox::resolve(a.lockbox.as_deref(), &a.data).map_err(|e| (e.exit, e.message))?;
    let spec = match &a.target_spec {
        Some(p) => Some(p.clone()),
        None => target::sidecar_path(&a.data),
    }
    .and_then(|p| std::fs::read_to_string(&p).ok())
    .and_then(|t| target::parse_file(&t).ok())
    .and_then(|specs| {
        specs
            .into_iter()
            .find(|s| s.return_column.as_deref() == Some(a.label.as_str()) || s.column == a.label)
    });
    let purge = match (a.purge_seconds, &spec) {
        (Some(p), _) => p,
        (None, Some(s)) => s.required_purge_seconds().ceil() as i64,
        (None, None) => {
            return Err((3, format!("no target spec entry for '{}': pass --purge-seconds (the label window length)", a.label)));
        }
    };
    let horizon = spec.as_ref().map_or(purge as f64, |s| s.horizon_seconds);
    refuse_range(a.range, &resolved.ranges(), purge).map_err(|e| (3, e))?;

    let store = FileStore::open(&a.data).map_err(|e| (3, e))?;
    let ts = crate::panel_rank::read_ts(&store).map_err(|e| (3, e))?;
    let sel: Vec<usize> = (0..ts.len()).filter(|&i| ts[i] >= a.range.0 && ts[i] < a.range.1).collect();
    let times: Vec<i64> = sel.iter().map(|&i| ts[i]).collect();
    if let Some(d) = resolved.declared() {
        d.check_times("--range", "ts", &times, purge).map_err(|e| (3, e))?;
    }
    if sel.is_empty() {
        return Err((3, "--range selects no rows".to_string()));
    }
    let take = |name: &str| -> Result<Vec<f64>, Refusal> {
        let col = crate::panel_rank::read_column(&store, name).map_err(|e| (3, e))?;
        Ok(sel.iter().map(|&i| col[i]).collect())
    };
    let feature = take(&a.feature)?;
    let label = take(&a.label)?;
    let mut o = a.opts.clone();
    let mut uniq = times.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let step = uniq.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0).min().unwrap_or(1);
    o.horizon_periods = ((horizon / step as f64).ceil() as usize).max(1);
    let r = spread(&feature, &label, &times, &o).map_err(|e| (3, e))?;
    let positive = r.hourly_spread.iter().filter(|v| **v > 0.0).count();
    let finite = r.hourly_spread.iter().filter(|v| v.is_finite()).count();
    let list = |v: &[f64]| v.iter().map(|x| format!("{:.9}", x * BP)).collect::<Vec<_>>().join(",");
    let locks = resolved
        .ranges()
        .iter()
        .map(|l| format!("{{\"start\":{},\"end\":{}}}", l.start, l.end))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"schema\":\"rcsr-spread/1\",\"data\":{},\"feature\":{},\"label\":{},\"buckets\":{},\"bucketing\":{},\"demean\":{},\
\"range\":{{\"start\":{},\"end\":{},\"start_iso\":{},\"end_iso\":{}}},\"purge_seconds\":{},\
\"lockbox\":{{\"source\":{},\"ranges\":[{}],\"rows_in_range_overlap\":0}},\
\"timestamps\":{},\"thin_timestamps\":{},\"rows\":{},\"block\":{},\"draws\":{},\"seed\":{},\"level\":{},\
\"spread_bp\":{:.9},\"ci_lo_bp\":{:.9},\"ci_hi_bp\":{:.9},\"se_bp\":{:.9},\
\"bucket_means_bp\":[{}],\"bucket_rows\":[{}],\"positive_timestamps\":{},\"finite_timestamps\":{}}}",
        devlog::text(&a.data),
        devlog::text(&a.feature),
        devlog::text(&a.label),
        o.buckets,
        devlog::text(if o.per_timestamp_buckets { "cross-section" } else { "pooled" }),
        devlog::text(if o.demean { "cross-section" } else { "none" }),
        a.range.0,
        a.range.1,
        devlog::text(&crate::protocol::iso(a.range.0 as f64)),
        devlog::text(&crate::protocol::iso(a.range.1 as f64)),
        purge,
        devlog::text(resolved.source.as_str()),
        locks,
        r.timestamps,
        r.thin_timestamps,
        r.rows,
        r.block,
        o.draws,
        o.seed,
        o.level,
        r.spread * BP,
        r.ci_lo * BP,
        r.ci_hi * BP,
        r.se * BP,
        list(&r.bucket_means),
        r.bucket_rows.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","),
        positive,
        finite
    ))
}

pub fn usage_text() -> &'static str {
    "rcsr panel spread --data panel.rcsb --feature clv1 --label lbl_h1_fwd --range FROM..TO [--buckets 5]
                   [--bucketing pooled|cross-section] [--demean cross-section|none] [--min-members 20]
                   [--block N] [--draws 2000] [--seed 1] [--level 0.95]
                   [--target-spec FILE] [--purge-seconds S] [--lockbox S,E|none]
  Market-neutral bucket spread of a feature. Rows with ts in the half-open --range (epoch seconds or
  ISO dates) are put in --buckets quantile buckets of --feature: pooled over the range (default) or
  per timestamp. --demean cross-section (default) subtracts the per-timestamp mean of --label first.
  The spread is mean(label | lowest bucket) - mean(label | highest bucket), reported in bp
  (label as a return fraction, x 1e4). The interval is a circular block bootstrap over timestamps
  (--block default: the larger of the label horizon in timestamps and ceil(T^(1/3)); percentile interval at --level).
  Rows with a non-finite feature or label are dropped, then timestamps with fewer than --min-members rows.
  Ties share a bucket; an empty lowest or highest bucket is an error.
  Prints one JSON object on stdout. Refuses (exit 3) a --range that overlaps the declared locked test
  or whose label windows (--purge-seconds, default from the target spec of --label: entry offset +
  horizon) reach it. --lockbox none is the only opt-out, and only for undeclared data.
"
}

fn fail(msg: &str, code: i32) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(code);
}

fn value(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| fail(&format!("{} requires a value", args[*i - 1]), 2))
}

fn num<T: std::str::FromStr>(flag: &str, v: String) -> T {
    v.parse().unwrap_or_else(|_| fail(&format!("{}: cannot parse '{}'", flag, v), 2))
}

pub fn cli(args: &[String]) -> ! {
    let (mut data, mut feature, mut label, mut range) = (None, None, None, None);
    let mut opts = SpreadOptions {
        buckets: 5,
        per_timestamp_buckets: false,
        demean: true,
        min_members: 20,
        block: None,
        horizon_periods: 1,
        draws: 2000,
        seed: 1,
        level: 0.95,
    };
    let (mut target_spec, mut purge_seconds, mut lockbox) = (None, None, None);
    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].clone();
        match flag.as_str() {
            "--data" => data = Some(value(args, &mut i)),
            "--feature" => feature = Some(value(args, &mut i)),
            "--label" => label = Some(value(args, &mut i)),
            "--range" => range = Some(value(args, &mut i)),
            "--buckets" => opts.buckets = num(&flag, value(args, &mut i)),
            "--bucketing" => {
                opts.per_timestamp_buckets = match value(args, &mut i).as_str() {
                    "pooled" => false,
                    "cross-section" => true,
                    v => fail(&format!("--bucketing: '{}' is not pooled or cross-section", v), 2),
                }
            }
            "--demean" => {
                opts.demean = match value(args, &mut i).as_str() {
                    "cross-section" => true,
                    "none" => false,
                    v => fail(&format!("--demean: '{}' is not cross-section or none", v), 2),
                }
            }
            "--min-members" => opts.min_members = num(&flag, value(args, &mut i)),
            "--block" => opts.block = Some(num(&flag, value(args, &mut i))),
            "--draws" => opts.draws = num(&flag, value(args, &mut i)),
            "--seed" => opts.seed = num(&flag, value(args, &mut i)),
            "--level" => opts.level = num(&flag, value(args, &mut i)),
            "--target-spec" => target_spec = Some(value(args, &mut i)),
            "--purge-seconds" => purge_seconds = Some(num(&flag, value(args, &mut i))),
            "--lockbox" => lockbox = Some(value(args, &mut i)),
            "-h" | "--help" => {
                print!("{}", usage_text());
                std::process::exit(0);
            }
            other => fail(&format!("unknown option '{}' for `rcsr panel spread`", other), 2),
        }
        i += 1;
    }
    if opts.buckets < 2 || opts.draws == 0 || !(opts.level > 0.0 && opts.level < 1.0) {
        fail("--buckets must be at least 2, --draws at least 1 and --level inside (0, 1)", 2);
    }
    let range = parse_range(&range.unwrap_or_else(|| fail("`rcsr panel spread` needs --range", 2))).unwrap_or_else(|e| fail(&e, 2));
    let a = Args {
        data: data.unwrap_or_else(|| fail("`rcsr panel spread` needs --data", 2)),
        feature: feature.unwrap_or_else(|| fail("`rcsr panel spread` needs --feature", 2)),
        label: label.unwrap_or_else(|| fail("`rcsr panel spread` needs --label", 2)),
        range,
        opts,
        target_spec,
        purge_seconds,
        lockbox,
    };
    match run(&a) {
        Ok(json) => {
            println!("{}", json);
            std::process::exit(0);
        }
        Err((code, msg)) => fail(&msg, code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(k: usize, per_ts: bool, demean: bool) -> SpreadOptions {
        SpreadOptions {
            buckets: k,
            per_timestamp_buckets: per_ts,
            demean,
            min_members: 2,
            block: Some(1),
            horizon_periods: 1,
            draws: 200,
            seed: 7,
            level: 0.95,
        }
    }

    // Two timestamps, four members each, shuffled row order.
    // t1 features 1 3 5 7 labels 6 2 0 -4 (mean 1); t2 features 2 4 6 8 labels 10 2 0 -8 (mean 1).
    fn panel() -> (Vec<f64>, Vec<f64>, Vec<i64>) {
        let f = vec![5.0, 2.0, 7.0, 8.0, 1.0, 4.0, 3.0, 6.0];
        let l = vec![0.0, 10.0, -4.0, -8.0, 6.0, 2.0, 2.0, 0.0];
        let t = vec![1, 2, 1, 2, 1, 2, 1, 2];
        (f, l, t)
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12, "{a} != {b}");
    }

    #[test]
    fn pooled_buckets_demeaned_spread_matches_the_hand_table() {
        let (f, l, t) = panel();
        let r = spread(&f, &l, &t, &opts(2, false, true)).unwrap();
        assert_eq!((r.timestamps, r.rows), (2, 8));
        assert_eq!(r.bucket_rows, vec![4, 4]);
        close(r.bucket_means[0], 4.0);
        close(r.bucket_means[1], -4.0);
        close(r.spread, 8.0);
        assert!(r.ci_lo <= r.spread && r.spread <= r.ci_hi);
    }

    #[test]
    fn raw_labels_shift_the_bucket_means_not_the_spread() {
        let (f, l, t) = panel();
        let r = spread(&f, &l, &t, &opts(2, false, false)).unwrap();
        close(r.bucket_means[0], 5.0);
        close(r.bucket_means[1], -3.0);
        close(r.spread, 8.0);
    }

    #[test]
    fn per_timestamp_buckets_give_one_spread_per_timestamp() {
        let (f, l, t) = panel();
        let r = spread(&f, &l, &t, &opts(2, true, true)).unwrap();
        close(r.hourly_spread[0], 6.0);
        close(r.hourly_spread[1], 10.0);
        close(r.spread, 8.0);
    }

    #[test]
    fn same_seed_same_interval_and_an_empty_bucket_is_an_error() {
        let (f, l, t) = panel();
        let a = spread(&f, &l, &t, &opts(2, false, true)).unwrap();
        let b = spread(&f, &l, &t, &opts(2, false, true)).unwrap();
        assert_eq!((a.ci_lo, a.ci_hi), (b.ci_lo, b.ci_hi));
        let same = vec![1.0; 8];
        let e = spread(&same, &l, &t, &opts(2, false, true)).unwrap_err();
        assert!(e.contains("empty"), "{e}");
    }

    #[test]
    fn nan_rows_and_thin_timestamps_are_dropped() {
        let (mut f, l, mut t) = panel();
        f[0] = f64::NAN;
        t.push(3);
        let mut f2 = f.clone();
        f2.push(9.0);
        let mut l2 = l.clone();
        l2.push(100.0);
        let r = spread(&f2, &l2, &t, &opts(2, false, true)).unwrap();
        assert_eq!(r.thin_timestamps, 1);
        assert_eq!(r.rows, 7);
    }
}
