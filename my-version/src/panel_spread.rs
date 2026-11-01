//! #261: `rcsr panel spread`, the market-neutral bucket spread of a feature.
//!
//! Rows of a half-open `--range` are bucketed by a feature (pooled over the
//! window or per timestamp), the forward label is optionally demeaned within
//! each timestamp, and the spread is the mean label of the lowest bucket minus
//! the mean label of the highest bucket. The interval is a circular block
//! bootstrap over timestamps. The command refuses a range whose label windows
//! reach a declared locked test.

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

pub fn spread(_feature: &[f64], _label: &[f64], _time: &[i64], _o: &SpreadOptions) -> Result<SpreadResult, String> {
    Err("not implemented".to_string())
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
