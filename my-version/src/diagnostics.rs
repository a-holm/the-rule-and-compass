//! Overfitting diagnostics.
//!
//! A symbolic search is powerful enough to fit noise, so "the error is tiny"
//! is not evidence of a real law. These helpers supply the standard checks:
//!
//! * **Information criteria** (AIC / AICc / BIC) trade fit against the number
//!   of free constants, so a slightly better fit that needs three extra
//!   constants is punished. AICc is the small-sample correction (Burnham &
//!   Anderson 2002).
//! * **Durbin–Watson** tests whether the residuals are still autocorrelated
//!   (Durbin & Watson 1950/1951). For ordered data a value near 2 means "no
//!   leftover structure"; far below 2 means the model left something in.
//! * **Ljung–Box** (Ljung & Box 1978) generalises that to several lags and
//!   turns it into a chi-square p-value; **Breusch–Pagan** (Breusch & Pagan
//!   1979) tests whether the residual variance changes with the fitted value;
//!   residual **skewness** and **excess kurtosis** flag non-normal residuals.
//! * **Null baselines**: the constant mean, a least-squares straight line and a
//!   k-nearest-neighbour learner. A model that does not beat these has not
//!   earned its complexity.
//! * **Permutation (y-randomisation) control**: run the *same* search on a
//!   shuffled target. If it finds a fit as good as the real one, the real fit
//!   is what the search can manufacture from noise (Rücker et al. 2007;
//!   the standard "y-scrambling" test).
//!
//! None of these proves a model is right; together they make "perfect fit"
//! falsifiable.

use crate::data::Sample;
use crate::expr::Node;
use crate::metrics::{self, Metric};
use crate::rng::Mt;

/// Least-squares information criteria for a model with `k` free constants fit
/// to `n` points with residual sum of squares `rss`.
///
/// Returns `(aic, aicc, bic)`, all "lower is better".
pub fn information_criteria(n: usize, rss: f64, k: usize) -> (f64, f64, f64) {
    let nf = n as f64;
    let kf = k as f64;
    if n == 0 || rss <= 0.0 {
        // A perfect fit drives the log-likelihood to +inf, so the criteria go
        // to -inf. Report exactly that instead of a NaN.
        return (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    }
    let aic = nf * (rss / nf).ln() + 2.0 * kf;
    let denom = nf - kf - 1.0;
    let aicc = if denom > 0.0 {
        aic + 2.0 * kf * (kf + 1.0) / denom
    } else {
        // More constants than the data can support: undefined.
        f64::INFINITY
    };
    let bic = nf * (rss / nf).ln() + kf * nf.ln();
    (aic, aicc, bic)
}

/// Durbin–Watson statistic `d = Σ(e_t − e_{t−1})² / Σ e_t²`, in row order.
///
/// `d ≈ 2` no autocorrelation, `d → 0` positive, `d → 4` negative. Only
/// meaningful when the rows are in a natural order (a temporal split); for a
/// shuffled split read it as a curiosity. NaN for <2 points or a perfect fit.
pub fn durbin_watson(resid: &[f64]) -> f64 {
    if resid.len() < 2 {
        return f64::NAN;
    }
    let num: f64 = resid.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum();
    let den: f64 = resid.iter().map(|e| e * e).sum();
    if den <= 0.0 {
        return f64::NAN;
    }
    num / den
}

// ---------------------------------------------------------------------------
// Null baselines
// ---------------------------------------------------------------------------

/// Training and test error of one trivial model, in the search's own metric.
#[derive(Clone, Copy, Debug)]
pub struct BaselineError {
    pub train_error: f64,
    pub test_error: f64,
}

/// The three null baselines, fit on the training rows and scored on the
/// training rows and on every test fold. A real model must beat all of them or
/// it has not earned its complexity (see `docs/overfitting.md` §3.5).
#[derive(Clone, Copy, Debug)]
pub struct Baselines {
    /// Constant predictor: the training mean of the target.
    pub mean: BaselineError,
    /// Ordinary least squares straight line (hyperplane) with intercept.
    pub linear: BaselineError,
    /// k-nearest-neighbour regression; `knn_k` is the `k` actually used.
    pub knn: BaselineError,
    pub knn_k: usize,
    /// #69: training rows searched for neighbours. **Always the full training
    /// size**: the reference set is never subsampled.
    pub knn_reference_rows: usize,
    /// #69: training query rows actually scored.
    pub knn_train_query_rows: usize,
    /// #69: standard error of the sampled **training** error, with the
    /// finite-population correction for sampling without replacement. `NaN`
    /// when every training row was scored.
    pub knn_train_query_sem: f64,
    /// #69: test query rows actually scored (summed over folds).
    pub knn_test_query_rows: usize,
    /// #69: standard error of the sampled **test** error (the fold mean), or
    /// `NaN` when every test row was scored.
    pub knn_test_query_sem: f64,
    /// #69: true when the query rows were subsampled. The reference set is
    /// always exact.
    pub knn_capped: bool,
}

/// #69: k-d trees answer exact k-NN queries for up to this many inputs. Above
/// it the exact brute-force path is used, with a reused distance buffer and
/// `select_nth_unstable` instead of a fresh allocation and a full sort. The
/// brute-force path is still `O(query_rows × reference_rows × d)`, so the
/// "200 000 rows under a minute" claim is a `d <= 8` claim.
pub const KNN_KD_MAX_DIM: usize = 8;

/// #69: query sampling is only an unbiased estimate of the row-average loss
/// for these metrics. The CLI refuses `--baseline-samples > 0` otherwise; the
/// library falls back to exact queries rather than reporting a different
/// estimator under the same name.
pub fn knn_sampling_supported(metric: Metric) -> bool {
    matches!(
        metric,
        Metric::Absolute | Metric::Squared | Metric::LogSquashed
    )
}

/// Number of neighbours used by the baseline k-NN learner.
///
/// `k = 5` is the choice: small enough to follow a gentle curve (a flexible
/// learner should be a *hard* baseline), large enough not to chase single
/// points the way `k = 1` does. It is capped at the number of training rows.
pub fn knn_k(n_train: usize) -> usize {
    5.min(n_train.max(1))
}

impl Baselines {
    /// Fit and score all three baselines. `test_folds` is one sample per test
    /// block; the reported test error is the mean over blocks, exactly as for a
    /// searched model. An empty `test_folds` yields an infinite test error.
    ///
    /// The k-NN baseline is exact: all training rows are neighbours and every
    /// training/test row is scored. Use [`Baselines::fit_sampled`] only when
    /// the exact query cost is unaffordable.
    pub fn fit(train: &Sample, test_folds: &[Sample], metric: Metric) -> Baselines {
        Baselines::fit_sampled(train, test_folds, metric, 0, 0)
    }

    /// #69: as [`Baselines::fit`], but optionally samples the **query** rows.
    ///
    /// `baseline_samples = 0` means "no caps at all": the exact reference set
    /// (every training row) and every query row. A positive value scores at
    /// most that many deterministic query rows per set (training and each test
    /// fold), drawn from `seed`. The reference set stays exact in both cases.
    ///
    /// Sampling is only offered for the mean-loss metrics accepted by
    /// [`knn_sampling_supported`]; for every other metric this falls back to
    /// exact queries, because a subsample of a median/correlation/AIC is a
    /// different estimator, not an unbiased estimate of the full-sample value.
    pub fn fit_sampled(
        train: &Sample,
        test_folds: &[Sample],
        metric: Metric,
        baseline_samples: usize,
        seed: u32,
    ) -> Baselines {
        // Mean.
        let mean = if train.is_empty() {
            f64::NAN
        } else {
            train.y.iter().sum::<f64>() / train.y.len() as f64
        };
        let mean_err = score_folds(train, test_folds, metric, &|_x| mean);

        // Least-squares line.
        let coef = least_squares_linear(train);
        let linear_err = match &coef {
            Some(c) => score_folds(train, test_folds, metric, &|x| dot(c, x)),
            None => BaselineError {
                train_error: f64::NAN,
                test_error: f64::NAN,
            },
        };

        // k-NN. The index is built once over the full training set and reused
        // for the training queries and every test fold.
        let k = knn_k(train.len());
        let mut index = KnnIndex::new(train);
        let sample = if knn_sampling_supported(metric) {
            baseline_samples
        } else {
            0
        };
        let train_q = score_knn_queries(
            &mut index,
            train,
            metric,
            k,
            sample,
            seed ^ 0x7472_6169,
        );
        let mut test_rows = 0usize;
        let mut test_var = 0.0;
        let mut test_sampled = false;
        let test_error = if test_folds.is_empty() {
            f64::INFINITY
        } else {
            let mut acc = 0.0;
            for (f, fold) in test_folds.iter().enumerate() {
                let fold_seed = seed
                    ^ 0x666f_6c64u32
                    ^ (0x9e37_79b9u32.wrapping_mul(f as u32 + 1));
                let q = score_knn_queries(&mut index, fold, metric, k, sample, fold_seed);
                test_rows += q.rows;
                if q.sem.is_finite() {
                    test_sampled = true;
                    // Var(mean over folds) = Σ Var(fold mean) / F².
                    test_var += q.sem * q.sem;
                }
                acc += q.error;
            }
            acc / test_folds.len() as f64
        };
        let f = test_folds.len() as f64;
        let test_sem = if test_sampled && f > 0.0 {
            test_var.sqrt() / f
        } else {
            f64::NAN
        };
        let sampled = train_q.rows < train.len() || test_sampled;

        Baselines {
            mean: mean_err,
            linear: linear_err,
            knn: BaselineError {
                train_error: train_q.error,
                test_error,
            },
            knn_k: k,
            knn_reference_rows: train.len(),
            knn_train_query_rows: train_q.rows,
            knn_train_query_sem: train_q.sem,
            knn_test_query_rows: test_rows,
            knn_test_query_sem: test_sem,
            knn_capped: sampled,
        }
    }

    /// #69: the extra fields of the `baselines.knn` object in `--json` /
    /// `--ndjson`, so the text and machine-readable surfaces cannot drift.
    pub fn knn_json_extra(&self) -> String {
        fn sem_json(v: f64) -> String {
            if v.is_finite() {
                format!("{:.6e}", v)
            } else {
                "null".to_string()
            }
        }
        format!(
            "\"reference_rows\":{},\"train_query_rows\":{},\"train_query_sem\":{},\"test_query_rows\":{},\"test_query_sem\":{},\"capped\":{}",
            self.knn_reference_rows,
            self.knn_train_query_rows,
            sem_json(self.knn_train_query_sem),
            self.knn_test_query_rows,
            sem_json(self.knn_test_query_sem),
            self.knn_capped
        )
    }
}

/// One scored query set: its error, how many rows it used, and the standard
/// error of the mean when it was sampled.
struct QueryScore {
    error: f64,
    rows: usize,
    sem: f64,
}

/// Score `query` against the fixed reference index, sampling only the query
/// rows when `sample > 0`.
fn score_knn_queries(
    index: &mut KnnIndex<'_>,
    query: &Sample,
    metric: Metric,
    k: usize,
    sample: usize,
    seed: u32,
) -> QueryScore {
    if query.is_empty() {
        return QueryScore {
            error: metrics::score(&[], &[], metric),
            rows: 0,
            sem: f64::NAN,
        };
    }
    let take = if sample == 0 {
        query.len()
    } else {
        sample.min(query.len())
    };
    let idx = sample_indices(query.len(), take, seed);
    let mut pred = Vec::with_capacity(take);
    for &i in &idx {
        pred.push(index.predict(&query.x[i], k));
    }
    let y: Vec<f64> = idx.iter().map(|&i| query.y[i]).collect();
    let error = metrics::score(&pred, &y, metric);
    let sem = if take < query.len() {
        mean_sem(metric, &pred, &y, query.len())
    } else {
        f64::NAN
    };
    QueryScore {
        error,
        rows: take,
        sem,
    }
}

/// Standard error of the mean for metrics that are a plain mean of per-row
/// losses, with the finite-population correction for sampling `q` distinct rows
/// without replacement from `population`:
///
/// `SE = (s / √q) · √((N − q) / (N − 1))`.
///
/// `NaN` when every row was scored or the metric is not a plain mean.
fn mean_sem(metric: Metric, pred: &[f64], y: &[f64], population: usize) -> f64 {
    let q = pred.len();
    let n = population;
    if q < 2 || q >= n || n < 2 {
        return f64::NAN;
    }
    let mut losses = Vec::with_capacity(q);
    for (&p, &t) in pred.iter().zip(y) {
        let e = p - t;
        let loss = match metric {
            Metric::Absolute => e.abs(),
            Metric::Squared => e * e,
            Metric::LogSquashed => (1.0 + e * e).ln(),
            _ => return f64::NAN,
        };
        losses.push(loss);
    }
    let mean = losses.iter().sum::<f64>() / q as f64;
    let var = losses
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / (q - 1) as f64;
    let s = var.sqrt();
    let fpc = (((n - q) as f64) / ((n - 1) as f64)).sqrt();
    (s / (q as f64).sqrt()) * fpc
}

/// `take` distinct row indices from `0..n`, deterministic in `seed` and
/// returned in ascending order.
fn sample_indices(n: usize, take: usize, seed: u32) -> Vec<usize> {
    let take = take.min(n);
    if take == n {
        return (0..n).collect();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut rng = Mt::new(seed);
    // Partial Fisher–Yates: the first `take` slots become a uniform sample.
    for i in 0..take {
        let j = i + rng.usize_below(n - i);
        idx.swap(i, j);
    }
    idx.truncate(take);
    idx.sort_unstable();
    idx
}

// ---------------------------------------------------------------------------
// #69: exact k-NN index
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Neighbor {
    dist: f64,
    idx: usize,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Squared Euclidean distance, summed in input order (the same arithmetic as
/// [`knn_predict`], so exact ties agree).
fn sqdist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

struct KdNode {
    point: usize,
    axis: usize,
    left: Option<usize>,
    right: Option<usize>,
    /// Bounding box of the subtree, used to prune whole subtrees. Only the
    /// first `dim` entries are meaningful.
    min: [f64; KNN_KD_MAX_DIM],
    max: [f64; KNN_KD_MAX_DIM],
}

/// An exact k-NN index over every training row.
///
/// A balanced k-d tree is built for `1 <= d <= KNN_KD_MAX_DIM`; otherwise the
/// brute-force path reuses one distance buffer and selects the k smallest with
/// `select_nth_unstable` rather than allocating and sorting per query.
struct KnnIndex<'a> {
    train: &'a Sample,
    dim: usize,
    nodes: Vec<KdNode>,
    root: Option<usize>,
    buf: Vec<(f64, usize)>,
    best: Vec<Neighbor>,
}

impl<'a> KnnIndex<'a> {
    fn new(train: &'a Sample) -> Self {
        let dim = train.x.first().map(|r| r.len()).unwrap_or(0);
        // `Sample::new` is public and accepts ragged rows. The k-d tree needs a
        // rectangular block; anything else falls back to the brute-force path,
        // whose `zip` arithmetic matches the old `knn_predict` semantics.
        let rectangular = train.x.iter().all(|r| r.len() == dim);
        let mut nodes = Vec::new();
        let root = if !train.is_empty() && rectangular && dim > 0 && dim <= KNN_KD_MAX_DIM {
            let mut idx: Vec<usize> = (0..train.len()).collect();
            build_kd(train, &mut idx, 0, &mut nodes)
        } else {
            None
        };
        KnnIndex {
            train,
            dim,
            nodes,
            root,
            buf: Vec::new(),
            best: Vec::new(),
        }
    }

    fn predict(&mut self, x: &[f64], k: usize) -> f64 {
        if self.train.is_empty() {
            return f64::NAN;
        }
        let kk = k.clamp(1, self.train.len());
        // A query of a different width would read the wrong split coordinate
        // and box bounds; use the brute-force path, which matches `zip`.
        if let Some(root) = self.root {
            if x.len() == self.dim {
                self.best.clear();
                search_kd(
                    &self.nodes,
                    self.train,
                    root,
                    x,
                    self.dim,
                    kk,
                    &mut self.best,
                );
                self.best.sort_unstable();
                return self.best.iter().map(|n| self.train.y[n.idx]).sum::<f64>() / kk as f64;
            }
        }
        self.predict_brute(x, kk)
    }

    fn predict_brute(&mut self, x: &[f64], kk: usize) -> f64 {
        self.buf.clear();
        self.buf.extend(
            self.train
                .x
                .iter()
                .enumerate()
                .map(|(i, r)| (sqdist(r, x), i)),
        );
        if kk < self.buf.len() {
            self.buf.select_nth_unstable_by(kk - 1, cmp_dist);
        }
        self.buf[..kk].sort_unstable_by(cmp_dist);
        self.buf[..kk]
            .iter()
            .map(|&(_, i)| self.train.y[i])
            .sum::<f64>()
            / kk as f64
    }
}

fn cmp_dist(a: &(f64, usize), b: &(f64, usize)) -> std::cmp::Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.1.cmp(&b.1))
}

fn build_kd(
    sample: &Sample,
    idx: &mut [usize],
    depth: usize,
    nodes: &mut Vec<KdNode>,
) -> Option<usize> {
    if idx.is_empty() {
        return None;
    }
    let dim = sample.x[0].len();
    // Bounding box and widest axis, both in one pass over the node's rows.
    let mut min = [f64::INFINITY; KNN_KD_MAX_DIM];
    let mut max = [f64::NEG_INFINITY; KNN_KD_MAX_DIM];
    for &i in idx.iter() {
        for j in 0..dim {
            let v = sample.x[i][j];
            if v < min[j] {
                min[j] = v;
            }
            if v > max[j] {
                max[j] = v;
            }
        }
    }
    let axis = (0..dim)
        .max_by(|&a, &b| {
            (max[a] - min[a])
                .partial_cmp(&(max[b] - min[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(depth % dim);
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        sample.x[a][axis]
            .partial_cmp(&sample.x[b][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let point = idx[mid];
    let (left_idx, rest) = idx.split_at_mut(mid);
    let right_idx = &mut rest[1..];
    let left = build_kd(sample, left_idx, depth + 1, nodes);
    let right = build_kd(sample, right_idx, depth + 1, nodes);
    let id = nodes.len();
    nodes.push(KdNode {
        point,
        axis,
        left,
        right,
        min,
        max,
    });
    Some(id)
}

fn search_kd(
    nodes: &[KdNode],
    train: &Sample,
    node: usize,
    x: &[f64],
    dim: usize,
    k: usize,
    best: &mut Vec<Neighbor>,
) {
    let nd = &nodes[node];
    if best.len() >= k {
        // The box lower bound is conservative: only prune when it clearly
        // exceeds the current worst distance (the small relative slack keeps
        // floating-point summation from ever pruning an exact tie).
        let worst = worst_dist(best, k);
        if bbox_dist2(nd, x, dim) > worst + worst.abs() * 1e-12 + 1e-12 {
            return;
        }
    }
    let p = &train.x[nd.point];
    offer(best, Neighbor {
        dist: sqdist(p, x),
        idx: nd.point,
    }, k);

    let diff = x[nd.axis] - p[nd.axis];
    let (near, far) = if diff <= 0.0 {
        (nd.left, nd.right)
    } else {
        (nd.right, nd.left)
    };
    if let Some(child) = near {
        search_kd(nodes, train, child, x, dim, k, best);
    }
    // `diff²` is a lower bound on every point in the far subtree; visit it
    // when it can still contain a neighbour (ties included, for row-order
    // tie-breaking).
    let worst = worst_dist(best, k);
    if diff * diff <= worst {
        if let Some(child) = far {
            search_kd(nodes, train, child, x, dim, k, best);
        }
    }
}

fn offer(best: &mut Vec<Neighbor>, cand: Neighbor, k: usize) {
    if best.len() < k {
        best.push(cand);
        return;
    }
    let mut worst = 0;
    for i in 1..best.len() {
        if best[i] > best[worst] {
            worst = i;
        }
    }
    if cand < best[worst] {
        best[worst] = cand;
    }
}

fn worst_dist(best: &[Neighbor], k: usize) -> f64 {
    if best.len() < k {
        f64::INFINITY
    } else {
        best.iter().map(|n| n.dist).fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Squared distance from `x` to the node's bounding box (a lower bound on the
/// distance to every point in the subtree). `dim` is the reference width; the
/// caller guarantees `x.len() == dim`, so the unused `min`/`max` entries are
/// never read.
fn bbox_dist2(nd: &KdNode, x: &[f64], dim: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..dim.min(x.len()) {
        let v = x[j];
        if v < nd.min[j] {
            let d = nd.min[j] - v;
            s += d * d;
        } else if v > nd.max[j] {
            let d = v - nd.max[j];
            s += d * d;
        }
    }
    s
}

/// Score one predictor on the training sample and on each test fold. The test
/// error is the mean over folds (matching how a searched model is scored).
fn score_folds(
    train: &Sample,
    test_folds: &[Sample],
    metric: Metric,
    predict: &dyn Fn(&[f64]) -> f64,
) -> BaselineError {
    let train_pred: Vec<f64> = train.x.iter().map(|x| predict(x)).collect();
    let train_error = metrics::score(&train_pred, &train.y, metric);
    let test_error = if test_folds.is_empty() {
        f64::INFINITY
    } else {
        let mut acc = 0.0;
        for f in test_folds {
            let pred: Vec<f64> = f.x.iter().map(|x| predict(x)).collect();
            acc += metrics::score(&pred, &f.y, metric);
        }
        acc / test_folds.len() as f64
    };
    BaselineError {
        train_error,
        test_error,
    }
}

/// `b0 + Σ b_j x_j` for coefficients `c = [b0, b1, ..]` (missing terms are 0).
fn dot(c: &[f64], x: &[f64]) -> f64 {
    let mut v = c[0];
    for (j, &xj) in x.iter().enumerate() {
        if j + 1 < c.len() {
            v += c[j + 1] * xj;
        }
    }
    v
}

/// OLS coefficients `[b0, b1, .., bp]` for `y ≈ b0 + Σ b_j x_j`.
///
/// The normal equations `(XᵀX) β = Xᵀy` are formed and solved by Gaussian
/// elimination with partial pivoting (pure `std`, no dependencies). A constant
/// input column makes `XᵀX` singular; a small ridge on the diagonal is added
/// once to keep the baseline defined (the intercept then absorbs the constant).
/// Returns `None` for an empty sample or a system that is still singular.
pub fn least_squares_linear(train: &Sample) -> Option<Vec<f64>> {
    if train.is_empty() {
        return None;
    }
    let p = 1 + train.x.first().map(|r| r.len()).unwrap_or(0);
    // Normal equations.
    let mut a = vec![vec![0.0; p]; p];
    let mut b = vec![0.0; p];
    for (row, &y) in train.x.iter().zip(&train.y) {
        let mut design = vec![1.0];
        design.extend_from_slice(row);
        for i in 0..p {
            for j in 0..p {
                a[i][j] += design[i] * design[j];
            }
            b[i] += design[i] * y;
        }
    }
    if let Some(x) = solve_normal(&a, &b) {
        return Some(x);
    }
    // Ridge fallback for a rank-deficient design.
    let scale = (0..p).map(|i| a[i][i]).fold(0.0_f64, f64::max).max(1.0);
    for i in 0..p {
        a[i][i] += 1e-9 * scale;
    }
    solve_normal(&a, &b)
}

/// Solve a dense linear system in place; `None` when a pivot is (numerically)
/// zero. Row `i` of `a` is one equation.
fn solve_normal(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut rhs = b.to_vec();
    let mut scale = 1.0_f64;
    for row in &m {
        for &v in row {
            scale = scale.max(v.abs());
        }
    }
    let tol = 1e-12 * scale;
    for col in 0..n {
        // Partial pivoting.
        let (mut piv, mut best) = (col, m[col][col].abs());
        for r in col + 1..n {
            let v = m[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= tol {
            return None;
        }
        if piv != col {
            m.swap(piv, col);
            rhs.swap(piv, col);
        }
        let d = m[col][col];
        for r in col + 1..n {
            let f = m[r][col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
            rhs[r] -= f * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= m[i][j] * x[j];
        }
        x[i] = s / m[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Average of the `k` nearest training targets to `x` (Euclidean distance in
/// the (normalised) input space). Ties are broken by training-row order, so the
/// result is deterministic.
pub fn knn_predict(train: &Sample, x: &[f64], k: usize) -> f64 {
    if train.is_empty() {
        return f64::NAN;
    }
    let mut d: Vec<(f64, usize)> = train
        .x
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let s: f64 = r
                .iter()
                .zip(x)
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (s, i)
        })
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let kk = k.clamp(1, d.len());
    d[..kk].iter().map(|&(_, i)| train.y[i]).sum::<f64>() / kk as f64
}

// ---------------------------------------------------------------------------
// Residual tests
// ---------------------------------------------------------------------------

/// Ljung–Box test for autocorrelation. `statistic ~ χ²` with `lags` degrees of
/// freedom under the null of no autocorrelation, so a small `p_value` means the
/// residuals are still autocorrelated.
#[derive(Clone, Copy, Debug)]
pub struct LjungBox {
    pub lags: usize,
    pub statistic: f64,
    pub p_value: f64,
}

/// Breusch–Pagan test for heteroscedasticity against the fitted values.
/// `statistic ~ χ²` with one degree of freedom; a small `p_value` means the
/// residual variance changes with the fitted value.
#[derive(Clone, Copy, Debug)]
pub struct BreuschPagan {
    pub statistic: f64,
    /// Number of auxiliary regressors (1: the fitted value).
    pub df: usize,
    pub p_value: f64,
}

/// The residual diagnostics for one model.
#[derive(Clone, Copy, Debug)]
pub struct ResidualTests {
    pub n: usize,
    /// Lags used by Ljung–Box ([`default_ljung_box_lags`]).
    pub lags: usize,
    pub durbin_watson: f64,
    pub ljung_box: LjungBox,
    pub breusch_pagan: BreuschPagan,
    /// Moment skewness (`0` for a symmetric residual).
    pub skewness: f64,
    /// Excess kurtosis (`0` for a Gaussian residual, `< 0` for flat tails).
    pub kurtosis: f64,
}

/// Lags used by [`residual_tests`] unless the caller chooses otherwise:
/// `min(10, n/5)`, the usual rule of thumb.
pub fn default_ljung_box_lags(n: usize) -> usize {
    (n / 5).min(10)
}

/// Every residual check at once, with the default lag count.
pub fn residual_tests(resid: &[f64], fitted: &[f64]) -> ResidualTests {
    let n = resid.len();
    let lags = default_ljung_box_lags(n);
    ResidualTests {
        n,
        lags,
        durbin_watson: durbin_watson(resid),
        ljung_box: ljung_box(resid, lags),
        breusch_pagan: breusch_pagan(resid, fitted),
        skewness: skewness(resid),
        kurtosis: kurtosis(resid),
    }
}

/// Sample autocorrelation of `resid` at `lag` (the usual biased `1/n`
/// estimator). NaN for a degenerate input.
pub fn autocorrelation(resid: &[f64], lag: usize) -> f64 {
    let n = resid.len();
    if lag == 0 || lag >= n {
        return f64::NAN;
    }
    let mean = resid.iter().sum::<f64>() / n as f64;
    let mut den = 0.0;
    let mut num = 0.0;
    for i in 0..n {
        den += (resid[i] - mean) * (resid[i] - mean);
        if i + lag < n {
            num += (resid[i] - mean) * (resid[i + lag] - mean);
        }
    }
    if den <= 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

/// Ljung–Box statistic `Q = n(n+2) Σ_{k=1}^{h} ρ_k² / (n−k)` and its
/// chi-square p-value (Ljung & Box 1978).
pub fn ljung_box(resid: &[f64], lags: usize) -> LjungBox {
    let n = resid.len();
    if lags == 0 || n <= lags + 1 {
        return LjungBox {
            lags,
            statistic: f64::NAN,
            p_value: f64::NAN,
        };
    }
    let mut sum = 0.0;
    for k in 1..=lags {
        let r = autocorrelation(resid, k);
        if r.is_finite() {
            sum += r * r / (n - k) as f64;
        }
    }
    let q = n as f64 * (n as f64 + 2.0) * sum;
    LjungBox {
        lags,
        statistic: q,
        p_value: chi2_sf(q, lags),
    }
}

/// Breusch–Pagan test: regress the squared residuals on the fitted values and
/// turn `n·R²` into a chi-square statistic (Breusch & Pagan 1979). One
/// auxiliary regressor (the fitted value) gives one degree of freedom.
pub fn breusch_pagan(resid: &[f64], fitted: &[f64]) -> BreuschPagan {
    let n = resid.len().min(fitted.len());
    if n < 3 {
        return BreuschPagan {
            statistic: f64::NAN,
            df: 1,
            p_value: f64::NAN,
        };
    }
    let u: Vec<f64> = resid[..n].iter().map(|e| e * e).collect();
    let f = &fitted[..n];
    let ubar = u.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = u.iter().map(|v| (v - ubar) * (v - ubar)).sum();
    if ss_tot <= 0.0 {
        // The squared residuals are constant: no evidence against the null.
        return BreuschPagan {
            statistic: 0.0,
            df: 1,
            p_value: 1.0,
        };
    }
    let aux = Sample {
        x: f.iter().map(|&v| vec![v]).collect(),
        y: u.clone(),
        w: Vec::new(),
        ys: Vec::new(),
    };
    let ss_res = match least_squares_linear(&aux) {
        Some(c) => u
            .iter()
            .zip(f)
            .map(|(v, &fv)| {
                let pred = c[0] + c[1] * fv;
                (v - pred) * (v - pred)
            })
            .sum(),
        None => ss_tot,
    };
    let r2 = (1.0 - ss_res / ss_tot).clamp(0.0, 1.0);
    let lm = n as f64 * r2;
    BreuschPagan {
        statistic: lm,
        df: 1,
        p_value: chi2_sf(lm, 1),
    }
}

/// Population (moment) skewness `m3 / m2^{3/2}`. NaN for fewer than 3 points or
/// zero variance.
pub fn skewness(x: &[f64]) -> f64 {
    if x.len() < 3 {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let (mut m2, mut m3) = (0.0, 0.0);
    for &v in x {
        let d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    if m2 <= 0.0 {
        f64::NAN
    } else {
        m3 / m2.powf(1.5)
    }
}

/// Excess kurtosis `m4 / m2² − 3`. `0` for a Gaussian; NaN guards as above.
pub fn kurtosis(x: &[f64]) -> f64 {
    if x.len() < 4 {
        return f64::NAN;
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &v in x {
        let d = v - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;
    if m2 <= 0.0 {
        f64::NAN
    } else {
        m4 / (m2 * m2) - 3.0
    }
}

// ---------------------------------------------------------------------------
// Chi-square tail probability (pure std)
// ---------------------------------------------------------------------------

/// Upper-tail probability `P(X > x)` for a chi-square with `df` degrees of
/// freedom. Uses the regularised incomplete gamma function, so it needs no
/// dependency.
pub fn chi2_sf(x: f64, df: usize) -> f64 {
    if df == 0 {
        return f64::NAN;
    }
    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    let a = df as f64 / 2.0;
    (1.0 - gamma_p(a, x / 2.0)).clamp(0.0, 1.0)
}

/// Regularised lower incomplete gamma `P(a, x)` (Numerical Recipes `gammp`).
fn gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Logarithm of the gamma function, Lanczos approximation (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Summary of a permutation (y-randomisation) control.
#[derive(Clone, Debug)]
pub struct NullControl {
    pub runs: usize,
    /// Best training error achieved on each shuffled target.
    pub best: Vec<f64>,
    /// Best (smallest) error any shuffled run reached.
    pub null_min: f64,
    /// Median of the shuffled runs' best errors.
    pub null_median: f64,
    /// `(#{null ≤ observed} + 1) / (runs + 1)`. Small ⇒ the real fit is not
    /// something the search can produce from noise.
    pub p_value: f64,
}

impl NullControl {
    pub fn summarise(observed: f64, mut best: Vec<f64>) -> NullControl {
        best.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = best.len();
        let null_min = best.first().copied().unwrap_or(f64::NAN);
        let null_median = if n == 0 {
            f64::NAN
        } else if n % 2 == 1 {
            best[n / 2]
        } else {
            0.5 * (best[n / 2 - 1] + best[n / 2])
        };
        let count = best.iter().filter(|&&e| e <= observed).count();
        let p_value = (count as f64 + 1.0) / (n as f64 + 1.0);
        NullControl {
            runs: n,
            best,
            null_min,
            null_median,
            p_value,
        }
    }

    /// A plain-language verdict. With `r` runs the smallest attainable p-value
    /// is `1/(r+1)`, so a p-value at that floor means "beats every control" and
    /// the honest advice is to run more controls.
    pub fn verdict(&self) -> &'static str {
        let floor = 1.0 / (self.runs as f64 + 1.0);
        if self.p_value <= 0.05 {
            "the real fit beats the noise control (unlikely to be chance)"
        } else if self.p_value <= floor + 1e-9 {
            "the real fit beats every control run; add more --null-runs to sharpen p"
        } else if self.p_value <= 0.2 {
            "the real fit is only weakly better than the noise control"
        } else {
            "the search fits shuffled targets as well: suspect overfitting"
        }
    }
}

// ---------------------------------------------------------------------------
// .632+ bootstrap error estimate
// ---------------------------------------------------------------------------

/// The `.632+` bootstrap error estimate of Efron & Tibshirani (1997).
///
/// Three numbers describe how much of a model's apparent accuracy is optimism:
///
/// * `apparent` — the error of the fitted model on the very rows it was fit to
///   (optimistic);
/// * `oob` — the mean error on the *out-of-bag* rows over bootstrap resamples
///   in which the model's **constants only** are refit (pessimistic);
/// * `estimate` — the `.632+` combination of the two, which lies between them.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapError {
    /// Bootstrap resamples requested.
    pub resamples: usize,
    /// Resamples that yielded a finite out-of-bag error (the rest were skipped,
    /// e.g. because the resample left no out-of-bag row).
    pub used: usize,
    /// #92: training rows in the source sample.
    pub source_rows: usize,
    /// #92: rows drawn per resample: `n`, or `n^0.7` under the BLB cap.
    pub resample_rows: usize,
    /// #92: true when the BLB cap subsampled the source rows.
    pub capped: bool,
    /// Error of the model on the full training set.
    pub apparent: f64,
    /// Mean out-of-bag error over the resamples.
    pub oob: f64,
    /// No-information error rate `γ` estimated from the data.
    pub no_information: f64,
    /// Relative overfitting rate `R̂ = (oob − apparent)/(γ − apparent)`, clamped
    /// to `[0, 1]`.
    pub relative_overfitting: f64,
    /// The `.632+` weight `ŵ = 0.632/(1 − 0.368·R̂)`, in `[0.632, 1]`.
    pub weight: f64,
    /// `(1 − ŵ)·apparent + ŵ·oob`.
    pub estimate: f64,
}

/// `1 − e^{−1}`, the `.632` of the method's name.
const BOOTSTRAP_632: f64 = 0.632;

/// Permutations used to estimate the no-information error rate.
const NO_INFO_PERMUTATIONS: usize = 32;

/// Estimate the `.632+` error of `tree` on `train` by bootstrap resampling.
///
/// For each of `resamples` draws, `n = train.len()` rows are sampled **with
/// replacement**; the tree's constants are refit on that resample alone with
/// [`crate::fit::fit`] and the model is scored on the rows left out (the
/// out-of-bag rows) in the search's own metric. The out-of-bag error is the mean
/// of those scores.
///
/// #92: when `n^0.7 >= 1_000` the draw is bounded with the bag of little
/// bootstraps (Kleiner et al. 2014): a `b = n^0.7`-row subsample is drawn first
/// and the resample is taken within it. `source_rows`, `resample_rows` and
/// `capped` report which estimator ran. Below the threshold the result is
/// bit-for-bit the exact bootstrap above.
///
/// The no-information error rate `γ` is what the model would score if its
/// predictions carried no information about the responses: the fitted values
/// are paired with a random permutation of the targets and averaged over
/// [`NO_INFO_PERMUTATIONS`] permutations.
///
/// Then, writing `err_app` for the apparent error:
///
/// ```text
/// R̂        = (oob − err_app) / (γ − err_app)      clamped to [0, 1]
/// ŵ        = 0.632 / (1 − 0.368·R̂)                so 0.632 ≤ ŵ ≤ 1
/// err_.632+ = (1 − ŵ)·err_app + ŵ·oob
/// ```
///
/// `R̂ = 0` (no extra overfitting beyond the apparent error) gives `ŵ = 0.632`;
/// `R̂ = 1` gives `ŵ = 1` and the fully pessimistic out-of-bag error. Because
/// `ŵ ∈ [0.632, 1]`, the estimate is a convex combination of the apparent and
/// out-of-bag errors and therefore always lies between them.
pub fn bootstrap_error(
    tree: &Node,
    train: &Sample,
    metric: Metric,
    resamples: usize,
    fit_iters: usize,
    rng: &mut Mt,
) -> BootstrapError {
    let n = train.len();

    // Apparent error: the model scored on the full training set. The fitted
    // values are kept for the no-information estimate below.
    let apparent_pred: Vec<f64> = train.x.iter().map(|row| crate::expr::eval(tree, row)).collect();
    let apparent = if n == 0 {
        f64::INFINITY
    } else {
        crate::search::eval_error(tree, train, metric)
    };

    // Out-of-bag error: refit constants on each bootstrap resample, score the
    // rows that did not make it into the resample.
    //
    // #92: the resample is an index draw; only the `b` selected rows are
    // materialised. Above the BLB cap (`b = n^0.7`) the out-of-bag rows are the
    // subsample rows not drawn. Below it the draw is exactly the old one.
    let plan = crate::resample::plan(n);
    let mut oob_sum = 0.0;
    let mut oob_used = 0usize;
    if n > 0 && resamples > 0 {
        let b = plan.rows;
        let mut in_bag = vec![false; b];
        let mut bx: Vec<Vec<f64>> = Vec::with_capacity(b);
        let mut by: Vec<f64> = Vec::with_capacity(b);
        for _ in 0..resamples {
            for f in in_bag.iter_mut() {
                *f = false;
            }
            bx.clear();
            by.clear();
            let base = if plan.capped {
                Some(crate::resample::subsample_indices(n, b, rng))
            } else {
                None
            };
            let draws = crate::resample::draw_indices(b, b, rng);
            for &j in &draws {
                let src = base.as_ref().map(|v| v[j]).unwrap_or(j);
                in_bag[j] = true;
                bx.push(train.x[src].clone());
                by.push(train.y[src]);
            }
            let oob: Vec<usize> = (0..b)
                .filter(|&j| !in_bag[j])
                .map(|j| base.as_ref().map(|v| v[j]).unwrap_or(j))
                .collect();
            if oob.is_empty() {
                continue;
            }
            // Constants-only refit on the resample (the shape is fixed).
            let mut t = tree.clone();
            crate::fit::fit(&mut t, &bx, &by, fit_iters);
            let scored = Sample {
                x: oob.iter().map(|&j| train.x[j].clone()).collect(),
                y: oob.iter().map(|&j| train.y[j]).collect(),
                w: Vec::new(),
                ys: Vec::new(),
            };
            let e = crate::search::eval_error(&t, &scored, metric);
            if e.is_finite() {
                oob_sum += e;
                oob_used += 1;
            }
        }
    }
    let oob = if oob_used > 0 {
        oob_sum / oob_used as f64
    } else {
        f64::INFINITY
    };

    // No-information error rate: predictions paired with permuted responses.
    let no_information = if n >= 2 && !apparent_pred.is_empty() {
        let mut shuffled = train.y.clone();
        let mut acc = 0.0;
        for _ in 0..NO_INFO_PERMUTATIONS {
            for j in (1..n).rev() {
                let t = rng.usize_below(j + 1);
                shuffled.swap(j, t);
            }
            acc += metrics::score_with_constants(
                &apparent_pred,
                &shuffled,
                metric,
                crate::expr::count_constants(tree),
            );
        }
        acc / NO_INFO_PERMUTATIONS as f64
    } else {
        f64::NAN
    };

    // Relative overfitting rate, clamped as Efron & Tibshirani prescribe.
    let denom = no_information - apparent;
    let relative_overfitting = if apparent.is_finite()
        && oob.is_finite()
        && denom.is_finite()
        && denom > 0.0
        && oob > apparent
    {
        ((oob - apparent) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let weight = BOOTSTRAP_632 / (1.0 - (1.0 - BOOTSTRAP_632) * relative_overfitting);
    let estimate = if apparent.is_finite() && oob.is_finite() {
        (1.0 - weight) * apparent + weight * oob
    } else if apparent.is_finite() {
        apparent
    } else {
        f64::INFINITY
    };

    BootstrapError {
        resamples,
        used: oob_used,
        source_rows: plan.source_rows,
        resample_rows: plan.rows,
        capped: plan.capped,
        apparent,
        oob,
        no_information,
        relative_overfitting,
        weight,
        estimate,
    }
}

// ---------------------------------------------------------------------------
// Variable sensitivity (issue #29)
// ---------------------------------------------------------------------------

/// How much one input moves the fitted law, measured over the training rows.
///
/// The derivative is analytic (the same adjoint sweep as the constant fit), not
/// a finite difference. `mean_abs_elasticity` is
/// `mean(|∂f/∂x_i| · σ_x_i / σ_f)`: the standardised, unit-free change in the
/// prediction for a one-standard-deviation change in the input. It is directly
/// comparable across inputs even when their units differ.
#[derive(Clone, Copy, Debug)]
pub struct SensitivityRow {
    /// Index into the input vector.
    pub input: usize,
    /// `mean(|∂f/∂x_i| · σ_x_i / σ_f)`, the ranking statistic.
    pub mean_abs_elasticity: f64,
    /// Smallest signed standardised contribution over the rows.
    pub min_contribution: f64,
    /// Largest signed standardised contribution over the rows.
    pub max_contribution: f64,
    /// Fraction of rows with a positive derivative.
    pub positive_fraction: f64,
    /// Fraction of rows with a negative derivative.
    pub negative_fraction: f64,
    /// Rows with a finite derivative (the fractions use this denominator).
    pub rows_used: usize,
}

/// Per-input sensitivity of `tree` over the training rows.
#[derive(Clone, Debug)]
pub struct SensitivityReport {
    /// One entry per input, ranked by `mean_abs_elasticity`, descending.
    pub rows: Vec<SensitivityRow>,
    /// Training rows used.
    pub n: usize,
    /// Standard deviation of the model's predictions (`σ_f`).
    pub output_std: f64,
}

/// Compute the analytic sensitivity of `tree` to each input over `train`.
///
/// `tree` is the search's model in normalised input space, and `stds[j]` is the
/// scaling applied to input `j` during preparation. The derivative with respect
/// to the original input is `∂f/∂x_j = (∂f/∂x_norm_j) / stds[j]`; the actual
/// original-scale standard deviation is `stds[j] · sd(x_norm_j)`, so the two
/// factors cancel in the elasticity and the result is scale-free even with
/// `--scale iqr` or `--scale range`.
pub fn sensitivity(tree: &Node, train: &Sample, stds: &[f64]) -> SensitivityReport {
    let n = train.len();
    let nv = train.x.first().map(|r| r.len()).unwrap_or(0);
    let pred: Vec<f64> = train.x.iter().map(|r| crate::expr::eval(tree, r)).collect();
    let output_std = std_dev(&pred);

    // Standard deviation of each normalised input column.
    let mut sd_norm = vec![0.0; nv];
    for (j, slot) in sd_norm.iter_mut().enumerate() {
        let vals: Vec<f64> = train
            .x
            .iter()
            .map(|r| r.get(j).copied().unwrap_or(f64::NAN))
            .collect();
        *slot = std_dev(&vals);
    }

    let mut sum_abs = vec![0.0; nv];
    let mut min_contrib = vec![f64::INFINITY; nv];
    let mut max_contrib = vec![f64::NEG_INFINITY; nv];
    let mut positive = vec![0usize; nv];
    let mut negative = vec![0usize; nv];
    let mut used = vec![0usize; nv];
    let scale_ok = output_std.is_finite() && output_std > 0.0;

    for row in &train.x {
        let grad = crate::fit::input_gradients(tree, row);
        for j in 0..nv {
            let d = grad.get(j).copied().unwrap_or(0.0);
            let scale = stds.get(j).copied().unwrap_or(1.0);
            let d_orig = if scale.abs() >= 1e-12 { d / scale } else { d };
            let contribution = if scale_ok {
                d_orig * (scale.abs() * sd_norm[j]) / output_std
            } else {
                0.0
            };
            if !contribution.is_finite() {
                continue;
            }
            used[j] += 1;
            sum_abs[j] += contribution.abs();
            min_contrib[j] = min_contrib[j].min(contribution);
            max_contrib[j] = max_contrib[j].max(contribution);
            if contribution > 0.0 {
                positive[j] += 1;
            } else if contribution < 0.0 {
                negative[j] += 1;
            }
        }
    }

    let mut rows: Vec<SensitivityRow> = (0..nv)
        .map(|j| {
            let denom = used[j] as f64;
            SensitivityRow {
                input: j,
                mean_abs_elasticity: if used[j] > 0 {
                    sum_abs[j] / denom
                } else {
                    0.0
                },
                min_contribution: if used[j] > 0 {
                    min_contrib[j]
                } else {
                    0.0
                },
                max_contribution: if used[j] > 0 {
                    max_contrib[j]
                } else {
                    0.0
                },
                positive_fraction: if used[j] > 0 {
                    positive[j] as f64 / denom
                } else {
                    0.0
                },
                negative_fraction: if used[j] > 0 {
                    negative[j] as f64 / denom
                } else {
                    0.0
                },
                rows_used: used[j],
            }
        })
        .collect();
    rows.sort_by(|a, b| {
        b.mean_abs_elasticity
            .partial_cmp(&a.mean_abs_elasticity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.input.cmp(&b.input))
    });
    SensitivityReport {
        rows,
        n,
        output_std,
    }
}

/// Population standard deviation, ignoring non-finite values.
fn std_dev(values: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sum = 0.0;
    for &v in values {
        if v.is_finite() {
            n += 1;
            sum += v;
        }
    }
    if n == 0 {
        return f64::NAN;
    }
    let mean = sum / n as f64;
    let var = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / n as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Median / IQR statistics (issue #29 benchmark)
// ---------------------------------------------------------------------------

/// Linear-interpolation quantile (Hyndman–Fan type 7) of unsorted values.
///
/// Non-finite values are ignored. Returns `NaN` for an empty slice. This is the
/// same definition `data` uses for preparation quantiles, so a report's IQR
/// means the same thing as an `--outlier-iqr` bound.
pub fn quantile(values: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let h = (v.len() - 1) as f64 * p.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (h - lo as f64) * (v[hi] - v[lo])
    }
}

/// Median (the 0.5 quantile). `NaN` for an empty or all-non-finite slice.
pub fn median(values: &[f64]) -> f64 {
    quantile(values, 0.5)
}

/// Interquartile range, `Q3 − Q1` (the 0.75 minus the 0.25 quantile).
pub fn iqr(values: &[f64]) -> f64 {
    quantile(values, 0.75) - quantile(values, 0.25)
}

/// One repeat's best result at one complexity level.
#[derive(Clone, Copy, Debug)]
pub struct BenchmarkPoint {
    pub complexity: usize,
    pub train_error: f64,
    pub test_error: f64,
}

/// Median and IQR of the best errors at one complexity, over repeats.
#[derive(Clone, Copy, Debug)]
pub struct BenchmarkRow {
    pub complexity: usize,
    /// Repeats that produced a solution at this complexity.
    pub runs: usize,
    pub train_median: f64,
    pub train_iqr: f64,
    pub test_median: f64,
    pub test_iqr: f64,
}

/// Aggregate repeated Pareto fronts by complexity.
///
/// Within one repeat, if several solutions share a complexity the one with the
/// lowest training error represents that repeat (its test error is reported
/// alongside). Across repeats, each complexity gets the median and IQR of those
/// per-repeat bests, so a single lucky search cannot move the summary.
pub fn benchmark_by_complexity(repeats: &[Vec<BenchmarkPoint>]) -> Vec<BenchmarkRow> {
    use std::collections::BTreeMap;
    let mut train: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut test: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for rep in repeats {
        let mut best: BTreeMap<usize, BenchmarkPoint> = BTreeMap::new();
        for p in rep {
            match best.get(&p.complexity) {
                Some(old) if old.train_error <= p.train_error => {}
                _ => {
                    best.insert(p.complexity, *p);
                }
            }
        }
        for (complexity, p) in best {
            train.entry(complexity).or_default().push(p.train_error);
            test.entry(complexity).or_default().push(p.test_error);
        }
    }
    train
        .into_iter()
        .map(|(complexity, train_errs)| {
            let test_errs = test.remove(&complexity).unwrap_or_default();
            BenchmarkRow {
                complexity,
                runs: train_errs.len(),
                train_median: median(&train_errs),
                train_iqr: iqr(&train_errs),
                test_median: median(&test_errs),
                test_iqr: iqr(&test_errs),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_iqr_and_quantile_use_linear_interpolation() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.0), 1.0);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 1.0), 4.0);
        assert!((quantile(&[1.0, 2.0, 3.0, 4.0], 0.25) - 1.75).abs() < 1e-12);
        assert!((iqr(&[1.0, 2.0, 3.0, 4.0]) - 1.5).abs() < 1e-12);
        assert!(median(&[]).is_nan());
        // Non-finite entries are ignored, not propagated.
        assert_eq!(median(&[1.0, f64::NAN, 3.0]), 2.0);
    }

    #[test]
    fn benchmark_aggregates_per_repeat_bests_by_complexity() {
        let repeats = vec![
            vec![
                BenchmarkPoint {
                    complexity: 2,
                    train_error: 1.0,
                    test_error: 2.0,
                },
                BenchmarkPoint {
                    complexity: 4,
                    train_error: 5.0,
                    test_error: 7.0,
                },
                // A second entry at complexity 2 must not double-count the
                // repeat; the lower training error wins.
                BenchmarkPoint {
                    complexity: 2,
                    train_error: 0.5,
                    test_error: 1.5,
                },
            ],
            vec![BenchmarkPoint {
                complexity: 2,
                train_error: 2.0,
                test_error: 4.0,
            }],
            vec![BenchmarkPoint {
                complexity: 2,
                train_error: 3.0,
                test_error: 6.0,
            }],
        ];
        let rows = benchmark_by_complexity(&repeats);
        assert_eq!(rows.len(), 2);
        let c2 = &rows[0];
        assert_eq!(c2.complexity, 2);
        assert_eq!(c2.runs, 3);
        assert!((c2.train_median - 2.0).abs() < 1e-12, "median {:?}", c2);
        assert!((c2.train_iqr - 1.25).abs() < 1e-12, "iqr {:?}", c2);
        assert!((c2.test_median - 4.0).abs() < 1e-12, "test {:?}", c2);
        assert!((c2.test_iqr - 2.25).abs() < 1e-12, "test iqr {:?}", c2);
        let c4 = &rows[1];
        assert_eq!(c4.runs, 1);
        assert!((c4.train_iqr - 0.0).abs() < 1e-12);
    }

    #[test]
    fn sensitivity_ranks_the_driving_input_over_noise() {
        use crate::expr::{constant, input, node, Op};
        // y = 3*x1; x2 has the same spread but no influence.
        let tree = node(
            Op::Mul,
            vec![constant(3.0), input(0)],
        );
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..40 {
            let x1 = -2.0 + 4.0 * i as f64 / 39.0;
            let x2 = ((i * 37 % 40) as f64 - 20.0) / 10.0;
            x.push(vec![x1, x2]);
            y.push(3.0 * x1);
        }
        let report = sensitivity(
            &tree,
            &Sample {
                x,
                y,
                w: Vec::new(),
                ys: Vec::new(),
            },
            &[1.0, 1.0],
        );
        assert_eq!(report.rows.len(), 2);
        assert_eq!(report.rows[0].input, 0, "x1 must rank first");
        assert!(
            report.rows[0].mean_abs_elasticity > 100.0 * report.rows[1].mean_abs_elasticity,
            "x1 {:?} vs x2 {:?}",
            report.rows[0],
            report.rows[1]
        );
        // For a linear law the standardised elasticity is exactly 1.
        assert!(
            (report.rows[0].mean_abs_elasticity - 1.0).abs() < 1e-9,
            "{:?}",
            report.rows[0]
        );
    }

    #[test]
    fn aicc_punishes_extra_constants() {
        // Same fit quality, more constants: AICc must rise.
        let n = 50;
        let rss = 1.0;
        let (_, small, _) = information_criteria(n, rss, 2);
        let (_, big, _) = information_criteria(n, rss, 10);
        assert!(big > small);
    }

    #[test]
    fn aicc_larger_than_aic() {
        let (aic, aicc, bic) = information_criteria(30, 4.0, 3);
        assert!(aicc > aic, "small-sample correction adds a penalty");
        assert!(bic > aic, "BIC penalises more than AIC at n>7");
    }

    #[test]
    fn durbin_watson_detects_structure() {
        // Alternating residuals: strongly negative autocorrelation -> d near 4.
        let alt: Vec<f64> = (0..40).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!(durbin_watson(&alt) > 3.0);
        // White-ish residuals around 2.
        let mut x = 0.0f64;
        let mut rng = crate::rng::Mt::new(1);
        let noise: Vec<f64> = (0..2000)
            .map(|_| {
                x = 0.9 * x + (rng.next_f64() - 0.5);
                x
            })
            .collect();
        let d = durbin_watson(&noise);
        assert!(d < 2.0, "positively autocorrelated noise has d < 2, got {}", d);
    }

    #[test]
    fn permutation_p_value_is_small_when_observed_wins() {
        let null_best = vec![1.0, 1.2, 0.9, 1.1, 1.3];
        let c = NullControl::summarise(0.01, null_best);
        assert!(c.p_value < 0.2, "p={}", c.p_value);
        let c2 = NullControl::summarise(5.0, vec![1.0, 1.2, 0.9]);
        assert!(c2.p_value > 0.9, "p={}", c2.p_value);
    }

    fn sample(x: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
        Sample {
            x,
            y,
            w: Vec::new(),
            ys: Vec::new(),
        }
    }

    #[test]
    fn chi2_sf_matches_known_quantiles() {
        // df = 2 has the closed form sf(x) = e^{-x/2}.
        let s = chi2_sf(2.0, 2);
        assert!((s - (-1.0f64).exp()).abs() < 1e-9, "sf(2,2)={}", s);
        // Standard 95% quantiles: 3.841459 (df 1) and 18.30704 (df 10).
        assert!(
            (chi2_sf(3.841_459, 1) - 0.05).abs() < 1e-3,
            "{}",
            chi2_sf(3.841_459, 1)
        );
        assert!(
            (chi2_sf(18.307_038, 10) - 0.05).abs() < 1e-3,
            "{}",
            chi2_sf(18.307_038, 10)
        );
        // Monotone decreasing in x.
        assert!(chi2_sf(1.0, 3) > chi2_sf(2.0, 3));
    }

    #[test]
    fn mean_baseline_is_the_training_mean() {
        let train = sample(vec![vec![0.0], vec![1.0], vec![2.0]], vec![1.0, 2.0, 3.0]);
        let test = sample(vec![vec![3.0], vec![4.0], vec![5.0]], vec![4.0, 5.0, 6.0]);
        let b = Baselines::fit(&train, &[test], Metric::Absolute);
        // train: |2-1| + |2-2| + |2-3| over 3 = 2/3.
        assert!((b.mean.train_error - 2.0 / 3.0).abs() < 1e-12, "{}", b.mean.train_error);
        // test: |2-4| + |2-5| + |2-6| over 3 = 3.
        assert!((b.mean.test_error - 3.0).abs() < 1e-12, "{}", b.mean.test_error);
    }

    #[test]
    fn linear_baseline_recovers_an_exact_line() {
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let train = sample(
            x.iter().map(|&v| vec![v]).collect(),
            x.iter().map(|&v| 2.0 + 3.0 * v).collect(),
        );
        let c = least_squares_linear(&train).expect("soluble");
        assert!((c[0] - 2.0).abs() < 1e-9, "intercept {}", c[0]);
        assert!((c[1] - 3.0).abs() < 1e-9, "slope {}", c[1]);
        let test = sample(vec![vec![10.0], vec![20.0]], vec![32.0, 62.0]);
        let b = Baselines::fit(&train, &[test], Metric::Squared);
        assert!(b.linear.train_error < 1e-12, "{}", b.linear.train_error);
        assert!(b.linear.test_error < 1e-12, "{}", b.linear.test_error);
    }

    #[test]
    fn linear_baseline_beats_mean_on_a_line() {
        let train = sample(
            (0..20).map(|i| vec![i as f64]).collect(),
            (0..20).map(|i| 5.0 + 0.5 * i as f64).collect(),
        );
        let b = Baselines::fit(&train, &[], Metric::Squared);
        assert!(b.linear.train_error < b.mean.train_error);
    }

    #[test]
    fn knn_predicts_the_mean_of_the_k_nearest() {
        let train = sample(
            vec![vec![0.0], vec![1.0], vec![10.0]],
            vec![0.0, 10.0, 100.0],
        );
        // Nearest two to 0.2 are x=0 (d=0.04) and x=1 (d=0.64): (0+10)/2 = 5.
        assert!((knn_predict(&train, &[0.2], 2) - 5.0).abs() < 1e-12);
        // k=1 picks the closest point, x=0.
        assert!((knn_predict(&train, &[0.2], 1) - 0.0).abs() < 1e-12);
        assert_eq!(knn_k(100), 5);
        assert_eq!(knn_k(2), 2);
    }

    #[test]
    fn ljung_box_matches_a_hand_computed_alternating_series() {
        // e = +1,-1,+1,-1,... has rho_k = (-1)^k (n-k)/n with the biased
        // estimator, so at n = 100, h = 4:
        // Q = n(n+2) * sum_{k=1..4} rho_k^2 / (n-k)
        //   = 10200 * (0.99^2/99 + 0.98^2/98 + 0.97^2/97 + 0.96^2/96).
        let e: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!((autocorrelation(&e, 1) + 0.99).abs() < 1e-12);
        assert!((autocorrelation(&e, 2) - 0.98).abs() < 1e-12);
        let lb = ljung_box(&e, 4);
        let expected =
            10200.0 * (0.99f64.powi(2) / 99.0 + 0.98f64.powi(2) / 98.0 + 0.97f64.powi(2) / 97.0 + 0.96f64.powi(2) / 96.0);
        assert!(
            (lb.statistic - expected).abs() < 1e-6,
            "{} vs {}",
            lb.statistic,
            expected
        );
        assert_eq!(lb.lags, 4);
        assert!(lb.p_value < 1e-6, "p={}", lb.p_value);
    }

    #[test]
    fn ljung_box_does_not_reject_white_noise() {
        let mut rng = crate::rng::Mt::new(7);
        let e: Vec<f64> = (0..500).map(|_| rng.next_f64() - 0.5).collect();
        let lb = ljung_box(&e, default_ljung_box_lags(500));
        assert!(lb.p_value > 0.01, "p={}", lb.p_value);
    }

    #[test]
    fn breusch_pagan_matches_a_hand_computed_example() {
        // fitted = 0,1,2; resid = 1,2,0 gives u = 1,4,0. The auxiliary OLS
        // gives R² = 1 - 8.1667/8.6667 = 0.057692, so LM = 3 R² = 0.173077.
        let bp = breusch_pagan(&[1.0, 2.0, 0.0], &[0.0, 1.0, 2.0]);
        assert!(
            (bp.statistic - 0.173_076_923).abs() < 1e-6,
            "{}",
            bp.statistic
        );
        assert_eq!(bp.df, 1);
        assert!((bp.p_value - chi2_sf(bp.statistic, 1)).abs() < 1e-12);
    }

    #[test]
    fn breusch_pagan_is_significant_for_growing_variance() {
        let fitted: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let resid: Vec<f64> = fitted
            .iter()
            .map(|&f| if f < 10.0 { 0.1 } else { 2.0 })
            .collect();
        let bp = breusch_pagan(&resid, &fitted);
        assert!(bp.p_value < 0.01, "p={}", bp.p_value);
    }

    #[test]
    fn breusch_pagan_flat_variance_is_not_significant() {
        let fitted: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let resid: Vec<f64> = fitted.iter().map(|_| 1.0).collect();
        let bp = breusch_pagan(&resid, &fitted);
        assert_eq!(bp.p_value, 1.0);
    }

    #[test]
    fn skewness_and_kurtosis_match_hand_checks() {
        // [0,0,0,0,1]: m2 = 0.16, m3 = 0.096, m4 = 0.0832
        // -> skew = 0.096 / 0.16^1.5 = 1.5, excess kurtosis = 0.25.
        let x = [0.0, 0.0, 0.0, 0.0, 1.0];
        assert!((skewness(&x) - 1.5).abs() < 1e-12, "{}", skewness(&x));
        assert!((kurtosis(&x) - 0.25).abs() < 1e-12, "{}", kurtosis(&x));
        // Symmetric data: skew 0, and [-2,-1,0,1,2] has excess kurtosis -1.3.
        let sym = [-2.0, -1.0, 0.0, 1.0, 2.0];
        assert!(skewness(&sym).abs() < 1e-12);
        assert!((kurtosis(&sym) + 1.3).abs() < 1e-12, "{}", kurtosis(&sym));
    }

    #[test]
    fn residual_tests_bundle_is_consistent() {
        let resid: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).sin()).collect();
        let fitted: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let rt = residual_tests(&resid, &fitted);
        assert_eq!(rt.n, 50);
        assert_eq!(rt.lags, 10);
        assert_eq!(rt.ljung_box.lags, 10);
        assert!(rt.durbin_watson.is_finite());
        assert!(rt.ljung_box.statistic.is_finite());
        assert!(rt.breusch_pagan.p_value.is_finite());
        assert!(rt.skewness.is_finite() && rt.kurtosis.is_finite());
    }

    /// `c0 + c1·x + c2·x²`, a shape with three free constants.
    fn poly2_tree() -> Node {
        use crate::expr::{constant, input, node, Op};
        let x = || input(0);
        let add = |a, b| node(Op::Add, vec![a, b]);
        let mul = |a, b| node(Op::Mul, vec![a, b]);
        let x2 = mul(x(), x());
        add(add(constant(0.0), mul(constant(1.0), x())), mul(constant(2.0), x2))
    }

    /// #92: the pre-index implementation, kept verbatim as the exactness oracle.
    fn legacy_bootstrap_error(
        tree: &Node,
        train: &Sample,
        metric: Metric,
        resamples: usize,
        fit_iters: usize,
        rng: &mut crate::rng::Mt,
    ) -> BootstrapError {
        let n = train.len();
        let apparent_pred: Vec<f64> =
            train.x.iter().map(|row| crate::expr::eval(tree, row)).collect();
        let apparent = if n == 0 {
            f64::INFINITY
        } else {
            crate::search::eval_error(tree, train, metric)
        };
        let mut oob_sum = 0.0;
        let mut oob_used = 0usize;
        if n > 0 && resamples > 0 {
            let mut in_bag = vec![false; n];
            let mut bx: Vec<Vec<f64>> = Vec::with_capacity(n);
            let mut by: Vec<f64> = Vec::with_capacity(n);
            for _ in 0..resamples {
                for f in in_bag.iter_mut() {
                    *f = false;
                }
                bx.clear();
                by.clear();
                for _ in 0..n {
                    let j = rng.usize_below(n);
                    in_bag[j] = true;
                    bx.push(train.x[j].clone());
                    by.push(train.y[j]);
                }
                let oob: Vec<usize> = (0..n).filter(|&j| !in_bag[j]).collect();
                if oob.is_empty() {
                    continue;
                }
                let mut t = tree.clone();
                crate::fit::fit(&mut t, &bx, &by, fit_iters);
                let scored = Sample {
                    x: oob.iter().map(|&j| train.x[j].clone()).collect(),
                    y: oob.iter().map(|&j| train.y[j]).collect(),
                    w: Vec::new(),
                    ys: Vec::new(),
                };
                let e = crate::search::eval_error(&t, &scored, metric);
                if e.is_finite() {
                    oob_sum += e;
                    oob_used += 1;
                }
            }
        }
        let oob = if oob_used > 0 {
            oob_sum / oob_used as f64
        } else {
            f64::INFINITY
        };
        let no_information = if n >= 2 && !apparent_pred.is_empty() {
            let mut shuffled = train.y.clone();
            let mut acc = 0.0;
            for _ in 0..NO_INFO_PERMUTATIONS {
                for j in (1..n).rev() {
                    let t = rng.usize_below(j + 1);
                    shuffled.swap(j, t);
                }
                acc += metrics::score_with_constants(
                    &apparent_pred,
                    &shuffled,
                    metric,
                    crate::expr::count_constants(tree),
                );
            }
            acc / NO_INFO_PERMUTATIONS as f64
        } else {
            f64::NAN
        };
        let denom = no_information - apparent;
        let relative_overfitting = if apparent.is_finite()
            && oob.is_finite()
            && denom.is_finite()
            && denom > 0.0
            && oob > apparent
        {
            ((oob - apparent) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let weight = BOOTSTRAP_632 / (1.0 - (1.0 - BOOTSTRAP_632) * relative_overfitting);
        let estimate = if apparent.is_finite() && oob.is_finite() {
            (1.0 - weight) * apparent + weight * oob
        } else if apparent.is_finite() {
            apparent
        } else {
            f64::INFINITY
        };
        BootstrapError {
            resamples,
            used: oob_used,
            source_rows: n,
            resample_rows: n,
            capped: false,
            apparent,
            oob,
            no_information,
            relative_overfitting,
            weight,
            estimate,
        }
    }

    #[test]
    fn index_resampling_is_bit_identical_to_row_cloning_below_the_cap() {
        // #92: 100 rows is well below the BLB threshold.
        let n = 100;
        let train = Sample {
            x: (0..n)
                .map(|i| vec![i as f64 * 0.1, (i as f64 * 0.37).sin(), i as f64 % 7.0])
                .collect(),
            y: (0..n).map(|i| 1.5 + 0.25 * i as f64).collect(),
            w: Vec::new(),
            ys: Vec::new(),
        };
        let mut tree = poly2_tree();
        crate::fit::fit(&mut tree, &train.x, &train.y, 8);

        let new = bootstrap_error(
            &tree,
            &train,
            Metric::Squared,
            20,
            8,
            &mut crate::rng::Mt::new(4242),
        );
        let old = legacy_bootstrap_error(
            &tree,
            &train,
            Metric::Squared,
            20,
            8,
            &mut crate::rng::Mt::new(4242),
        );
        assert!(!new.capped, "100 rows must stay on the exact bootstrap");
        assert_eq!(new.resample_rows, n);
        assert_eq!(new.source_rows, n);
        assert_eq!(new.resamples, old.resamples);
        assert_eq!(new.used, old.used);
        for (a, b, what) in [
            (new.apparent, old.apparent, "apparent"),
            (new.oob, old.oob, "oob"),
            (new.no_information, old.no_information, "no_information"),
            (
                new.relative_overfitting,
                old.relative_overfitting,
                "relative_overfitting",
            ),
            (new.weight, old.weight, "weight"),
            (new.estimate, old.estimate, "estimate"),
        ] {
            assert_eq!(a.to_bits(), b.to_bits(), "{what}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn bootstrap_reports_the_blb_resample_size_above_the_cap() {
        // 20 000 rows puts n^0.7 just above the 1 000-row BLB floor.
        let n = 20_000;
        let train = Sample {
            x: (0..n).map(|i| vec![i as f64]).collect(),
            y: (0..n).map(|i| 2.0 + 0.001 * i as f64).collect(),
            w: Vec::new(),
            ys: Vec::new(),
        };
        let tree = crate::expr::constant(2.0);
        let mut rng = crate::rng::Mt::new(5);
        let b = bootstrap_error(&tree, &train, Metric::Absolute, 1, 0, &mut rng);
        let expected = (n as f64).powf(crate::resample::BLB_GAMMA).floor() as usize;
        assert!(b.capped, "20k rows must engage the BLB cap");
        assert_eq!(b.source_rows, n);
        assert_eq!(b.resample_rows, expected);
        assert!(b.resample_rows < b.source_rows);
    }

    #[test]
    fn bootstrap_632_plus_lies_between_apparent_and_oob() {
        // A quadratic fit to noisy data: the constants are refit on each
        // resample, so the out-of-bag error is at least as large as the
        // apparent (in-sample) error.
        let n = 30;
        let mut noise_rng = crate::rng::Mt::new(12345);
        let xs: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 7.0).collect();
        let ys: Vec<f64> = xs
            .iter()
            .map(|&x| 2.0 + 3.0 * x - 0.5 * x * x + (noise_rng.next_f64() - 0.5) * 8.0)
            .collect();
        let train = Sample {
            x: xs.iter().map(|&v| vec![v]).collect(),
            y: ys,
            w: Vec::new(),
            ys: Vec::new(),
        };
        let mut tree = poly2_tree();
        crate::fit::fit(&mut tree, &train.x, &train.y, 16);

        let mut rng = crate::rng::Mt::new(7);
        let b = bootstrap_error(&tree, &train, Metric::Absolute, 200, 16, &mut rng);

        assert!(b.resamples == 200 && b.used > 0, "used={}", b.used);
        assert!(b.apparent.is_finite(), "apparent={}", b.apparent);
        assert!(b.oob.is_finite(), "oob={}", b.oob);
        assert!(b.no_information.is_finite(), "γ={}", b.no_information);
        // The whole point: the .632+ estimate is a convex combination of the
        // apparent and out-of-bag errors, so it cannot leave that interval.
        let lo = b.apparent.min(b.oob);
        let hi = b.apparent.max(b.oob);
        assert!(
            b.estimate >= lo - 1e-9 && b.estimate <= hi + 1e-9,
            ".632+={} not in [{}, {}]",
            b.estimate,
            lo,
            hi
        );
        assert!(
            (0.632 - 1e-12..=1.0 + 1e-12).contains(&b.weight),
            "weight={}",
            b.weight
        );
        assert!(
            (0.0..=1.0).contains(&b.relative_overfitting),
            "R={}",
            b.relative_overfitting
        );
        // No-information error is larger than an in-sample fit that tracks the
        // target, so `γ − apparent` is positive here.
        assert!(b.no_information > b.apparent);
    }

    #[test]
    fn bootstrap_weight_is_0632_without_overfitting() {
        // A constant model on a constant target has no overfitting to correct:
        // the out-of-bag error equals the apparent one, so R = 0 and the weight
        // is exactly 0.632.
        let train = Sample {
            x: (0..20).map(|i| vec![i as f64]).collect(),
            y: vec![5.0; 20],
            w: Vec::new(),
            ys: Vec::new(),
        };
        let mut tree = crate::expr::constant(5.0);
        crate::fit::fit(&mut tree, &train.x, &train.y, 8);
        let mut rng = crate::rng::Mt::new(3);
        let b = bootstrap_error(&tree, &train, Metric::Absolute, 50, 8, &mut rng);
        assert!(b.apparent.abs() < 1e-12, "apparent={}", b.apparent);
        assert!(b.oob.abs() < 1e-12, "oob={}", b.oob);
        assert_eq!(b.relative_overfitting, 0.0);
        assert!((b.weight - BOOTSTRAP_632).abs() < 1e-12, "w={}", b.weight);
        assert!(b.estimate.abs() < 1e-12, "est={}", b.estimate);
    }

    #[test]
    fn bootstrap_handles_zero_resamples_and_zero_constants() {
        let train = Sample {
            x: (0..10).map(|i| vec![i as f64]).collect(),
            y: (0..10).map(|i| 2.0 * i as f64 + 1.0).collect(),
            w: Vec::new(),
            ys: Vec::new(),
        };
        // A tree with no constants: the refit is a no-op, but the estimator
        // still fits and scores.
        let tree = crate::expr::input(0);
        let mut rng = crate::rng::Mt::new(1);
        let b = bootstrap_error(&tree, &train, Metric::Squared, 0, 8, &mut rng);
        assert_eq!(b.resamples, 0);
        assert_eq!(b.used, 0);
        assert!(!b.oob.is_finite(), "oob={}", b.oob);

        let mut rng = crate::rng::Mt::new(1);
        let b = bootstrap_error(&tree, &train, Metric::Squared, 20, 8, &mut rng);
        assert!(b.used > 0);
        assert!(b.oob.is_finite());
    }

    #[test]
    fn mean_sem_uses_the_finite_population_correction() {
        // Losses 1 and 2 sampled from N = 5: s = sqrt(0.5), q = 2.
        let pred = vec![1.0, 2.0];
        let y = vec![0.0, 0.0];
        let sem = mean_sem(Metric::Absolute, &pred, &y, 5);
        let expected = (0.5f64).sqrt() / (2.0f64).sqrt() * (3.0f64 / 4.0).sqrt();
        assert!((sem - expected).abs() < 1e-12, "{sem} vs {expected}");
        // No sampling error to report when every row was scored, and the
        // correction only exists for plain mean losses.
        assert!(mean_sem(Metric::Absolute, &pred, &y, 2).is_nan());
        assert!(mean_sem(Metric::Median, &pred, &y, 5).is_nan());
    }

    #[test]
    fn sample_indices_are_distinct_deterministic_and_sorted() {
        let a = sample_indices(10_000, 500, 42);
        let b = sample_indices(10_000, 500, 42);
        assert_eq!(a, b);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "not sorted/unique");
        let mut sorted = a.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 500);
    }

    #[test]
    fn knn_index_falls_back_for_ragged_rows_and_wider_queries() {
        // A short training row: `Sample::new` is public and accepts ragged
        // rows, so the tree must not be built (or indexed) as rectangular.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..300usize {
            x.push(vec![i as f64 * 0.01, (i % 7) as f64, 1.0]);
            y.push((i % 5) as f64);
        }
        x[17] = vec![0.5, -0.25];
        let ragged = Sample::new(x, y);
        let mut index = KnnIndex::new(&ragged);
        for i in 0..40usize {
            let q = vec![i as f64 * 0.02, (i % 3) as f64, 0.5];
            let got = index.predict(&q, 5);
            let want = knn_predict(&ragged, &q, 5);
            assert_eq!(got.to_bits(), want.to_bits(), "short training row");
        }

        // A query wider than a rectangular reference: the extra inputs must be
        // ignored exactly as `knn_predict`'s `zip` does.
        let rect = sample(
            (0..300)
                .map(|i| vec![i as f64 * 0.01, (i % 7) as f64, 1.0])
                .collect(),
            (0..300).map(|i| (i % 5) as f64).collect(),
        );
        let mut index = KnnIndex::new(&rect);
        let wide = vec![0.2, 1.0, 0.5, 99.0, -99.0];
        assert_eq!(
            index.predict(&wide, 5).to_bits(),
            knn_predict(&rect, &wide, 5).to_bits(),
            "wider query"
        );
    }
}
