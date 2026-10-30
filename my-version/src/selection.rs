//! Model selection on a validation slice (issue #111).
//!
//! The test block is a **lockbox**: it may be scored once, after a candidate is
//! chosen, and must never influence which candidate wins. Selection therefore
//! ranks on `train`, `validation` or `aicc` only. `Test` exists here for one
//! purpose — to demonstrate, in a test, the optimism of ranking by the block
//! you are about to report.

use crate::trading_gpu::{self, ValidMask};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectBy {
    Train,
    Validation,
    Aicc,
}

impl SelectBy {
    pub fn parse(s: &str) -> Result<SelectBy, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "train" => Ok(SelectBy::Train),
            "validation" | "val" => Ok(SelectBy::Validation),
            "aicc" | "aic" => Ok(SelectBy::Aicc),
            other => Err(format!(
                "unknown --select-by '{}' (expected train, validation or aicc)",
                other
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SelectBy::Train => "train",
            SelectBy::Validation => "validation",
            SelectBy::Aicc => "aicc",
        }
    }
}

/// The scores a candidate exposes. `test` is recorded for reporting and for
/// the leak demonstration; [`select`] never reads it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Candidate {
    pub train: f64,
    pub validation: Option<f64>,
    pub test: Option<f64>,
    pub aicc: f64,
}

fn finite_or_worst(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::INFINITY
    }
}

impl Candidate {
    fn score(self, by: SelectBy) -> f64 {
        match by {
            SelectBy::Train => finite_or_worst(self.train),
            SelectBy::Validation => finite_or_worst(self.validation.unwrap_or(f64::INFINITY)),
            SelectBy::Aicc => finite_or_worst(self.aicc),
        }
    }
}

/// Index of the winning candidate under `by`. `None` only for an empty slice.
///
/// `by == SelectBy::Validation` requires every candidate to carry a validation
/// score; a missing score is treated as infinitely bad, and the caller should
/// have rejected that combination earlier (the CLI does).
pub fn select(candidates: &[Candidate], by: SelectBy) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.score(by)
                .partial_cmp(&b.score(by))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

/// Indices ordered best-first under `by`, stable on ties.
pub fn order(candidates: &[Candidate], by: SelectBy) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..candidates.len()).collect();
    idx.sort_by(|&a, &b| {
        candidates[a]
            .score(by)
            .partial_cmp(&candidates[b].score(by))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

// ---------------------------------------------------------------------------
// #308: banded finalist selection around the P&L screening score
// ---------------------------------------------------------------------------

/// Smallest band cut. `K = max(K_MIN, P / K_DIVISOR)`.
pub const BAND_K_MIN: usize = 8;
/// Population divisor for the band cut.
pub const BAND_K_DIVISOR: usize = 16;
/// Absolute floor of `epsilon`.
pub const BAND_EPSILON_FLOOR: f64 = 2e-3;
/// Relative part of `epsilon`: `0.5 % * |J_K|`.
pub const BAND_EPSILON_RELATIVE: f64 = 5e-3;

/// Rows the exact long-only rescore sees. All slices are row-aligned.
///
/// The selector canonicalises the rows to `(group, time, original index)`
/// before calling [`trading_gpu::reduce`], so the reported bits do not depend
/// on the order in which the caller stored the validation block.
#[derive(Clone, Copy, Debug)]
pub struct BandRows<'a> {
    pub targets: &'a [f64],
    pub groups: &'a [u64],
    pub times: &'a [f64],
}

impl<'a> BandRows<'a> {
    fn validate(&self) -> Result<(), String> {
        let n = self.targets.len();
        if self.groups.len() != n || self.times.len() != n {
            return Err(format!(
                "band rows length mismatch: targets {}, groups {}, times {}",
                n,
                self.groups.len(),
                self.times.len()
            ));
        }
        Ok(())
    }
}

/// One candidate entering the band.
///
/// `surrogate` is the screening score: larger is better. In the GPU path it is
/// the device reduction over `f32` score rows; it is deliberately *not* the
/// reported value. `scores` is the full-row exact `f64` score vector used by
/// the rescore, and `threshold` is the exact type-7 threshold for those rows.
#[derive(Clone, Copy, Debug)]
pub struct BandCandidate<'a> {
    /// Caller's stable index (reported as `index` in the provenance).
    pub index: usize,
    /// Screening score, higher is better.
    pub surrogate: f64,
    /// Expression complexity (lower wins a tie).
    pub complexity: usize,
    /// Full-row `f64` candidate scores, aligned with [`BandRows`].
    pub scores: &'a [f64],
    /// Exact long threshold for this candidate.
    pub threshold: f64,
}

/// Parameters of the band and the exact rescore.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandOptions {
    pub k_min: usize,
    pub k_divisor: usize,
    pub epsilon_floor: f64,
    pub epsilon_relative: f64,
    /// Per-side cost in basis points used by the exact CPU rescore.
    pub cost_bps: f64,
    /// Canonical segment length of the exact reduction.
    pub segment_rows: usize,
    /// Cash return passed to the exact reduction.
    pub cash_return: f64,
    /// Compute the exhaustive exact ranking as an audit. This is the only way
    /// to report `recall_band` honestly; the strict default fails the run when
    /// the band misses the exhaustive winner.
    pub verify_exhaustive: bool,
    /// With `verify_exhaustive`, return an error instead of a selection with
    /// `recall_band < 1`.
    pub fail_on_miss: bool,
}

impl Default for BandOptions {
    fn default() -> Self {
        BandOptions {
            k_min: BAND_K_MIN,
            k_divisor: BAND_K_DIVISOR,
            epsilon_floor: BAND_EPSILON_FLOOR,
            epsilon_relative: BAND_EPSILON_RELATIVE,
            cost_bps: trading_gpu::DEFAULT_COST_BPS,
            segment_rows: trading_gpu::DEFAULT_SEGMENT_ROWS,
            cash_return: 0.0,
            verify_exhaustive: true,
            fail_on_miss: true,
        }
    }
}

/// The exact CPU `f64` objective of one candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactScore {
    /// Additive long-only net P&L over the full rows.
    pub net: f64,
    /// `sum |position change|`; the final exit is not included, matching
    /// [`trading_gpu::PnlTotals::turnover`].
    pub turnover: f64,
}

/// The deterministic total order over exact scores.
///
/// Best first: higher net, then lower turnover, then lower complexity, then
/// lower index. Non-finite values have a fixed, transitive placement:
///
/// * net `+inf` ranks above every finite net; `-inf` and `NaN` rank below it;
/// * turnover `-inf` ranks above every finite turnover; `+inf` and `NaN` rank
///   below it.
///
/// Within a class `f64::total_cmp` is used, so the comparison is transitive even
/// when infinities or NaNs are present.
pub fn exact_cmp(a: &ExactScore, b: &ExactScore) -> std::cmp::Ordering {
    fn net_class(v: f64) -> u8 {
        if v == f64::INFINITY {
            0
        } else if v.is_finite() {
            1
        } else {
            2
        }
    }
    fn turnover_class(v: f64) -> u8 {
        if v == f64::NEG_INFINITY {
            0
        } else if v.is_finite() {
            1
        } else {
            2
        }
    }

    net_class(a.net)
        .cmp(&net_class(b.net))
        .then_with(|| {
            if a.net.is_finite() && b.net.is_finite() {
                b.net.total_cmp(&a.net)
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .then_with(|| turnover_class(a.turnover).cmp(&turnover_class(b.turnover)))
        .then_with(|| {
            if a.turnover.is_finite() && b.turnover.is_finite() {
                a.turnover.total_cmp(&b.turnover)
            } else {
                std::cmp::Ordering::Equal
            }
        })
}

fn exact_order(a: &BandFinalist, b: &BandFinalist) -> std::cmp::Ordering {
    let ea = a.exact.expect("ranked finalists have an exact score");
    let eb = b.exact.expect("ranked finalists have an exact score");
    exact_cmp(&ea, &eb)
        .then_with(|| a.complexity.cmp(&b.complexity))
        .then_with(|| a.index.cmp(&b.index))
}

/// One candidate's band and exact-rescore provenance.
#[derive(Clone, Copy, Debug)]
pub struct BandFinalist {
    pub index: usize,
    pub surrogate: f64,
    pub complexity: usize,
    pub threshold: f64,
    /// Zero-based rank on the screening score, all candidates.
    pub surrogate_rank: usize,
    /// Admitted to the exact rescore.
    pub band: bool,
    /// Exact CPU `f64` score; `None` when the candidate was outside the band
    /// and the exhaustive audit was disabled.
    pub exact: Option<ExactScore>,
    /// Zero-based rank on the exact objective. With the exhaustive audit this
    /// is a rank over all candidates; otherwise over the admitted set only.
    pub exact_rank: Option<usize>,
}

/// Result of the band protocol.
#[derive(Clone, Debug)]
pub struct BandSelection {
    /// All candidates in caller order, annotated with band membership.
    pub candidates: Vec<BandFinalist>,
    /// Caller index of the reported finalist.
    pub selected_index: usize,
    /// Exact score of the reported finalist.
    pub selected_exact: ExactScore,
    /// Screening rank of the reported finalist (zero based).
    pub surrogate_rank: usize,
    /// Exact rank of the reported finalist (zero based).
    pub exact_rank: usize,
    /// Candidate at rank `K` (`J_K`), i.e. the cut value.
    pub cut_surrogate: f64,
    /// `K`, the rank of the cut candidate (`1..=P`).
    pub cut_rank: usize,
    /// Measured band width: `max(epsilon_floor, epsilon_relative * |J_K|)`.
    pub epsilon: f64,
    /// Admit when `surrogate >= cut_surrogate - epsilon`.
    pub band_lower: f64,
    /// Number of candidates exactly rescored.
    pub finalists_considered: usize,
    /// Fraction of the exhaustive exact winner that the band contained, when
    /// the exhaustive audit ran.
    pub recall_band: Option<f64>,
    /// Always `cpu_f64`: the reported objective is not a device estimate.
    pub scored_by: &'static str,
}

impl BandSelection {
    /// The reported finalist's annotation.
    pub fn selected(&self) -> &BandFinalist {
        self.candidates
            .iter()
            .find(|c| c.index == self.selected_index)
            .expect("selected index is present in candidates")
    }

    /// Machine-readable provenance block (used by `rcsr walkforward --json`).
    pub fn json(&self) -> String {
        let candidates: Vec<String> = self
            .candidates
            .iter()
            .map(|c| {
                format!(
                    "{{\"index\":{},\"surrogate\":{},\"surrogate_rank\":{},\
                     \"complexity\":{},\"threshold\":{},\"band\":{},\
                     \"exact\":{},\"exact_rank\":{}}}",
                    c.index,
                    jnum(c.surrogate),
                    c.surrogate_rank,
                    c.complexity,
                    jnum(c.threshold),
                    c.band,
                    c.exact
                        .map(|e| format!(
                            "{{\"net\":{},\"turnover\":{}}}",
                            jnum(e.net),
                            jnum(e.turnover)
                        ))
                        .unwrap_or_else(|| "null".to_string()),
                    c.exact_rank
                        .map(|r| r.to_string())
                        .unwrap_or_else(|| "null".to_string())
                )
            })
            .collect();
        format!(
            "{{\"finalist\":{},\"surrogate_rank\":{},\"exact_rank\":{},\
             \"band\":{{\"cut_rank\":{},\"cut_surrogate\":{},\"epsilon\":{},\
             \"lower\":{},\"finalists_considered\":{}}},\
             \"recall_band\":{},\"scored_by\":{},\
             \"selected\":{{\"net\":{},\"turnover\":{},\"complexity\":{},\
             \"index\":{}}},\"candidates\":[{}]}}",
            self.selected_index,
            self.surrogate_rank,
            self.exact_rank,
            self.cut_rank,
            jnum(self.cut_surrogate),
            jnum(self.epsilon),
            jnum(self.band_lower),
            self.finalists_considered,
            self.recall_band.map_or("null".to_string(), jnum),
            jstr(self.scored_by),
            jnum(self.selected_exact.net),
            jnum(self.selected_exact.turnover),
            self.selected().complexity,
            self.selected_index,
            candidates.join(",")
        )
    }
}

fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            _ => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Screening order: finite scores descending, then non-finite scores, ties by
/// caller index. This is a total order independent of batch composition.
fn surrogate_order(candidates: &[BandCandidate<'_>]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..candidates.len()).collect();
    idx.sort_by(|&a, &b| {
        let sa = candidates[a].surrogate;
        let sb = candidates[b].surrogate;
        let ca = if sa.is_finite() { 0u8 } else { 1 };
        let cb = if sb.is_finite() { 0u8 } else { 1 };
        ca.cmp(&cb)
            .then_with(|| {
                if ca == 0 {
                    sb.total_cmp(&sa)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| candidates[a].index.cmp(&candidates[b].index))
    });
    idx
}

/// Exact CPU `f64` long-only rescore over the full rows.
///
/// The rows are first filtered with the shared [`ValidMask`] (score **and**
/// target finite) and sorted to the canonical `(group, time, row)` order. The
/// reduction is [`trading_gpu::reduce`], which is the same CPU `f64` arithmetic
/// the report uses, including one final-exit charge per group.
fn canonical_rows(
    rows: &BandRows<'_>,
    scores: &[f64],
    options: &BandOptions,
) -> Result<(Vec<f64>, Vec<f64>, Vec<u64>, trading_gpu::PnlParams), String> {
    rows.validate()?;
    if scores.len() != rows.targets.len() {
        return Err(format!(
            "candidate has {} scores for {} band rows",
            scores.len(),
            rows.targets.len()
        ));
    }
    if options.segment_rows == 0 {
        return Err("segment_rows must be at least 1".to_string());
    }

    let mask = ValidMask::from_finites(scores, rows.targets);
    let mut order: Vec<usize> = mask.indices().collect();
    order.sort_by(|&a, &b| {
        rows.groups[a]
            .cmp(&rows.groups[b])
            .then_with(|| rows.times[a].total_cmp(&rows.times[b]))
            .then_with(|| a.cmp(&b))
    });

    let mut ord_scores = Vec::with_capacity(order.len());
    let mut ord_returns = Vec::with_capacity(order.len());
    let mut ord_groups = Vec::with_capacity(order.len());
    for &i in &order {
        ord_scores.push(scores[i]);
        ord_returns.push(rows.targets[i]);
        ord_groups.push(rows.groups[i]);
    }
    let params = trading_gpu::PnlParams {
        threshold: 0.0,
        per_side_bps: options.cost_bps,
        segment_rows: options.segment_rows,
        cash_return: options.cash_return,
    };
    Ok((ord_scores, ord_returns, ord_groups, params))
}

/// Score one candidate with the screening path.
///
/// With `--features cuda` and a device this is the fused reduction, which
/// converts score rows to `f32` and accumulates the P&L in `f64`; otherwise the
/// identical canonical CPU reduction runs and the screen is exact. The value is
/// a *screening* score only: the reported finalist is always the CPU `f64`
/// rescore from [`select_band`].
pub fn screening_net(
    rows: &BandRows<'_>,
    scores: &[f64],
    threshold: f64,
    options: &BandOptions,
) -> Result<f64, String> {
    if !threshold.is_finite() {
        return Err("the screening score requires a finite threshold".to_string());
    }
    let (ord_scores, ord_returns, ord_groups, mut params) =
        canonical_rows(rows, scores, options)?;
    params.threshold = threshold;
    #[cfg(feature = "cuda")]
    if let Some(report) = trading_gpu::reduce_device(
        &ord_scores,
        &ord_returns,
        &ord_groups,
        &params,
        256,
        65_536,
    ) {
        return Ok(report.strategy.net);
    }
    let report = trading_gpu::reduce(&ord_scores, &ord_returns, &ord_groups, &params)?;
    Ok(report.strategy.net)
}

/// Exact CPU `f64` long-only rescore over the full rows.
pub fn rescore_exact(
    rows: &BandRows<'_>,
    scores: &[f64],
    threshold: f64,
    options: &BandOptions,
) -> Result<ExactScore, String> {
    if !threshold.is_finite() {
        return Err("the exact rescore requires a finite threshold".to_string());
    }
    let (ord_scores, ord_returns, ord_groups, mut params) =
        canonical_rows(rows, scores, options)?;
    params.threshold = threshold;
    let report = trading_gpu::reduce(&ord_scores, &ord_returns, &ord_groups, &params)?;
    Ok(ExactScore {
        net: report.strategy.net,
        turnover: report.strategy.turnover,
    })
}

/// Select the finalist with the band protocol.
///
/// 1. rank every candidate on the screening score;
/// 2. set `K = min(P, max(k_min, P / k_divisor))` and
///    `epsilon = max(epsilon_floor, epsilon_relative * |J_K|)`;
/// 3. admit every finite candidate with `surrogate >= J_K - epsilon`;
/// 4. rescore the admitted set exactly on the full rows in CPU `f64`;
/// 5. take the exact argmax under [`exact_cmp`] (`net`, `turnover`,
///    `complexity`, `index`).
///
/// With `verify_exhaustive` the function also rescores every candidate and
/// reports `recall_band`; when `fail_on_miss` is set a band that does not
/// contain the exhaustive exact winner returns an error rather than a wrong
/// finalist.
pub fn select_band(
    rows: &BandRows<'_>,
    candidates: &[BandCandidate<'_>],
    options: &BandOptions,
) -> Result<BandSelection, String> {
    rows.validate()?;
    if candidates.is_empty() {
        return Err("band selection needs at least one candidate".to_string());
    }
    if options.k_min == 0 || options.k_divisor == 0 {
        return Err("band k_min and k_divisor must be at least 1".to_string());
    }

    let surrogate_idx = surrogate_order(candidates);
    let mut ranked = vec![None::<BandFinalist>; candidates.len()];
    for (rank, &i) in surrogate_idx.iter().enumerate() {
        let c = &candidates[i];
        ranked[i] = Some(BandFinalist {
            index: c.index,
            surrogate: c.surrogate,
            complexity: c.complexity,
            threshold: c.threshold,
            surrogate_rank: rank,
            band: false,
            exact: None,
            exact_rank: None,
        });
    }
    let mut ranked: Vec<BandFinalist> = ranked
        .into_iter()
        .map(|c| c.expect("every candidate is ranked"))
        .collect();

    let finite = surrogate_idx
        .iter()
        .filter(|&&i| candidates[i].surrogate.is_finite())
        .count();
    if finite == 0 {
        return Err(
            "band selection needs at least one finite screening score; \
             a device/surrogate failure must be fixed before selection"
                .to_string(),
        );
    }
    let cut_rank = options
        .k_min
        .max(candidates.len() / options.k_divisor)
        .min(finite);
    let cut_surrogate = candidates[surrogate_idx[cut_rank - 1]].surrogate;
    let epsilon = options
        .epsilon_floor
        .max(options.epsilon_relative * cut_surrogate.abs());
    let band_lower = cut_surrogate - epsilon;

    for (i, c) in candidates.iter().enumerate() {
        if c.surrogate.is_finite() && c.surrogate >= band_lower {
            ranked[i].band = true;
        }
    }

    // Exact rescore. The band is always scored; the exhaustive audit adds the
    // out-of-band candidates so `recall_band` is a real measurement.
    for (i, c) in candidates.iter().enumerate() {
        if ranked[i].band || options.verify_exhaustive {
            ranked[i].exact = Some(rescore_exact(rows, c.scores, c.threshold, options)?);
        }
    }

    // Rank the admitted set (and, with the audit, every rescored candidate).
    let mut exact_idx: Vec<usize> = (0..candidates.len())
        .filter(|&i| ranked[i].exact.is_some())
        .collect();
    exact_idx.sort_by(|&a, &b| exact_order(&ranked[a], &ranked[b]));
    for (rank, &i) in exact_idx.iter().enumerate() {
        ranked[i].exact_rank = Some(rank);
    }

    let winner = *exact_idx
        .first()
        .ok_or_else(|| "band selection rescored no candidates".to_string())?;
    let exact_winner = if options.verify_exhaustive {
        // The exhaustive winner is the first among every rescored candidate,
        // because the audit rescored all of them.
        Some(winner)
    } else {
        None
    };

    let recall_band = exact_winner.map(|w| {
        // Recall is 1 when the band contains the exhaustive exact winner.
        if ranked[w].band {
            1.0
        } else {
            0.0
        }
    });
    if let (Some(recall), true) = (recall_band, options.fail_on_miss) {
        if recall < 1.0 {
            let w = exact_winner.expect("verified recall has a winner");
            return Err(format!(
                "band recall failure: exhaustive exact winner candidate {} \
                 (surrogate rank {}) is outside the band (cut rank {}, \
                 epsilon {}); increase epsilon or fix the screening score",
                ranked[w].index, ranked[w].surrogate_rank, cut_rank, epsilon
            ));
        }
    }

    // The reported finalist is the exact best *inside* the band. With the
    // audit passing this is also the exhaustive best.
    let selected = exact_idx
        .iter()
        .copied()
        .find(|&i| ranked[i].band)
        .ok_or_else(|| "band contained no rescored candidate".to_string())?;
    let selected_exact = ranked[selected].exact.expect("selected is rescored");
    let selected_rank = ranked[selected]
        .exact_rank
        .expect("selected candidate has an exact rank");
    let selected_surrogate_rank = ranked[selected].surrogate_rank;
    let finalists_considered = ranked.iter().filter(|r| r.band).count();

    Ok(BandSelection {
        candidates: ranked,
        selected_index: candidates[selected].index,
        selected_exact,
        surrogate_rank: selected_surrogate_rank,
        exact_rank: selected_rank,
        cut_surrogate,
        cut_rank,
        epsilon,
        band_lower,
        finalists_considered,
        recall_band,
        scored_by: "cpu_f64",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_and_aicc_can_disagree() {
        let candidates = [
            Candidate {
                train: 0.10,
                validation: Some(0.50),
                test: Some(0.10),
                aicc: 1.0,
            },
            Candidate {
                train: 0.12,
                validation: Some(0.30),
                test: Some(0.20),
                aicc: 0.9,
            },
        ];
        assert_eq!(select(&candidates, SelectBy::Validation), Some(1));
        assert_eq!(select(&candidates, SelectBy::Aicc), Some(1));
        assert_eq!(select(&candidates, SelectBy::Train), Some(0));
    }

    #[test]
    fn a_missing_validation_score_is_never_selected() {
        let candidates = [
            Candidate {
                train: 0.01,
                validation: None,
                test: Some(0.01),
                aicc: 1.0,
            },
            Candidate {
                train: 0.9,
                validation: Some(0.8),
                test: Some(0.8),
                aicc: 9.0,
            },
        ];
        assert_eq!(select(&candidates, SelectBy::Validation), Some(1));
    }
}
