//! Validation gates (#257): a noise-scaled supported front, per-member
//! comparisons against the reference rungs, and an explicit verdict.
//!
//! The in-search archive is untouched. After the search, the identified
//! members with a finite validation loss are walked in canonical cost order
//! starting from the reference rung `R`: a member is *supported* when its
//! validation loss improves on the last supported entry by more than `z`
//! Driscoll-Kraay standard errors of the paired per-row differential. This is
//! `Archive::canonicalize()` with "strictly improves" replaced by "improves by
//! more than z SE". Test predictions are scored for reporting and never enter
//! the walk, the verdict or the selection.

use crate::metrics::Metric;
use crate::noise::{
    dk_mean_se_indexed, normal_quantile, row_decomposable, row_losses, KeyIndex, LagSource, MeanSe,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Better,
    Equivalent,
    Worse,
}

impl Cmp {
    pub fn name(self) -> &'static str {
        match self {
            Cmp::Better => "better",
            Cmp::Equivalent => "equivalent",
            Cmp::Worse => "worse",
        }
    }
}

/// `better` when `delta > z*se`, `worse` when `delta < -z*se`, else `equivalent`.
/// With `se == 0` the threshold is zero.
pub fn classify(delta: f64, se: f64, z: f64) -> Cmp {
    if delta.is_nan() || se.is_nan() {
        return Cmp::Equivalent;
    }
    let threshold = z * se;
    if delta > threshold {
        Cmp::Better
    } else if delta < -threshold {
        Cmp::Worse
    } else {
        Cmp::Equivalent
    }
}

/// Member versus one reference: `delta = mean(loss_reference - loss_member)`,
/// positive when the member is better.
#[derive(Clone, Debug)]
pub struct Comparison {
    pub reference: &'static str,
    pub delta: f64,
    pub se: f64,
    pub verdict: Cmp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Signal,
    NoSignal,
    Unvalidated,
    NotGated,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Signal => "signal",
            Verdict::NoSignal => "no_signal",
            Verdict::Unvalidated => "unvalidated",
            Verdict::NotGated => "not_gated",
        }
    }
}

/// One front member: archive rank, cost, structural hash, expression, the
/// structural identification flag and its predictions on the two slices.
pub struct MemberInput<'a> {
    pub rank: usize,
    pub cost: f64,
    pub hash: u64,
    pub expression: &'a str,
    pub identified: bool,
    /// False when the member must never be selected for a reason outside the
    /// loss comparison (negative return-space R2 on the selection sample).
    pub presentable: bool,
    pub val_pred: &'a [f64],
    pub test_pred: &'a [f64],
}

/// Reference predictions on both slices. `gate` indexes the reference the
/// verdict is taken against.
pub struct References {
    pub names: Vec<&'static str>,
    pub val: Vec<Vec<f64>>,
    pub test: Vec<Vec<f64>>,
    pub gate: usize,
}

/// Targets, weights and time keys of one slice.
pub struct RowsView {
    pub y: Vec<f64>,
    /// Empty means every row weighs 1.
    pub w: Vec<f64>,
    pub index: KeyIndex,
    pub lag: usize,
    pub lag_source: LagSource,
}

impl RowsView {
    pub fn new(y: Vec<f64>, w: Vec<f64>, keys: &[f64], lag: usize, lag_source: LagSource) -> RowsView {
        let index = KeyIndex::new(keys);
        RowsView { y, w, index, lag, lag_source }
    }

    /// One key per row, lag zero.
    pub fn iid(y: Vec<f64>, w: Vec<f64>) -> RowsView {
        let n = y.len();
        RowsView { y, w, index: KeyIndex::per_row(n), lag: 0, lag_source: LagSource::Iid }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GateConfig {
    pub metric: Metric,
    pub z: f64,
    pub alpha: f64,
}

#[derive(Clone, Debug, Default)]
pub struct MemberGate {
    pub identified: bool,
    /// Identified and presentable with a finite validation loss: part of the walk.
    pub eligible: bool,
    pub supported: bool,
    /// Label of the entry that blocked it, `delta` and `se`.
    pub not_better_than: Option<(String, f64, f64)>,
    pub validation: Vec<Comparison>,
    pub test: Vec<Comparison>,
}

pub struct GateReport {
    pub verdict: Verdict,
    pub selected_rank: Option<usize>,
    pub supported_ranks: Vec<usize>,
    pub reference: &'static str,
    pub z: f64,
    pub z_null: f64,
    pub alpha: f64,
    pub eligible: usize,
    /// Noise of the paired differential between the reference and the selected
    /// member, else the eligible member with the lowest validation loss.
    pub noise: MeanSe,
    pub lag_source: LagSource,
    pub members: Vec<MemberGate>,
}

fn paired(reference: &[f64], member: &[f64], view: &RowsView) -> MeanSe {
    let d: Vec<f64> = reference.iter().zip(member).map(|(a, b)| a - b).collect();
    dk_mean_se_indexed(&d, &view.w, &view.index, view.lag)
}

fn mean_loss(loss: &[f64], w: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut sum = 0.0;
    for (i, &l) in loss.iter().enumerate() {
        let wi = if w.is_empty() { 1.0 } else { w[i] };
        total += wi;
        sum += wi * l;
    }
    sum / total
}

fn finite(loss: &[f64]) -> bool {
    loss.iter().all(|v| v.is_finite())
}

fn empty_report(verdict: Verdict, refs: &References, cfg: &GateConfig, n: usize, view: &RowsView) -> GateReport {
    let mut noise = MeanSe::empty();
    noise.rows = view.y.len();
    noise.timestamps = view.index.groups;
    noise.lag = view.lag;
    GateReport {
        verdict,
        selected_rank: None,
        supported_ranks: Vec::new(),
        reference: refs.names.get(refs.gate).copied().unwrap_or("mean"),
        z: cfg.z,
        z_null: f64::NAN,
        alpha: cfg.alpha,
        eligible: 0,
        noise,
        lag_source: view.lag_source,
        members: vec![MemberGate::default(); n],
    }
}

/// Run the gates. The result is a pure function of the member predictions, the
/// slice targets, weights and keys, and the configuration; the order of
/// `members` does not matter.
pub fn evaluate(
    members: &[MemberInput],
    refs: &References,
    val: &RowsView,
    test: &RowsView,
    cfg: &GateConfig,
) -> GateReport {
    let n = members.len();
    if !row_decomposable(cfg.metric) {
        return empty_report(Verdict::NotGated, refs, cfg, n, val);
    }
    let have_val = !val.y.is_empty();
    let losses = |preds: &[f64], view: &RowsView| -> Option<Vec<f64>> {
        if preds.len() != view.y.len() {
            return None;
        }
        row_losses(preds, &view.y, &view.w, cfg.metric)
    };
    let ref_val: Vec<Option<Vec<f64>>> = refs.val.iter().map(|p| losses(p, val)).collect();
    let ref_test: Vec<Option<Vec<f64>>> = refs.test.iter().map(|p| losses(p, test)).collect();
    let member_val: Vec<Option<Vec<f64>>> = members
        .iter()
        .map(|m| losses(m.val_pred, val).filter(|l| finite(l)))
        .collect();
    let member_test: Vec<Option<Vec<f64>>> = members
        .iter()
        .map(|m| losses(m.test_pred, test).filter(|l| finite(l)))
        .collect();

    let mut gates: Vec<MemberGate> = members
        .iter()
        .enumerate()
        .map(|(i, m)| MemberGate {
            identified: m.identified,
            eligible: have_val && m.identified && m.presentable && member_val[i].is_some(),
            ..MemberGate::default()
        })
        .collect();

    // Comparisons against every reference, on both slices.
    for (i, g) in gates.iter_mut().enumerate() {
        let compare = |member: &Option<Vec<f64>>, refl: &[Option<Vec<f64>>], view: &RowsView| -> Vec<Comparison> {
            let Some(member) = member else { return Vec::new() };
            refs.names
                .iter()
                .zip(refl)
                .filter_map(|(name, r)| {
                    let r = r.as_ref().filter(|l| finite(l))?;
                    let m = paired(r, member, view);
                    Some(Comparison {
                        reference: name,
                        delta: m.mean,
                        se: m.se,
                        verdict: classify(m.mean, m.se, cfg.z),
                    })
                })
                .collect()
        };
        if have_val {
            g.validation = compare(&member_val[i], &ref_val, val);
        }
        g.test = compare(&member_test[i], &ref_test, test);
    }
    if !have_val {
        let mut rep = empty_report(Verdict::Unvalidated, refs, cfg, 0, val);
        rep.members = gates;
        return rep;
    }
    let Some(gate_loss) = ref_val.get(refs.gate).and_then(|l| l.as_ref()).filter(|l| finite(l)) else {
        let mut rep = empty_report(Verdict::NotGated, refs, cfg, 0, val);
        rep.members = gates;
        return rep;
    };

    // Canonical walk over the eligible members.
    let mut order: Vec<usize> = (0..n).filter(|&i| gates[i].eligible).collect();
    let val_mean: Vec<f64> = (0..n)
        .map(|i| member_val[i].as_ref().map_or(f64::NAN, |l| mean_loss(l, &val.w)))
        .collect();
    order.sort_by(|&a, &b| {
        members[a]
            .cost
            .total_cmp(&members[b].cost)
            .then(val_mean[a].total_cmp(&val_mean[b]))
            .then(members[a].hash.cmp(&members[b].hash))
            .then(members[a].expression.cmp(members[b].expression))
            .then(members[a].rank.cmp(&members[b].rank))
    });
    let eligible = order.len();
    let reference = refs.names[refs.gate];
    let mut last: &Vec<f64> = gate_loss;
    let mut last_label = reference.to_string();
    let mut supported: Vec<usize> = Vec::new();
    for &i in &order {
        let loss = member_val[i].as_ref().expect("eligible members have a validation loss");
        let m = paired(last, loss, val);
        if classify(m.mean, m.se, cfg.z) == Cmp::Better {
            gates[i].supported = true;
            supported.push(i);
            last = loss;
            last_label = format!("rank {}", members[i].rank);
        } else {
            gates[i].not_better_than = Some((last_label.clone(), m.mean, m.se));
        }
    }

    let z_null = if eligible > 0 {
        normal_quantile(1.0 - cfg.alpha / eligible as f64)
    } else {
        f64::NAN
    };
    let top = supported.last().copied();
    let headline = top.or_else(|| {
        order
            .iter()
            .copied()
            .min_by(|&a, &b| val_mean[a].total_cmp(&val_mean[b]).then(members[a].rank.cmp(&members[b].rank)))
    });
    let noise = match headline {
        Some(i) => paired(gate_loss, member_val[i].as_ref().unwrap(), val),
        None => empty_report(Verdict::NoSignal, refs, cfg, 0, val).noise,
    };
    let (verdict, selected) = match top {
        None => (Verdict::NoSignal, None),
        Some(i) => {
            let m = paired(gate_loss, member_val[i].as_ref().unwrap(), val);
            if m.mean > z_null * m.se {
                (Verdict::Signal, Some(members[i].rank))
            } else {
                (Verdict::NoSignal, None)
            }
        }
    };
    let mut supported_ranks: Vec<usize> = supported.iter().map(|&i| members[i].rank).collect();
    supported_ranks.sort_unstable();
    GateReport {
        verdict,
        selected_rank: selected,
        supported_ranks,
        reference,
        z: cfg.z,
        z_null,
        alpha: cfg.alpha,
        eligible,
        noise,
        lag_source: val.lag_source,
        members: gates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: usize = 100;

    fn p1(t: usize) -> f64 {
        if t % 2 == 0 { 1.0 } else { -1.0 }
    }
    fn p2(t: usize) -> f64 {
        if (t / 2) % 2 == 0 { 1.0 } else { -1.0 }
    }
    fn r(t: usize) -> f64 {
        1.0 + 0.001 * t as f64
    }
    fn series(f: impl Fn(usize) -> f64) -> Vec<f64> {
        (0..T).map(f).collect()
    }

    // The fixture series are per-row losses. With `y = -10` and the absolute
    // metric a prediction `p` has loss `p + 10`, so differences of losses are
    // differences of the series.
    fn view() -> RowsView {
        let keys: Vec<f64> = (0..T).map(|t| t as f64).collect();
        RowsView::new(vec![-10.0; T], Vec::new(), &keys, 0, LagSource::Flag)
    }

    fn cfg() -> GateConfig {
        GateConfig { metric: Metric::Absolute, z: 1.0, alpha: 0.05 }
    }

    fn refs() -> References {
        References {
            names: vec!["mean"],
            val: vec![series(r)],
            test: vec![series(r)],
            gate: 0,
        }
    }

    struct Fixture {
        preds: Vec<Vec<f64>>,
        costs: Vec<f64>,
        identified: Vec<bool>,
        presentable: Vec<bool>,
    }

    fn members<'a>(f: &'a Fixture, test: &'a [f64], names: &'a [String]) -> Vec<MemberInput<'a>> {
        (0..f.preds.len())
            .map(|i| MemberInput {
                rank: i + 1,
                cost: f.costs[i],
                hash: i as u64,
                expression: &names[i],
                identified: f.identified[i],
                presentable: f.presentable[i],
                val_pred: &f.preds[i],
                test_pred: test,
            })
            .collect()
    }

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("m{i}")).collect()
    }

    fn walk_fixture() -> Fixture {
        let a = series(|t| r(t) - 0.20 - p1(t));
        let b: Vec<f64> = (0..T).map(|t| a[t] - 0.05 - p2(t)).collect();
        let c: Vec<f64> = (0..T).map(|t| a[t] - 0.15 - p2(t)).collect();
        Fixture { preds: vec![a, b, c], costs: vec![3.0, 5.0, 8.0], identified: vec![true; 3], presentable: vec![true; 3] }
    }

    fn run(f: &Fixture) -> GateReport {
        let test = series(|t| r(t) + t as f64);
        let nm = names(f.preds.len());
        evaluate(&members(f, &test, &nm), &refs(), &view(), &view(), &cfg())
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn walk_keeps_a_and_c_and_selects_c() {
        let f = walk_fixture();
        let rep = run(&f);
        let a = &rep.members[0];
        assert!(a.supported);
        let za = a.validation[0].delta / a.validation[0].se;
        assert!(close(za, 2.0, 1e-9), "{za}");
        assert!(!rep.members[1].supported);
        let nb = rep.members[1].not_better_than.clone().unwrap();
        assert_eq!(nb.0, "rank 1");
        assert!(close(nb.1, 0.05, 1e-12) && close(nb.2, 0.1, 1e-12), "{nb:?}");
        assert!(rep.members[2].supported);
        assert_eq!(rep.supported_ranks, vec![1, 3]);
        assert_eq!(rep.selected_rank, Some(3));
        assert!(close(rep.z_null, 2.1280452342, 1e-9), "{}", rep.z_null);
        let c = &rep.members[2].validation[0];
        assert!(close(c.delta / c.se, 2.4749, 1e-4), "{}", c.delta / c.se);
        assert_eq!(rep.verdict, Verdict::Signal);
        assert_eq!(rep.eligible, 3);
    }

    #[test]
    fn an_equivalent_cheap_member_does_not_block_a_better_one() {
        let a = series(|t| r(t) - 0.09 - 2.0 * p1(t));
        let b = series(|t| r(t) - 0.25 - p2(t));
        let f = Fixture { preds: vec![a, b], costs: vec![3.0, 5.0], identified: vec![true; 2], presentable: vec![true; 2] };
        let rep = run(&f);
        let za = rep.members[0].validation[0].delta / rep.members[0].validation[0].se;
        let zb = rep.members[1].validation[0].delta / rep.members[1].validation[0].se;
        assert!(close(za, 0.45, 1e-9) && close(zb, 2.5, 1e-9), "{za} {zb}");
        assert!(!rep.members[0].supported);
        assert!(rep.members[1].supported);
        assert_eq!(rep.selected_rank, Some(2));
        assert_eq!(rep.verdict, Verdict::Signal);
    }

    #[test]
    fn supported_but_not_beyond_bonferroni_is_no_signal() {
        let a = series(|t| r(t) - 0.15 - p1(t));
        let f = Fixture { preds: vec![a], costs: vec![3.0], identified: vec![true], presentable: vec![true] };
        let rep = run(&f);
        assert!(rep.members[0].supported);
        assert!(close(rep.z_null, 1.6448536270, 1e-9));
        assert_eq!(rep.verdict, Verdict::NoSignal);
        assert_eq!(rep.selected_rank, None);
        assert_eq!(rep.supported_ranks, vec![1]);
    }

    #[test]
    fn unidentified_members_are_never_supported_or_selected() {
        let mut f = walk_fixture();
        f.identified[2] = false;
        let rep = run(&f);
        assert!(!rep.members[2].supported && !rep.members[2].eligible);
        assert_eq!(rep.selected_rank, Some(1));
        assert!(!rep.supported_ranks.contains(&3));
        // The unidentified member is still compared.
        assert!(!rep.members[2].validation.is_empty());
        assert_eq!(rep.eligible, 2);
    }

    #[test]
    fn non_presentable_members_are_never_supported_or_selected() {
        let mut f = walk_fixture();
        f.presentable[2] = false;
        let rep = run(&f);
        assert!(!rep.members[2].supported && !rep.members[2].eligible);
        assert_eq!(rep.selected_rank, Some(1));
        assert!(!rep.supported_ranks.contains(&3));
        assert!(!rep.members[2].validation.is_empty());
        assert_eq!(rep.eligible, 2);
    }

    #[test]
    fn gate_output_is_independent_of_member_order() {
        let f = walk_fixture();
        let test = series(|t| r(t) + t as f64);
        let nm = names(3);
        let base = members(&f, &test, &nm);
        let a = evaluate(&base, &refs(), &view(), &view(), &cfg());
        for perm in [[2usize, 1, 0], [1, 2, 0], [0, 2, 1]] {
            let shuffled: Vec<MemberInput> = perm
                .iter()
                .map(|&i| MemberInput { ..clone_member(&base[i]) })
                .collect();
            let b = evaluate(&shuffled, &refs(), &view(), &view(), &cfg());
            assert_eq!(a.verdict, b.verdict);
            assert_eq!(a.selected_rank, b.selected_rank);
            assert_eq!(a.supported_ranks, b.supported_ranks);
            assert_eq!(a.z_null.to_bits(), b.z_null.to_bits());
            assert_eq!(a.noise.se.to_bits(), b.noise.se.to_bits());
            for (k, &i) in perm.iter().enumerate() {
                assert_eq!(a.members[i].supported, b.members[k].supported);
                assert_eq!(
                    a.members[i].validation[0].delta.to_bits(),
                    b.members[k].validation[0].delta.to_bits()
                );
                assert_eq!(
                    a.members[i].not_better_than.as_ref().map(|n| (n.0.clone(), n.1.to_bits(), n.2.to_bits())),
                    b.members[k].not_better_than.as_ref().map(|n| (n.0.clone(), n.1.to_bits(), n.2.to_bits()))
                );
            }
        }
    }

    fn clone_member<'a>(m: &MemberInput<'a>) -> MemberInput<'a> {
        MemberInput {
            rank: m.rank,
            cost: m.cost,
            hash: m.hash,
            expression: m.expression,
            identified: m.identified,
            presentable: m.presentable,
            val_pred: m.val_pred,
            test_pred: m.test_pred,
        }
    }

    #[test]
    fn test_predictions_never_change_the_gate() {
        let f = walk_fixture();
        let nm = names(3);
        let t1 = series(|t| r(t) + t as f64);
        let t2 = series(|t| -(t as f64) * 7.0 + p1(t));
        let a = evaluate(&members(&f, &t1, &nm), &refs(), &view(), &view(), &cfg());
        let b = evaluate(&members(&f, &t2, &nm), &refs(), &view(), &view(), &cfg());
        assert_eq!(a.verdict, b.verdict);
        assert_eq!(a.selected_rank, b.selected_rank);
        assert_eq!(a.supported_ranks, b.supported_ranks);
        for (x, y) in a.members.iter().zip(&b.members) {
            assert_eq!(x.supported, y.supported);
            assert_eq!(
                x.not_better_than.as_ref().map(|n| (n.0.clone(), n.1.to_bits(), n.2.to_bits())),
                y.not_better_than.as_ref().map(|n| (n.0.clone(), n.1.to_bits(), n.2.to_bits()))
            );
            assert_eq!(x.validation[0].delta.to_bits(), y.validation[0].delta.to_bits());
        }
        assert_ne!(a.members[0].test[0].delta.to_bits(), b.members[0].test[0].delta.to_bits());
    }

    #[test]
    fn metrics_without_a_row_loss_are_not_gated_and_no_slice_is_unvalidated() {
        let f = walk_fixture();
        let nm = names(3);
        let test = series(|t| t as f64);
        let mut c = cfg();
        c.metric = Metric::Correlation;
        let rep = evaluate(&members(&f, &test, &nm), &refs(), &view(), &view(), &c);
        assert_eq!(rep.verdict, Verdict::NotGated);
        let empty = RowsView::iid(Vec::new(), Vec::new());
        let rep = evaluate(&members(&f, &test, &nm), &refs(), &empty, &view(), &cfg());
        assert_eq!(rep.verdict, Verdict::Unvalidated);
        assert_eq!(rep.selected_rank, None);
    }

    #[test]
    fn classify_uses_a_zero_threshold_when_the_se_is_zero() {
        assert_eq!(classify(0.1, 0.0, 1.0), Cmp::Better);
        assert_eq!(classify(0.0, 0.0, 1.0), Cmp::Equivalent);
        assert_eq!(classify(-0.1, 0.0, 1.0), Cmp::Worse);
        assert_eq!(classify(1.0, 1.0, 1.0), Cmp::Equivalent);
    }
}
