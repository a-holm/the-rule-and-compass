//! #1091: the fit route. Candidates go in; their fitted constants come out with
//! the provenance of each: a device fit with its status label, an exact CPU
//! refit with its fallback cause, or a fallback held past `--fit-fallback-max`.
//!
//! A [`FitRoute`] collects one synchronous fit stage or one generation of the
//! overlap pipeline. Both schedules call the same verbs in the same order:
//! `begin` is [`FitRoute::fold`], [`FitRoute::admit`] or [`FitRoute::reject`]
//! and [`FitRoute::submit_early`]; `read` is [`FitRoute::fit_on_device`] and
//! [`FitRoute::submit_early`]; `join` is [`FitRoute::finish`]. Counters come
//! back as data ([`RouteCounts`]).
//!
//! Adapters: [`DeviceFit`] for the device evaluator and for recorded fit calls
//! (`ReplayDevice`, tests); [`ExactFits`] for the fallback pool (`PoolFits` in
//! `search.rs`) and for scoped threads (`ScopedFits`, tests).

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use super::{
    fallback_batch_split, gate_proposals, gpu_status_label, maximum_status_fallback_cause, plan_fit_batch,
    order_stat_status_fallback_cause, plan_fallback_cause, status_fallback_cause, tail_status_cause, Config,
    FallbackFitJob, FitBatch, FitFallbackCause, Ind, PipelineJob, TailJoin,
};
use crate::data::Sample;
use crate::expr::Node;
use crate::fit::{DeviceLoss, FitPolish, GpuFitPlan, OrderStat};
use crate::metrics::Metric;
use std::collections::BTreeMap;

/// The fit settings of a run, built once from the configuration. The exact
/// refit jobs take every value from here, so a new fit setting reaches the
/// serial arm, the scoped threads and the pool workers by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FitSettings {
    pub(super) iters: usize,
    pub(super) metric: Metric,
    pub(super) polish: FitPolish,
    /// #1024: run the threshold step after the LM.
    pub(super) thresholds: bool,
    /// `--fit-fallback-max`; `0` means unbounded.
    pub(super) fallback_max: usize,
}

impl FitSettings {
    pub(super) fn of(cfg: &Config) -> Self {
        FitSettings {
            iters: cfg.fit_iters,
            metric: cfg.metric,
            polish: cfg.fit_polish,
            thresholds: cfg.fit_thresholds,
            fallback_max: cfg.fit_fallback_max,
        }
    }
}

/// The device arm that fits a metric. Slope is `Lm(DeviceLoss::Slope)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DeviceArm {
    /// #169/#229: the batched LM kernel (#236: slope chains its warm start).
    Lm(DeviceLoss),
    /// #230: the squared start, then the exact rank polish.
    Rank { signed: bool },
    /// #226/#232: the squared start, then the exact order-statistic polish.
    OrderStat(OrderStat),
    /// #227: the squared start, then the exact maximum-error polish.
    Maximum,
}

impl DeviceArm {
    /// The arm the device fits `metric` with on `rows` rows, or `None` when no
    /// arm exists (`metric_not_native`). #227: the maximum arm keeps its
    /// measured crossover, so below it the metric has no device arm.
    pub(super) fn for_metric(metric: Metric, rows: usize, iters: usize) -> Option<Self> {
        let fit_loss = crate::fit::fit_loss(metric);
        let maximum = crate::fit::maximum_fit_only(metric) && crate::fit::maximum_gpu_fit_worthwhile(rows, 1, iters);
        if maximum {
            return Some(DeviceArm::Maximum);
        }
        if let Some(want) = OrderStat::for_metric(metric) {
            return Some(DeviceArm::OrderStat(want));
        }
        match crate::fit::rank_metric_signed(metric) {
            Some(signed) => Some(DeviceArm::Rank { signed }),
            None => fit_loss.map(DeviceArm::Lm),
        }
    }

    /// #936/#959: the cause of a routed tail status (4, 5, 7, 9). Only the LM
    /// arm routes its tail; the polish arms label their own statuses.
    pub(super) fn tail_cause(self, status: i32) -> Option<FitFallbackCause> {
        match self {
            DeviceArm::Lm(_) => tail_status_cause(status),
            _ => None,
        }
    }

    /// #169/#226/#227: the cause of a non-usable, non-tail status.
    fn status_cause(self, status: i32) -> FitFallbackCause {
        match self {
            DeviceArm::OrderStat(want) => order_stat_status_fallback_cause(status, want),
            DeviceArm::Maximum => maximum_status_fallback_cause(status),
            DeviceArm::Lm(_) | DeviceArm::Rank { .. } => status_fallback_cause(status),
        }
    }
}

/// #234: a single-class fit sample has no finite log-loss optimum. The counts
/// come from the same midpoint rule the CPU arm uses, so the device never fits
/// a sign the sample does not support. #289: the predicate is the fit
/// objective; the hinge codes read the same sign and are declined the same way.
pub(super) fn single_class(metric: Metric, rows: &Sample) -> bool {
    crate::fit::fit_loss(metric).is_some_and(|loss| loss.needs_label_sign()) && {
        let (pos, neg) = crate::fit::class_counts_weighted(&rows.y, &rows.w);
        pos == 0 || neg == 0
    }
}

/// One candidate's device answer: the kernel status and its constants.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DeviceAnswer {
    pub(super) status: i32,
    pub(super) c: Vec<f64>,
}

/// One device call. `answers` is `Some` exactly when the arm returned one
/// answer per plan; the rank arm's declined polish returns every answer with
/// status 3 and a negative time, which is a labelled fallback, not a failed
/// launch. The polish time and the span start are read only when `ms > 0`.
#[derive(Debug)]
pub(super) struct DeviceRun {
    pub(super) answers: Option<Vec<DeviceAnswer>>,
    pub(super) ms: f32,
    pub(super) polish_ms: f64,
    pub(super) span_start_s: Option<f64>,
}

#[cfg(test)]
impl DeviceRun {
    fn failed() -> Self {
        DeviceRun { answers: None, ms: -1.0, polish_ms: 0.0, span_start_s: None }
    }
}

/// The device side of the route.
pub(super) trait DeviceFit {
    /// Fit `plans` with `arm` in one device call. `rows` are the rows the
    /// device holds (the rank arm reads their targets).
    fn fit(&mut self, arm: DeviceArm, plans: &[&GpuFitPlan], settings: &FitSettings, rows: &Sample) -> DeviceRun;
}

#[cfg(feature = "cuda")]
impl DeviceFit for crate::gpu::GpuEvaluator {
    fn fit(&mut self, arm: DeviceArm, plans: &[&GpuFitPlan], settings: &FitSettings, rows: &Sample) -> DeviceRun {
        self.set_fit_polish(settings.polish);
        let iters = settings.iters;
        let (out, ms) = match arm {
            DeviceArm::Maximum => self.max_fit_batch(plans, iters),
            DeviceArm::OrderStat(want) => self.order_stat_fit_batch(plans, iters, want),
            DeviceArm::Rank { signed } => self.rank_fit_batch(plans, iters, &rows.y, signed),
            DeviceArm::Lm(DeviceLoss::Slope) => self.slope_fit_batch(plans, iters),
            DeviceArm::Lm(loss) => self.fit_batch(plans, iters, loss),
        };
        let (polish_ms, span_start_s) = if ms > 0.0 {
            let polish = self.take_polish_ms(ms);
            (polish, self.take_span_start_s(crate::gpu::SPAN_FIT))
        } else {
            (0.0, None)
        };
        let answers = (out.len() == plans.len())
            .then(|| out.into_iter().map(|o| DeviceAnswer { status: o.status, c: o.c }).collect());
        DeviceRun { answers, ms, polish_ms, span_start_s }
    }
}

/// #704: recorded fit calls as the device, one call per device step, in order.
/// Pure `std`: it decodes the outputs a recording holds and needs no device. A
/// recording holds one call per kernel launch, so a run replays one to one
/// only on the single-launch LM arms; the chained arms (slope, rank, order
/// statistic, maximum) record their squared start.
#[cfg(test)]
pub(super) struct ReplayDevice {
    calls: std::collections::VecDeque<crate::fit_record::RecCall>,
}

#[cfg(test)]
impl ReplayDevice {
    pub(super) fn new(calls: impl IntoIterator<Item = crate::fit_record::RecCall>) -> Self {
        ReplayDevice { calls: calls.into_iter().collect() }
    }

    /// The calls of a recording read with [`crate::fit_record::load_recording`].
    pub(super) fn from_recording(recording: crate::fit_record::Recording) -> Self {
        Self::new(recording.calls.into_iter().map(|(_, call)| call))
    }

    /// Whether `call` launched exactly `plans` with `iters` on `arm`.
    fn launched(call: &crate::fit_record::RecCall, arm: DeviceArm, plans: &[&GpuFitPlan], iters: usize) -> bool {
        let kmax = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
        if call.nprog as usize != plans.len() || call.kmax as usize != kmax || call.iters as usize != iters {
            return false;
        }
        if let DeviceArm::Lm(loss) = arm {
            if loss != DeviceLoss::Slope && call.loss != loss.code() {
                return false;
            }
        }
        if call.cbase.is_some()
            || call.status.len() != plans.len()
            || call.c.len() != plans.len() * kmax
            || call.c0.len() != plans.len() * kmax
            || call.off.len() != plans.len() + 1
        {
            return false;
        }
        let mut words = 0usize;
        for (k, plan) in plans.iter().enumerate() {
            let (a, b) = (call.off[k] as usize, call.off[k + 1] as usize);
            if a != words
                || b < a
                || call.words.get(a..b) != Some(&plan.words[..])
                || call.pcount[k] as usize != plan.nconst
                || call.c0[k * kmax..k * kmax + plan.nconst].iter().map(|v| v.to_bits()).ne(plan.c0.iter().map(|v| v.to_bits()))
            {
                return false;
            }
            words = b;
        }
        words == call.words.len()
    }
}

#[cfg(test)]
impl DeviceFit for ReplayDevice {
    fn fit(&mut self, arm: DeviceArm, plans: &[&GpuFitPlan], settings: &FitSettings, _rows: &Sample) -> DeviceRun {
        let Some(call) = self.calls.pop_front() else {
            return DeviceRun::failed();
        };
        if !Self::launched(&call, arm, plans, settings.iters) || call.ms < 0.0 {
            return DeviceRun::failed();
        }
        let kmax = call.kmax as usize;
        let answers = plans
            .iter()
            .enumerate()
            .map(|(k, plan)| DeviceAnswer { status: call.status[k], c: call.c[k * kmax..k * kmax + plan.nconst].to_vec() })
            .collect();
        DeviceRun { answers: Some(answers), ms: call.ms, polish_ms: 0.0, span_start_s: None }
    }
}

/// The exact CPU side of the route. Every job is a pure function of its tree,
/// device tree, cause and settings and of the adapter's rows, so the result
/// does not depend on the adapter.
pub(super) trait ExactFits {
    /// Start `jobs` beside other work. An empty batch is not submitted.
    fn submit(&mut self, jobs: Vec<FallbackFitJob>);
    /// Fit `jobs` now on the calling thread's workers; returns the workers used.
    fn run(&mut self, jobs: &mut [FallbackFitJob]) -> usize;
    /// Collect every submitted job, in submission order.
    fn join(&mut self) -> TailJoin;
}

/// #255: the exact CPU fits on scoped threads. A submitted batch runs at the
/// join.
#[cfg(test)]
pub(super) struct ScopedFits<'a> {
    rows: &'a Sample,
    iters: usize,
    metric: Metric,
    workers: usize,
    queued: Vec<FallbackFitJob>,
}

#[cfg(test)]
impl<'a> ScopedFits<'a> {
    pub(super) fn new(rows: &'a Sample, settings: &FitSettings, workers: usize) -> Self {
        ScopedFits { rows, iters: settings.iters, metric: settings.metric, workers, queued: Vec::new() }
    }
}

#[cfg(test)]
impl ExactFits for ScopedFits<'_> {
    fn submit(&mut self, jobs: Vec<FallbackFitJob>) {
        self.queued.extend(jobs);
    }

    fn run(&mut self, jobs: &mut [FallbackFitJob]) -> usize {
        super::fit_fallback_jobs(jobs, &self.rows.x, &self.rows.y, &self.rows.w, self.iters, self.metric, self.workers)
    }

    fn join(&mut self) -> TailJoin {
        let mut jobs = std::mem::take(&mut self.queued);
        let workers = self.run(&mut jobs);
        TailJoin { jobs, overlap_s: 0.0, wait_s: 0.0, workers }
    }
}

/// Where a schedule keeps the trees the route fits, addressed by `K`.
pub(super) trait FitSlots<K: Copy> {
    fn tree(&mut self, key: K) -> &mut Node;
    /// The island and slot of `key`.
    fn place(&self, key: K) -> (usize, usize);
    /// A value the schedule gets back with the settled tree.
    fn tag(&self, _key: K) -> u64 {
        0
    }
}

/// The synchronous stage: the islands, keyed by `(island, slot)`.
impl FitSlots<(usize, usize)> for [Vec<Ind>] {
    fn tree(&mut self, (i, j): (usize, usize)) -> &mut Node {
        &mut self[i][j].tree
    }

    fn place(&self, key: (usize, usize)) -> (usize, usize) {
        key
    }
}

/// The overlap pipeline: the queue's jobs keyed by position, tagged with the
/// structural hash each was offered with.
pub(super) struct QueueSlots<'a> {
    pub(super) jobs: &'a mut [PipelineJob],
    pub(super) offered: &'a [u64],
}

impl FitSlots<usize> for QueueSlots<'_> {
    fn tree(&mut self, k: usize) -> &mut Node {
        &mut self.jobs[k].2
    }

    fn place(&self, k: usize) -> (usize, usize) {
        (self.jobs[k].0, self.jobs[k].1)
    }

    fn tag(&self, k: usize) -> u64 {
        self.offered[k]
    }
}

/// The insertion groups. The synchronous stage inserts plan rejects, then the
/// row-floor refusals, then the dispatcher's refusal of the rest of the batch,
/// then device outcomes; each group is in admission order, which is sorted by
/// `(island, slot)`. So its insertion order is the order of the key
/// `(group, island, slot)`. A refused batch makes no device call, so the
/// dispatch and device groups never meet in one stage.
pub(super) const PLAN: u8 = 0;
pub(super) const ROW_FLOOR: u8 = 1;
pub(super) const DISPATCH: u8 = 1;
pub(super) const DEVICE: u8 = 2;

/// One candidate that needs the exact CPU fit.
#[derive(Clone, Debug)]
struct Fallback {
    /// `(group, island, slot)`.
    key: (u8, usize, usize),
    cause: FitFallbackCause,
    /// The tree at fallback time: the device constants when the device produced
    /// finite ones for a non-tail status, else the starting constants.
    tree: Node,
    /// #968: the tree at the device constants of a routed tail status.
    device: Option<Node>,
    /// #958: index of this fallback's early job among the submitted jobs.
    early: Option<usize>,
    tag: u64,
}

/// One settled candidate of [`FitRoute::finish`], in key order.
#[derive(Clone, Debug)]
pub(super) struct Settled {
    pub(super) island: usize,
    pub(super) slot: usize,
    pub(super) tree: Node,
    pub(super) tag: u64,
}

/// What [`FitRoute::finish`] returns: the settled trees, the counters since
/// the last [`FitRoute::take_counts`], and the seconds and workers to book.
#[derive(Debug)]
pub(super) struct RouteDone {
    pub(super) settled: Vec<Settled>,
    pub(super) counts: RouteCounts,
    pub(super) overlap_s: f64,
    pub(super) wait_s: f64,
    pub(super) workers: usize,
}

/// What [`FitRoute::fit_on_device`] returns.
#[derive(Debug)]
pub(super) struct DeviceStep<K> {
    pub(super) ms: f32,
    pub(super) polish_ms: f64,
    pub(super) span_start_s: Option<f64>,
    /// The device returned one answer per plan.
    pub(super) ok: bool,
    /// Keys whose device fit is kept, in job order.
    pub(super) fitted: Vec<K>,
    /// #498: seconds of the exact gate, for a gated metric.
    pub(super) gate_seconds: Option<f64>,
}

/// The fit counters of a route, by cause and label. All are sums of events.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct RouteCounts {
    /// #169: CPU fallbacks.
    pub(super) fallback: BTreeMap<FitFallbackCause, usize>,
    /// #265: refits that lowered the exact error.
    pub(super) improved: BTreeMap<FitFallbackCause, usize>,
    /// #988: refits skipped because they cannot win.
    pub(super) skipped: BTreeMap<FitFallbackCause, usize>,
    /// Device fits kept.
    pub(super) device_fits: usize,
    /// #286: kept device fits by status label.
    pub(super) labels: BTreeMap<&'static str, usize>,
    /// #255: fallbacks past `--fit-fallback-max`.
    pub(super) bounded: usize,
    /// #982: trees folded and constants removed.
    pub(super) folded: usize,
    pub(super) fold_removed: usize,
}

/// The outcome of one device answer.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Triage {
    /// A usable, finite answer: the constants are applied.
    Fitted { label: Option<&'static str> },
    /// The exact CPU fit decides. The tree holds the finite device constants
    /// unless the status is a routed tail status; `device` is the tree at the
    /// device constants of a routed tail status (#968).
    Fallback { cause: FitFallbackCause, device: Option<Node> },
}

/// #169/#578/#936: one device answer for `tree`. Finite constants are applied
/// unless the status is a routed tail status, which keeps the start.
pub(super) fn triage(arm: DeviceArm, plan: &GpuFitPlan, tree: &mut Node, answer: &DeviceAnswer) -> Triage {
    let finite = answer.c.iter().all(|v| v.is_finite());
    let tail = arm.tail_cause(answer.status);
    if finite && tail.is_none() {
        plan.apply(tree, &answer.c);
    }
    if finite && tail.is_none() && crate::fit::fit_status_usable(answer.status) {
        return Triage::Fitted { label: gpu_status_label(answer.status) };
    }
    match tail {
        Some(cause) => {
            let device = finite.then(|| {
                let mut device = tree.clone();
                plan.apply(&mut device, &answer.c);
                device
            });
            Triage::Fallback { cause, device }
        }
        None if !finite => Triage::Fallback { cause: FitFallbackCause::NonFinite, device: None },
        None => Triage::Fallback { cause: arm.status_cause(answer.status), device: None },
    }
}

/// #1071: the tree a fallback past `--fit-fallback-max` keeps: its own tree.
fn hold_past_cap(fallback: Fallback) -> Node {
    fallback.tree
}

/// The one place a refit job is built.
fn refit_job(f: &Fallback, device: Option<Node>, settings: &FitSettings) -> FallbackFitJob {
    let (_, island, slot) = f.key;
    FallbackFitJob::new(island, slot, f.tree.clone())
        .with_cause(f.cause)
        .with_device(device)
        .with_thresholds(settings.thresholds)
}

/// The route of one fit stage or one pipeline generation.
pub(super) struct FitRoute {
    settings: FitSettings,
    /// #958: the exact fits start beside other work.
    overlap: bool,
    fallbacks: Vec<Fallback>,
    /// `fallbacks[..offered]` were seen by `submit_early`.
    offered: usize,
    /// Early jobs submitted so far.
    early_jobs: usize,
    counts: RouteCounts,
}

impl FitRoute {
    pub(super) fn new(settings: FitSettings, overlap: bool) -> Self {
        FitRoute { settings, overlap, fallbacks: Vec::new(), offered: 0, early_jobs: 0, counts: RouteCounts::default() }
    }

    pub(super) fn settings(&self) -> &FitSettings {
        &self.settings
    }

    pub(super) fn is_empty(&self) -> bool {
        self.fallbacks.is_empty()
    }

    /// The counters since the last call.
    pub(super) fn take_counts(&mut self) -> RouteCounts {
        std::mem::take(&mut self.counts)
    }

    /// #982: fold the redundant constants of a candidate before its fit, so
    /// the device fit, a CPU fallback and the archive all see the fitted tree.
    pub(super) fn fold(&mut self, tree: &mut Node) {
        let (folded, removed) = crate::fold::fold_constants(tree);
        if removed > 0 {
            *tree = folded;
            self.counts.fold_removed += removed;
            self.counts.folded += 1;
        }
    }

    /// Record a fallback of `cause` in `group` with the tree it has now.
    pub(super) fn reject(&mut self, island: usize, slot: usize, group: u8, cause: FitFallbackCause, tree: &Node, tag: u64) {
        self.push(Fallback { key: (group, island, slot), cause, tree: tree.clone(), device: None, early: None, tag });
    }

    fn push(&mut self, fallback: Fallback) {
        *self.counts.fallback.entry(fallback.cause).or_default() += 1;
        self.fallbacks.push(fallback);
    }

    /// #169/#234: the admission of one candidate of the synchronous stage. A
    /// candidate the device cannot fit is a `PLAN` fallback; otherwise its plan.
    pub(super) fn admit(
        &mut self,
        island: usize,
        slot: usize,
        tree: &Node,
        arm: Option<DeviceArm>,
        single_class: bool,
    ) -> Option<GpuFitPlan> {
        let cause = if single_class {
            FitFallbackCause::SingleClass
        } else if matches!(self.settings.metric, Metric::Correlation | Metric::CorrelationSigned)
            && crate::fit::is_affine_correlation_tree(tree)
        {
            FitFallbackCause::AffineClosedForm
        } else {
            match crate::fit::gpu_fit_plan_reason(tree) {
                _ if arm.is_none() => FitFallbackCause::MetricNotNative,
                Ok(plan) => return Some(plan),
                Err(reason) => plan_fallback_cause(reason),
            }
        };
        self.reject(island, slot, PLAN, cause, tree, 0);
        None
    }

    /// #453: plan the device batch of the synchronous stage once. Its refusals
    /// become fallbacks with their current trees: the row floor, then the
    /// dispatcher's refusal of the rest.
    pub(super) fn plan_batch<S: FitSlots<(usize, usize)> + ?Sized>(
        &mut self,
        dispatcher: &mut crate::dispatch::Dispatcher,
        rows: usize,
        planned: Vec<(usize, usize, GpuFitPlan)>,
        slots: &mut S,
    ) -> FitBatch {
        let loss = crate::fit::fit_loss(self.settings.metric);
        let batch = plan_fit_batch(dispatcher, rows, self.settings.iters, loss, planned);
        for (n, &(island, slot, cause)) in batch.refused.iter().enumerate() {
            let group = if n < batch.floor { ROW_FLOOR } else { DISPATCH };
            let tree = slots.tree((island, slot)).clone();
            self.push(Fallback { key: (group, island, slot), cause, tree, device: None, early: None, tag: 0 });
        }
        batch
    }

    /// One device call over `jobs`, then the triage of every answer in job
    /// order. A failed call makes every job a `launch` fallback with its
    /// current tree. #234/#393: a gated metric judges each usable proposal in
    /// exact f64 on `rows` (in parallel on `threads`); an accepted proposal is a
    /// device fit without a status label.
    pub(super) fn fit_on_device<K: Copy, S: FitSlots<K> + ?Sized, D: DeviceFit + ?Sized>(
        &mut self,
        device: &mut D,
        arm: DeviceArm,
        jobs: &[(K, GpuFitPlan)],
        slots: &mut S,
        rows: &Sample,
        threads: usize,
    ) -> DeviceStep<K> {
        let plans: Vec<&GpuFitPlan> = jobs.iter().map(|(_, p)| p).collect();
        let run = device.fit(arm, &plans, &self.settings, rows);
        let mut step = DeviceStep {
            ms: run.ms,
            polish_ms: run.polish_ms,
            span_start_s: run.span_start_s,
            ok: false,
            fitted: Vec::new(),
            gate_seconds: None,
        };
        let Some(answers) = run.answers else {
            for (key, _) in jobs {
                let (island, slot) = slots.place(*key);
                let tag = slots.tag(*key);
                let tree = slots.tree(*key).clone();
                self.reject(island, slot, DEVICE, FitFallbackCause::Launch, &tree, tag);
            }
            return step;
        };
        step.ok = true;
        let gated = crate::fit::device_fit_is_gated(self.settings.metric);
        let mut verdicts: Vec<Option<(crate::fit::Verdict, Node)>> = Vec::new();
        if gated {
            let gate_start = std::time::Instant::now();
            let gate_jobs: Vec<(usize, Node, Vec<f64>)> = jobs
                .iter()
                .zip(&answers)
                .enumerate()
                .filter(|(_, (_, a))| {
                    crate::fit::fit_status_usable(a.status)
                        && arm.tail_cause(a.status).is_none()
                        && a.c.iter().all(|v| v.is_finite())
                })
                .map(|(k, ((key, _), a))| (k, slots.tree(*key).clone(), a.c.clone()))
                .collect();
            verdicts.resize_with(jobs.len(), || None);
            for (k, judged) in gate_proposals(gate_jobs, rows, self.settings.metric, threads) {
                verdicts[k] = Some(judged);
            }
            step.gate_seconds = Some(gate_start.elapsed().as_secs_f64());
        }
        for (k, ((key, plan), answer)) in jobs.iter().zip(answers).enumerate() {
            let (island, slot) = slots.place(*key);
            let tag = slots.tag(*key);
            let gate = gated
                && crate::fit::fit_status_usable(answer.status)
                && arm.tail_cause(answer.status).is_none()
                && answer.c.iter().all(|v| v.is_finite());
            if gate {
                let (verdict, judged) = verdicts[k].take().expect("every gated job has a verdict");
                let cause = match verdict {
                    crate::fit::Verdict::Accepted(_) => {
                        *slots.tree(*key) = judged;
                        self.counts.device_fits += 1;
                        step.fitted.push(*key);
                        continue;
                    }
                    crate::fit::Verdict::NonFinite => FitFallbackCause::NonFinite,
                    crate::fit::Verdict::Rejected if self.settings.metric == Metric::LogLoss => FitFallbackCause::LogLossWorse,
                    crate::fit::Verdict::Rejected => FitFallbackCause::Worse,
                    crate::fit::Verdict::NotIdentifiable => FitFallbackCause::NotIdentifiable,
                };
                let tree = slots.tree(*key).clone();
                self.reject(island, slot, DEVICE, cause, &tree, tag);
                continue;
            }
            let tree = slots.tree(*key);
            match triage(arm, plan, tree, &answer) {
                Triage::Fitted { label } => {
                    self.counts.device_fits += 1;
                    if let Some(label) = label {
                        *self.counts.labels.entry(label).or_default() += 1;
                    }
                    step.fitted.push(*key);
                }
                Triage::Fallback { cause, device } => {
                    let tree = tree.clone();
                    self.push(Fallback { key: (DEVICE, island, slot), cause, tree, device, early: None, tag });
                }
            }
        }
        step
    }

    /// #958: with overlap, hand the fallbacks not yet offered to `exact`, in
    /// insertion order, while fewer than `--fit-fallback-max` jobs are out.
    pub(super) fn submit_early<E: ExactFits + ?Sized>(&mut self, exact: &mut E) {
        if !self.overlap {
            return;
        }
        let cap = self.settings.fallback_max;
        let room = if cap == 0 { usize::MAX } else { cap.saturating_sub(self.early_jobs) };
        let from = self.offered;
        let n = (self.fallbacks.len() - from).min(room);
        let mut jobs = Vec::with_capacity(n);
        for (k, f) in self.fallbacks[from..from + n].iter_mut().enumerate() {
            f.early = Some(self.early_jobs + k);
            jobs.push(refit_job(f, f.device.clone(), &self.settings));
        }
        self.offered = self.fallbacks.len();
        self.early_jobs += n;
        exact.submit(jobs);
    }

    /// #255/#578/#958: join the early jobs, take the `--fit-fallback-max`
    /// prefix of the fallbacks in key order, give each refit fallback its early
    /// result or fit it now, hold the rest (#1071), and tally the refits. The
    /// route is empty afterwards.
    pub(super) fn finish<E: ExactFits + ?Sized>(&mut self, exact: &mut E) -> RouteDone {
        let joined = exact.join();
        let mut fallbacks = std::mem::take(&mut self.fallbacks);
        self.offered = 0;
        self.early_jobs = 0;
        fallbacks.sort_by_key(|f| f.key);
        let (refit, bounded) = fallback_batch_split(fallbacks.len(), self.settings.fallback_max);
        self.counts.bounded += bounded;
        let mut early: Vec<Option<FallbackFitJob>> = joined.jobs.into_iter().map(Some).collect();
        let mut done: Vec<Option<FallbackFitJob>> = (0..refit).map(|_| None).collect();
        let mut rest: Vec<FallbackFitJob> = Vec::new();
        let mut rest_at: Vec<usize> = Vec::new();
        for (n, f) in fallbacks[..refit].iter_mut().enumerate() {
            match f.early.and_then(|k| early[k].take()) {
                Some(job) => done[n] = Some(job),
                None => {
                    let device = f.device.take();
                    rest.push(refit_job(f, device, &self.settings));
                    rest_at.push(n);
                }
            }
        }
        let (rest_s, rest_workers) = if rest.is_empty() {
            (0.0, 0)
        } else {
            let start = std::time::Instant::now();
            let workers = exact.run(&mut rest);
            (start.elapsed().as_secs_f64(), workers)
        };
        for (n, job) in rest_at.into_iter().zip(rest) {
            done[n] = Some(job);
        }
        for (f, job) in fallbacks.iter_mut().zip(done) {
            let job = job.expect("every refit fallback has a job");
            debug_assert_eq!((job.island, job.slot), (f.key.1, f.key.2), "a refit belongs to its fallback");
            if job.improved {
                *self.counts.improved.entry(f.cause).or_default() += 1;
            }
            if job.skipped {
                *self.counts.skipped.entry(f.cause).or_default() += 1;
            }
            f.tree = job.tree;
        }
        let settled = fallbacks
            .into_iter()
            .enumerate()
            .map(|(n, f)| {
                let (_, island, slot) = f.key;
                let tag = f.tag;
                let tree = if n < refit { f.tree } else { hold_past_cap(f) };
                Settled { island, slot, tree, tag }
            })
            .collect();
        RouteDone {
            settled,
            counts: self.take_counts(),
            overlap_s: joined.overlap_s,
            wait_s: joined.wait_s + rest_s,
            workers: joined.workers.max(rest_workers),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::fit_fallback_jobs;
    use crate::expr::{constant, input, node, Op};
    use crate::fit::GpuFitReject;
    use crate::fit_record::RecCall;
    use crate::rng::Mt;
    use std::collections::HashMap;

    const ITERS: usize = 3;

    fn rows(metric: Metric, n: usize) -> Sample {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
            let b = (0.7 * i as f64).sin();
            x.push(vec![a, b]);
            let t = 1.5 * a + 0.3 + 0.2 * b * b;
            y.push(if metric == Metric::LogLoss { f64::from(u8::from(t > 0.4)) } else { t });
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
    }

    fn c(rng: &mut Mt) -> Node {
        constant((rng.range(-2.0, 2.0) * 64.0).round() / 64.0)
    }

    /// Small trees with one to three constants; shape 2 has a foldable product
    /// of constants, shape 4 has more constants than the device fits, shape 8
    /// has an `if` threshold constant (#1024).
    fn random_tree(rng: &mut Mt) -> Node {
        match rng.usize_below(9) {
            0 => node(Op::Add, vec![node(Op::Mul, vec![c(rng), input(0)]), c(rng)]),
            1 => node(Op::Mul, vec![c(rng), node(Op::Sin, vec![node(Op::Mul, vec![c(rng), input(1)])])]),
            2 => node(Op::Add, vec![node(Op::Mul, vec![node(Op::Mul, vec![c(rng), c(rng)]), input(0)]), c(rng)]),
            3 => node(
                Op::Add,
                vec![
                    node(Op::Mul, vec![c(rng), node(Op::Mul, vec![input(0), input(1)])]),
                    node(Op::Mul, vec![c(rng), input(1)]),
                ],
            ),
            4 => (0..21).fold(c(rng), |acc, _| node(Op::Add, vec![acc, node(Op::Mul, vec![c(rng), input(0)])])),
            5 => node(Op::Add, vec![node(Op::Mul, vec![c(rng), input(0)]), node(Op::Mul, vec![c(rng), input(1)])]),
            6 => node(Op::Mul, vec![c(rng), input(0)]),
            7 => node(
                Op::Add,
                vec![node(Op::Add, vec![node(Op::Mul, vec![c(rng), input(0)]), node(Op::Mul, vec![c(rng), input(1)])]), c(rng)],
            ),
            _ => node(Op::If, vec![node(Op::Sub, vec![input(0), c(rng)]), node(Op::Mul, vec![c(rng), input(1)]), c(rng)]),
        }
    }

    /// One candidate's scripted device answer.
    #[derive(Clone, Copy)]
    struct Script {
        status: i32,
        non_finite: bool,
        seed: u32,
    }

    fn script(rng: &mut Mt) -> Script {
        let status = *rng.pick(&[0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        Script { status, non_finite: rng.bool(0.1), seed: rng.next_u32() }
    }

    /// The recorded call a device made for `plans`, encoded and decoded
    /// through the recording format.
    fn recorded(plans: &[&GpuFitPlan], scripts: &[Script], arm: DeviceArm, failed: bool) -> RecCall {
        let kmax = plans.iter().map(|p| p.nconst).max().unwrap_or(0);
        let mut words = Vec::new();
        let mut off = vec![0u32];
        let mut c0 = vec![0.0; plans.len() * kmax];
        let mut c = vec![0.0; plans.len() * kmax];
        let mut gauge = Vec::new();
        for (k, (plan, s)) in plans.iter().zip(scripts).enumerate() {
            words.extend_from_slice(&plan.words);
            off.push(words.len() as u32);
            c0[k * kmax..k * kmax + plan.nconst].copy_from_slice(&plan.c0);
            let mut r = Mt::new(s.seed);
            for i in 0..plan.nconst {
                c[k * kmax + i] = plan.c0[i] + (r.range(-1.0, 1.0) * 256.0).round() / 256.0;
            }
            if s.non_finite {
                c[k * kmax + r.usize_below(plan.nconst)] = f64::NAN;
            }
            gauge.extend_from_slice(&plan.gauge);
        }
        let loss = match arm {
            DeviceArm::Lm(loss) => loss.code(),
            _ => DeviceLoss::Squared.code(),
        };
        let call = RecCall {
            data: 0,
            flags: 0,
            iters: ITERS as u32,
            loss,
            polish: 0,
            nprog: plans.len() as u32,
            kmax: kmax as u32,
            words,
            off,
            pcount: plans.iter().map(|p| p.nconst as i32).collect(),
            c0,
            cbase: None,
            gauge,
            ms: if failed { -1.0 } else { 0.5 },
            status: scripts.iter().map(|s| s.status).collect(),
            sse: vec![0.0; plans.len()],
            c,
        };
        RecCall::decode(&call.encode()).expect("the recorded call decodes")
    }

    /// The answers the reference reads from a recorded call, `None` for a
    /// failed launch.
    fn answers_of(call: &RecCall, plans: &[&GpuFitPlan]) -> Option<Vec<DeviceAnswer>> {
        if call.ms < 0.0 {
            return None;
        }
        let kmax = call.kmax as usize;
        Some(
            plans
                .iter()
                .enumerate()
                .map(|(k, p)| DeviceAnswer { status: call.status[k], c: call.c[k * kmax..k * kmax + p.nconst].to_vec() })
                .collect(),
        )
    }

    fn add(map: &mut BTreeMap<FitFallbackCause, usize>, cause: FitFallbackCause) {
        *map.entry(cause).or_default() += 1;
    }

    fn settings(metric: Metric, cap: usize, thresholds: bool) -> FitSettings {
        FitSettings { iters: ITERS, metric, polish: FitPolish::default(), thresholds, fallback_max: cap }
    }

    /// A dispatcher that sends every batch to the device (0), refuses every
    /// batch (1), or decides by the default cost model (2).
    fn dispatcher(kind: usize) -> crate::dispatch::Dispatcher {
        match kind {
            0 => crate::dispatch::Dispatcher::new(crate::dispatch::Calibration::default())
                .with_mode(crate::dispatch::DispatchMode::Device),
            1 => crate::dispatch::Dispatcher::new(crate::dispatch::Calibration {
                launch_s: 1.0e9,
                ..crate::dispatch::Calibration::default()
            }),
            _ => crate::dispatch::Dispatcher::new(crate::dispatch::Calibration::default()),
        }
    }

    /// The reference: the synchronous route as `fit_pending` wrote it inline
    /// before the module, kept verbatim (pool fits replaced by the scoped arm,
    /// which gives the same bits). Returns the counters.
    #[allow(clippy::too_many_arguments)]
    fn reference_sync(
        grid: &mut [Vec<Node>],
        admitted: &[(usize, usize)],
        dispatcher: &mut crate::dispatch::Dispatcher,
        arm: Option<DeviceArm>,
        single_class: bool,
        call: &dyn Fn(&[&GpuFitPlan]) -> RecCall,
        sample: &Sample,
        settings: &FitSettings,
    ) -> RouteCounts {
        let metric = settings.metric;
        let want = match arm {
            Some(DeviceArm::OrderStat(w)) => Some(w),
            _ => None,
        };
        let maximum_arm = arm == Some(DeviceArm::Maximum);
        let rank_loss = match arm {
            Some(DeviceArm::Rank { signed }) => Some(signed),
            _ => None,
        };
        let mut counts = RouteCounts::default();
        let mut planned: Vec<(usize, usize, GpuFitPlan)> = Vec::new();
        let mut fallbacks: Vec<(usize, usize, FitFallbackCause)> = Vec::new();
        let mut device_alts: HashMap<(usize, usize), Node> = HashMap::new();
        for &(i, j) in admitted {
            let (folded, removed) = crate::fold::fold_constants(&grid[i][j]);
            if removed > 0 {
                grid[i][j] = folded;
                counts.fold_removed += removed;
                counts.folded += 1;
            }
            if single_class {
                fallbacks.push((i, j, FitFallbackCause::SingleClass));
                continue;
            }
            if matches!(metric, Metric::Correlation | Metric::CorrelationSigned)
                && crate::fit::is_affine_correlation_tree(&grid[i][j])
            {
                fallbacks.push((i, j, FitFallbackCause::AffineClosedForm));
                continue;
            }
            match crate::fit::gpu_fit_plan_reason(&grid[i][j]) {
                _ if arm.is_none() => fallbacks.push((i, j, FitFallbackCause::MetricNotNative)),
                Ok(plan) => planned.push((i, j, plan)),
                Err(reason) => fallbacks.push((i, j, plan_fallback_cause(reason))),
            }
        }
        let batch = plan_fit_batch(dispatcher, sample.len(), settings.iters, crate::fit::fit_loss(metric), planned);
        fallbacks.extend(batch.refused);
        let jobs = batch.jobs;
        if !jobs.is_empty() {
            let plans: Vec<&GpuFitPlan> = jobs.iter().map(|(_, _, p)| p).collect();
            let out = answers_of(&call(&plans), &plans);
            if let Some(out) = out {
                let gated = crate::fit::device_fit_is_gated(metric);
                let lm_arm = want.is_none() && !maximum_arm && rank_loss.is_none();
                let mut verdicts: Vec<Option<(crate::fit::Verdict, Node)>> = Vec::new();
                if gated {
                    let gate_jobs: Vec<(usize, Node, Vec<f64>)> = jobs
                        .iter()
                        .zip(&out)
                        .enumerate()
                        .filter(|(_, (_, res))| {
                            crate::fit::fit_status_usable(res.status)
                                && !(lm_arm && tail_status_cause(res.status).is_some())
                                && res.c.iter().all(|v| v.is_finite())
                        })
                        .map(|(k, ((i, j, _), res))| (k, grid[*i][*j].clone(), res.c.clone()))
                        .collect();
                    verdicts.resize_with(jobs.len(), || None);
                    for (k, judged) in gate_proposals(gate_jobs, sample, metric, 2) {
                        verdicts[k] = Some(judged);
                    }
                }
                for (k, ((i, j, plan), res)) in jobs.iter().zip(out).enumerate() {
                    let finite = res.c.iter().all(|v| v.is_finite());
                    let tail = if lm_arm { tail_status_cause(res.status) } else { None };
                    let usable = crate::fit::fit_status_usable(res.status) && tail.is_none();
                    if gated && usable && finite {
                        let (verdict, judged) = verdicts[k].take().expect("every gated job has a verdict");
                        if matches!(verdict, crate::fit::Verdict::Accepted(_)) {
                            grid[*i][*j] = judged;
                        }
                        let cause = match verdict {
                            crate::fit::Verdict::Accepted(_) => {
                                counts.device_fits += 1;
                                continue;
                            }
                            crate::fit::Verdict::NonFinite => FitFallbackCause::NonFinite,
                            crate::fit::Verdict::Rejected if metric == Metric::LogLoss => FitFallbackCause::LogLossWorse,
                            crate::fit::Verdict::Rejected => FitFallbackCause::Worse,
                            crate::fit::Verdict::NotIdentifiable => FitFallbackCause::NotIdentifiable,
                        };
                        fallbacks.push((*i, *j, cause));
                        continue;
                    }
                    if finite && tail.is_none() {
                        plan.apply(&mut grid[*i][*j], &res.c);
                    }
                    if usable && finite {
                        counts.device_fits += 1;
                        if let Some(label) = gpu_status_label(res.status) {
                            *counts.labels.entry(label).or_default() += 1;
                        }
                    } else {
                        let cause = if let Some(cause) = tail {
                            if finite {
                                let mut alt = grid[*i][*j].clone();
                                plan.apply(&mut alt, &res.c);
                                device_alts.insert((*i, *j), alt);
                            }
                            cause
                        } else if !finite {
                            FitFallbackCause::NonFinite
                        } else if let Some(want) = want {
                            order_stat_status_fallback_cause(res.status, want)
                        } else if maximum_arm {
                            maximum_status_fallback_cause(res.status)
                        } else {
                            status_fallback_cause(res.status)
                        };
                        fallbacks.push((*i, *j, cause));
                    }
                }
            } else {
                for (i, j, _) in &jobs {
                    fallbacks.push((*i, *j, FitFallbackCause::Launch));
                }
            }
        }
        for &(_, _, cause) in &fallbacks {
            add(&mut counts.fallback, cause);
        }
        let (refit, bounded) = fallback_batch_split(fallbacks.len(), settings.fallback_max);
        counts.bounded += bounded;
        let mut jobs: Vec<FallbackFitJob> = fallbacks[..refit]
            .iter()
            .map(|&(island, slot, cause)| {
                FallbackFitJob::new(island, slot, grid[island][slot].clone())
                    .with_cause(cause)
                    .with_thresholds(settings.thresholds)
                    .with_device(device_alts.remove(&(island, slot)))
            })
            .collect();
        fit_fallback_jobs(&mut jobs, &sample.x, &sample.y, &sample.w, settings.iters, metric, 2);
        for (&(_, _, cause), job) in fallbacks[..refit].iter().zip(&jobs) {
            if job.improved {
                add(&mut counts.improved, cause);
            }
            if job.skipped {
                add(&mut counts.skipped, cause);
            }
        }
        for job in jobs {
            grid[job.island][job.slot] = job.tree;
        }
        counts
    }

    /// One reference pipeline fallback.
    struct OldFallback {
        group: u8,
        island: usize,
        slot: usize,
        cause: FitFallbackCause,
        offered: u64,
        tree: Node,
        early: Option<usize>,
        device: Option<Node>,
    }

    /// The reference: the pipeline route as `fit_pipeline_queue` and
    /// `finish_pipeline_fallbacks` wrote it inline before the module, kept
    /// verbatim (pool fits replaced by the scoped arm). With `relabel` a plan
    /// reject takes the synchronous label (#1091 decision 9). Returns the final
    /// tree and tag of every fitted or settled slot, and the counters.
    #[allow(clippy::type_complexity)]
    fn reference_pipeline(
        flushes: &[Vec<(usize, usize, Node)>],
        call: &dyn Fn(usize, &[&GpuFitPlan]) -> RecCall,
        overlap: bool,
        relabel: bool,
        sample: &Sample,
        settings: &FitSettings,
    ) -> (BTreeMap<(usize, usize), (Node, u64)>, RouteCounts) {
        let mut counts = RouteCounts::default();
        let mut out = BTreeMap::new();
        let mut fallbacks: Vec<OldFallback> = Vec::new();
        let mut early_jobs: Vec<FallbackFitJob> = Vec::new();
        let cap = settings.fallback_max;
        for (n, flush) in flushes.iter().enumerate() {
            let mut jobs = flush.clone();
            let offered_hash: Vec<u64> = jobs.iter().map(|(_, _, t)| crate::expr::structural_hash(t)).collect();
            let mut plans: Vec<(usize, GpuFitPlan)> = Vec::new();
            let mut fallback_jobs: Vec<usize> = Vec::new();
            for (k, (_, _, tree)) in jobs.iter().enumerate() {
                match crate::fit::gpu_fit_plan_reason(tree) {
                    Ok(plan) => plans.push((k, plan)),
                    Err(_) => fallback_jobs.push(k),
                }
            }
            let fresh = fallbacks.len();
            for &k in &fallback_jobs {
                let cause = match crate::fit::gpu_fit_plan_reason(&jobs[k].2) {
                    // #1091 decision 9: the one label the route changes.
                    Err(reason) if relabel => plan_fallback_cause(reason),
                    Err(GpuFitReject::TooManyConstants) => FitFallbackCause::TooManyConstants,
                    _ => FitFallbackCause::UnsupportedPlan,
                };
                add(&mut counts.fallback, cause);
                let (i, j, tree) = &jobs[k];
                fallbacks.push(OldFallback {
                    group: 0,
                    island: *i,
                    slot: *j,
                    cause,
                    offered: offered_hash[k],
                    tree: tree.clone(),
                    early: None,
                    device: None,
                });
            }
            if !plans.is_empty() {
                let refs: Vec<&GpuFitPlan> = plans.iter().map(|(_, p)| p).collect();
                let answers = answers_of(&call(n, &refs), &refs);
                match answers {
                    None => {
                        for (k, _) in &plans {
                            add(&mut counts.fallback, FitFallbackCause::Launch);
                            let (i, j, tree) = &jobs[*k];
                            fallbacks.push(OldFallback {
                                group: 1,
                                island: *i,
                                slot: *j,
                                cause: FitFallbackCause::Launch,
                                offered: offered_hash[*k],
                                tree: tree.clone(),
                                early: None,
                                device: None,
                            });
                        }
                    }
                    Some(answers) => {
                        for ((k, plan), res) in plans.iter().zip(answers) {
                            let tree = &mut jobs[*k].2;
                            let verdict = if let Some(cause) = tail_status_cause(res.status) {
                                Some(cause)
                            } else {
                                let finite = res.c.iter().all(|v| v.is_finite());
                                if finite {
                                    plan.apply(tree, &res.c);
                                }
                                if !finite {
                                    Some(FitFallbackCause::NonFinite)
                                } else if crate::fit::fit_status_usable(res.status) {
                                    None
                                } else {
                                    Some(status_fallback_cause(res.status))
                                }
                            };
                            match verdict {
                                None => {
                                    if let Some(label) = gpu_status_label(res.status) {
                                        *counts.labels.entry(label).or_default() += 1;
                                    }
                                    counts.device_fits += 1;
                                    out.insert((jobs[*k].0, jobs[*k].1), (jobs[*k].2.clone(), offered_hash[*k]));
                                }
                                Some(cause) => {
                                    add(&mut counts.fallback, cause);
                                    let device = (tail_status_cause(res.status).is_some()
                                        && res.c.iter().all(|v| v.is_finite()))
                                    .then(|| {
                                        let mut device = jobs[*k].2.clone();
                                        plan.apply(&mut device, &res.c);
                                        device
                                    });
                                    let (i, j, tree) = &jobs[*k];
                                    fallbacks.push(OldFallback {
                                        group: 1,
                                        island: *i,
                                        slot: *j,
                                        cause,
                                        offered: offered_hash[*k],
                                        tree: tree.clone(),
                                        early: None,
                                        device,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            if overlap {
                let base = early_jobs.len();
                let room = if cap == 0 { usize::MAX } else { cap.saturating_sub(base) };
                for (k, f) in fallbacks[fresh..].iter_mut().take(room).enumerate() {
                    f.early = Some(base + k);
                    early_jobs.push(
                        FallbackFitJob::new(f.island, f.slot, f.tree.clone())
                            .with_cause(f.cause)
                            .with_device(f.device.clone())
                            .with_thresholds(settings.thresholds),
                    );
                }
            }
        }
        if fallbacks.is_empty() {
            return (out, counts);
        }
        fallbacks.sort_by_key(|f| (f.group, f.island, f.slot));
        let (refit, bounded) = fallback_batch_split(fallbacks.len(), cap);
        counts.bounded += bounded;
        fit_fallback_jobs(&mut early_jobs, &sample.x, &sample.y, &sample.w, settings.iters, settings.metric, 2);
        let mut early: Vec<Option<FallbackFitJob>> = early_jobs.into_iter().map(Some).collect();
        let mut rest: Vec<usize> = Vec::new();
        for (n, f) in fallbacks[..refit].iter_mut().enumerate() {
            match f.early.and_then(|k| early[k].take()) {
                Some(job) => {
                    if job.improved {
                        add(&mut counts.improved, f.cause);
                    }
                    if job.skipped {
                        add(&mut counts.skipped, f.cause);
                    }
                    f.tree = job.tree;
                }
                None => rest.push(n),
            }
        }
        if !rest.is_empty() {
            let mut jobs: Vec<FallbackFitJob> = rest
                .iter()
                .map(|&n| {
                    let f = &mut fallbacks[n];
                    FallbackFitJob::new(f.island, f.slot, f.tree.clone())
                        .with_cause(f.cause)
                        .with_device(f.device.take())
                        .with_thresholds(settings.thresholds)
                })
                .collect();
            fit_fallback_jobs(&mut jobs, &sample.x, &sample.y, &sample.w, settings.iters, settings.metric, 2);
            for (&n, job) in rest.iter().zip(jobs) {
                if job.improved {
                    add(&mut counts.improved, fallbacks[n].cause);
                }
                if job.skipped {
                    add(&mut counts.skipped, fallbacks[n].cause);
                }
                fallbacks[n].tree = job.tree;
            }
        }
        for f in fallbacks {
            out.insert((f.island, f.slot), (f.tree, f.offered));
        }
        (out, counts)
    }

    const SYNC_ARMS: [(Metric, DeviceArm); 8] = [
        (Metric::Squared, DeviceArm::Lm(DeviceLoss::Squared)),
        (Metric::Squared, DeviceArm::Lm(DeviceLoss::Slope)),
        (Metric::Absolute, DeviceArm::Lm(DeviceLoss::Absolute)),
        (Metric::LogLoss, DeviceArm::Lm(DeviceLoss::Logistic)),
        (Metric::Correlation, DeviceArm::Lm(DeviceLoss::Correlation)),
        (Metric::RankCorrelation, DeviceArm::Rank { signed: false }),
        (Metric::Median, DeviceArm::OrderStat(OrderStat::Median)),
        (Metric::Maximum, DeviceArm::Maximum),
    ];

    fn cap_for(rng: &mut Mt, n: usize) -> usize {
        *rng.pick(&[0, 1, 2, n / 2, n, n + 3])
    }

    fn assert_same_tree(a: &Node, b: &Node, what: &str) {
        assert!(crate::expr::same(a, b), "{what}: {a:?} against {b:?}");
    }

    /// One synchronous fit stage through the route and through the reference.
    fn sync_case(seed: u32) {
        let mut rng = Mt::new(seed);
        let (metric, arm) = *rng.pick(&SYNC_ARMS);
        // Below 64 rows only one-constant trees pass the row floor, below 96
        // only one and two; 160 rows pass every planned tree.
        let sample = rows(metric, *rng.pick(&[48usize, 72, 160]));
        let cfg = Config::default();
        let islands_n = 1 + rng.usize_below(3);
        let slots_n = 2 + rng.usize_below(5);
        let mut islands: Vec<Vec<Ind>> = (0..islands_n)
            .map(|_| (0..slots_n).map(|_| Ind::pending(random_tree(&mut rng), &cfg)).collect())
            .collect();
        let admitted: Vec<(usize, usize)> = (0..islands_n)
            .flat_map(|i| (0..slots_n).map(move |j| (i, j)))
            .filter(|_| rng.bool(0.8))
            .collect();
        let scripts: HashMap<(usize, usize), Script> = admitted.iter().map(|&a| (a, script(&mut rng))).collect();
        let single_class = rng.bool(0.05);
        let arm = (!rng.bool(0.05)).then_some(arm);
        let failed = rng.bool(0.05);
        let overlap = rng.bool(0.5);
        let kind = rng.usize_below(3);
        let thresholds = rng.bool(0.25);
        let settings = settings(metric, cap_for(&mut rng, admitted.len()), thresholds);
        let mut grid: Vec<Vec<Node>> = islands.iter().map(|pop| pop.iter().map(|ind| ind.tree.clone()).collect()).collect();
        let call = |plans: &[&GpuFitPlan], keys: &[(usize, usize)]| {
            let s: Vec<Script> = keys.iter().map(|k| scripts[k]).collect();
            recorded(plans, &s, arm.unwrap_or(DeviceArm::Maximum), failed)
        };

        // #1024: the threshold flag is a scope on the main thread for the plan
        // and a setting of every refit job.
        crate::fit::with_fit_thresholds(thresholds, || {
            let mut route = FitRoute::new(settings, overlap);
            let mut exact = ScopedFits::new(&sample, &settings, 2);
            let mut planned: Vec<(usize, usize, GpuFitPlan)> = Vec::new();
            for &(i, j) in &admitted {
                route.fold(&mut islands[i][j].tree);
                if let Some(plan) = route.admit(i, j, &islands[i][j].tree, arm, single_class) {
                    planned.push((i, j, plan));
                }
            }
            let batch = route.plan_batch(&mut dispatcher(kind), sample.len(), planned, &mut islands[..]);
            let jobs: Vec<((usize, usize), GpuFitPlan)> = batch.jobs.into_iter().map(|(i, j, p)| ((i, j), p)).collect();
            route.submit_early(&mut exact);
            let keys: Vec<(usize, usize)> = jobs.iter().map(|(k, _)| *k).collect();
            if !jobs.is_empty() {
                let plans: Vec<&GpuFitPlan> = jobs.iter().map(|(_, p)| p).collect();
                let mut device = ReplayDevice::new([call(&plans, &keys)]);
                let step = route.fit_on_device(&mut device, arm.expect("planned"), &jobs, &mut islands[..], &sample, 2);
                assert_eq!(step.ok, !failed, "seed {seed}");
            }
            route.submit_early(&mut exact);
            let done = route.finish(&mut exact);
            for s in done.settled {
                islands[s.island][s.slot].tree = s.tree;
            }

            let reference = reference_sync(
                &mut grid,
                &admitted,
                &mut dispatcher(kind),
                arm,
                single_class,
                &|plans: &[&GpuFitPlan]| call(plans, &keys),
                &sample,
                &settings,
            );
            assert_eq!(done.counts, reference, "seed {seed}: counters");
        });
        for &(i, j) in &admitted {
            assert_same_tree(&islands[i][j].tree, &grid[i][j], &format!("seed {seed}: slot ({i}, {j})"));
        }
    }

    /// One pipeline generation through the route and through the reference.
    fn pipeline_case(seed: u32) {
        let mut rng = Mt::new(seed);
        let (metric, arm) = *rng.pick(&[
            (Metric::Squared, DeviceArm::Lm(DeviceLoss::Squared)),
            (Metric::Squared, DeviceArm::Lm(DeviceLoss::Slope)),
            (Metric::Absolute, DeviceArm::Lm(DeviceLoss::Absolute)),
        ]);
        let sample = rows(metric, 48);
        let islands_n = 1 + rng.usize_below(3);
        let slots_n = 2 + rng.usize_below(6);
        let mut all: Vec<(usize, usize, Node)> = Vec::new();
        for i in 0..islands_n {
            for j in 0..slots_n {
                if rng.bool(0.8) {
                    all.push((i, j, random_tree(&mut rng)));
                }
            }
        }
        // Arrival order across islands is not admission order.
        for k in (1..all.len()).rev() {
            all.swap(k, rng.usize_below(k + 1));
        }
        let flush_n = 1 + rng.usize_below(3);
        let per = all.len().div_ceil(flush_n).max(1);
        let flushes: Vec<Vec<(usize, usize, Node)>> = all.chunks(per).map(|c| c.to_vec()).collect();
        let scripts: HashMap<(usize, usize), Script> = all.iter().map(|(i, j, _)| ((*i, *j), script(&mut rng))).collect();
        let failed: Vec<bool> = flushes.iter().map(|_| rng.bool(0.05)).collect();
        let overlap = rng.bool(0.5);
        let thresholds = rng.bool(0.25);
        let settings = settings(metric, cap_for(&mut rng, all.len()), thresholds);
        let call = |n: usize, plans: &[&GpuFitPlan], keys: &[(usize, usize)]| {
            let s: Vec<Script> = keys.iter().map(|k| scripts[k]).collect();
            recorded(plans, &s, arm, failed[n])
        };

        crate::fit::with_fit_thresholds(thresholds, || {
            let mut route = FitRoute::new(settings, overlap);
            let mut exact = ScopedFits::new(&sample, &settings, 2);
            let mut got: BTreeMap<(usize, usize), (Node, u64)> = BTreeMap::new();
            let mut keys_by_flush: Vec<Vec<(usize, usize)>> = Vec::new();
            for (n, flush) in flushes.iter().enumerate() {
                let mut jobs = flush.clone();
                let offered: Vec<u64> = jobs.iter().map(|(_, _, t)| crate::expr::structural_hash(t)).collect();
                let mut plans: Vec<(usize, GpuFitPlan)> = Vec::new();
                for (k, (i, j, tree)) in jobs.iter().enumerate() {
                    match crate::fit::gpu_fit_plan_reason(tree) {
                        Ok(plan) => plans.push((k, plan)),
                        Err(reason) => route.reject(*i, *j, PLAN, plan_fallback_cause(reason), tree, offered[k]),
                    }
                }
                let keys: Vec<(usize, usize)> = plans.iter().map(|(k, _)| (jobs[*k].0, jobs[*k].1)).collect();
                keys_by_flush.push(keys.clone());
                if !plans.is_empty() {
                    let refs: Vec<&GpuFitPlan> = plans.iter().map(|(_, p)| p).collect();
                    let mut device = ReplayDevice::new([call(n, &refs, &keys)]);
                    let mut slots = QueueSlots { jobs: &mut jobs, offered: &offered };
                    let step = route.fit_on_device(&mut device, arm, &plans, &mut slots, &sample, 1);
                    for k in step.fitted {
                        got.insert((jobs[k].0, jobs[k].1), (jobs[k].2.clone(), offered[k]));
                    }
                }
                route.submit_early(&mut exact);
            }
            let counts = if route.is_empty() {
                route.take_counts()
            } else {
                let done = route.finish(&mut exact);
                for s in done.settled {
                    got.insert((s.island, s.slot), (s.tree, s.tag));
                }
                done.counts
            };

            let (want, reference) = reference_pipeline(
                &flushes,
                &|n: usize, plans: &[&GpuFitPlan]| call(n, plans, &keys_by_flush[n]),
                overlap,
                thresholds,
                &sample,
                &settings,
            );
            assert_eq!(counts, reference, "seed {seed}: counters");
            assert_eq!(got.len(), want.len(), "seed {seed}: settled slots");
            for ((key, (a, ta)), (want_key, (b, tb))) in got.iter().zip(&want) {
                assert_eq!(key, want_key, "seed {seed}: settled slots");
                assert_same_tree(a, b, &format!("seed {seed}: slot {key:?}"));
                assert_eq!(ta, tb, "seed {seed}: slot {key:?} tag");
            }
        });
    }

    /// #255/#453: with a cap inside the batch refusals, the stage refits the
    /// row-floor refusal that `plan_fit_batch` returns first, not the
    /// dispatcher's refusal of a slot that sorts before it.
    #[test]
    fn a_cap_inside_the_batch_refusals_keeps_their_order() {
        let sample = rows(Metric::Squared, 40);
        let cfg = Config::default();
        let one = node(Op::Mul, vec![constant(0.25), input(0)]);
        let three = node(
            Op::Add,
            vec![node(Op::Add, vec![node(Op::Mul, vec![constant(0.5), input(0)]), node(Op::Mul, vec![constant(0.5), input(1)])]), constant(0.5)],
        );
        for overlap in [false, true] {
            let mut islands = vec![vec![Ind::pending(one.clone(), &cfg), Ind::pending(three.clone(), &cfg)]];
            let start: Vec<Node> = islands[0].iter().map(|ind| ind.tree.clone()).collect();
            let settings = settings(Metric::Squared, 1, false);
            let mut route = FitRoute::new(settings, overlap);
            let mut exact = ScopedFits::new(&sample, &settings, 1);
            let planned: Vec<(usize, usize, GpuFitPlan)> = (0..2)
                .map(|j| (0, j, crate::fit::gpu_fit_plan_reason(&islands[0][j].tree).expect("a plan")))
                .collect();
            let batch = route.plan_batch(&mut dispatcher(1), sample.len(), planned, &mut islands[..]);
            assert!(batch.jobs.is_empty());
            assert_eq!(batch.refused, vec![(0, 1, FitFallbackCause::TooFewRows), (0, 0, FitFallbackCause::DispatchCost)]);
            route.submit_early(&mut exact);
            let done = route.finish(&mut exact);
            for s in done.settled {
                islands[s.island][s.slot].tree = s.tree;
            }
            assert_eq!(done.counts.bounded, 1);
            assert_same_tree(&islands[0][0].tree, &start[0], "the dispatch refusal past the cap is held");
            assert!(!crate::expr::same(&islands[0][1].tree, &start[1]), "the row-floor refusal is refit");
            let mut grid = vec![start.clone()];
            let reference = reference_sync(
                &mut grid,
                &[(0, 0), (0, 1)],
                &mut dispatcher(1),
                Some(DeviceArm::Lm(DeviceLoss::Squared)),
                false,
                &|_: &[&GpuFitPlan]| unreachable!("a refused batch makes no device call"),
                &sample,
                &settings,
            );
            assert_eq!(done.counts, reference);
            for j in 0..2 {
                assert_same_tree(&islands[0][j].tree, &grid[0][j], &format!("overlap {overlap}: slot {j}"));
            }
        }
    }

    /// #1091: the route's decisions and outputs equal the inline route it
    /// replaced over random status mixes, caps, arms and both schedules, with
    /// recorded device answers and exact fits on scoped threads.
    #[test]
    fn route_matches_the_inline_route() {
        for seed in 0..1500 {
            sync_case(seed);
        }
        for seed in 0..500 {
            pipeline_case(10_000 + seed);
        }
    }

    /// #578/#936: a non-usable status with finite constants moves the tree; a
    /// routed tail status keeps the start and carries the device tree; a
    /// non-finite answer is never applied.
    #[test]
    fn triage_follows_the_status() {
        let tree = node(Op::Add, vec![node(Op::Mul, vec![constant(1.0), input(0)]), constant(0.5)]);
        let plan = crate::fit::gpu_fit_plan_reason(&tree).expect("an eligible plan");
        let lm = DeviceArm::Lm(DeviceLoss::Squared);
        let answer = |status: i32, c: Vec<f64>| DeviceAnswer { status, c };
        let mut moved = tree.clone();
        plan.apply(&mut moved, &[2.0, 3.0]);
        for (status, cause) in [
            (1, FitFallbackCause::IllConditioned),
            (2, FitFallbackCause::NonFinite),
            (3, FitFallbackCause::RankDeclined),
        ] {
            let mut t = tree.clone();
            let got = triage(lm, &plan, &mut t, &answer(status, vec![2.0, 3.0]));
            assert_eq!(got, Triage::Fallback { cause, device: None });
            assert_eq!(t, moved, "status {status}");
        }
        for (status, cause) in [
            (4, FitFallbackCause::RankDeficient),
            (5, FitFallbackCause::RecoveredIterate),
            (7, FitFallbackCause::NonFiniteStart),
            (9, FitFallbackCause::Unconverged),
        ] {
            let mut t = tree.clone();
            let got = triage(lm, &plan, &mut t, &answer(status, vec![2.0, 3.0]));
            assert_eq!(got, Triage::Fallback { cause, device: Some(moved.clone()) });
            assert_eq!(t, tree, "status {status} keeps the start");
            // A polish arm labels the same status as a device fit.
            let mut t = tree.clone();
            let polish = triage(DeviceArm::Maximum, &plan, &mut t, &answer(status, vec![2.0, 3.0]));
            assert_eq!(polish, Triage::Fitted { label: gpu_status_label(status) });
            assert_eq!(t, moved);
        }
        let mut t = tree.clone();
        assert_eq!(triage(lm, &plan, &mut t, &answer(0, vec![2.0, 3.0])), Triage::Fitted { label: None });
        assert_eq!(t, moved);
        let mut t = tree.clone();
        assert_eq!(
            triage(lm, &plan, &mut t, &answer(0, vec![f64::NAN, 3.0])),
            Triage::Fallback { cause: FitFallbackCause::NonFinite, device: None }
        );
        assert_eq!(t, tree, "non-finite constants are never applied");
        let mut t = tree.clone();
        assert_eq!(
            triage(lm, &plan, &mut t, &answer(9, vec![f64::NAN, 3.0])),
            Triage::Fallback { cause: FitFallbackCause::Unconverged, device: None }
        );
        let mut t = tree.clone();
        assert_eq!(
            triage(DeviceArm::OrderStat(OrderStat::Median), &plan, &mut t, &answer(3, vec![2.0, 3.0])),
            Triage::Fallback { cause: FitFallbackCause::MedianDeclined, device: None }
        );
        let mut t = tree.clone();
        assert_eq!(
            triage(DeviceArm::Maximum, &plan, &mut t, &answer(3, vec![2.0, 3.0])),
            Triage::Fallback { cause: FitFallbackCause::MaximumDeclined, device: None }
        );
    }

    /// #229/#230/#226/#227: the arm equals the inline choice for every metric:
    /// no arm exactly when the admission said `metric_not_native`, the same
    /// device call, and a routed tail only on the LM arm.
    #[test]
    fn device_arm_matches_the_inline_choice() {
        for &metric in Metric::all() {
            for rows in [10usize, 1_000, 100_000, 3_000_000] {
                for iters in [1usize, 8] {
                    let fit_loss = crate::fit::fit_loss(metric);
                    let rank_loss = crate::fit::rank_fit_only(metric);
                    let order_stat_arm = crate::fit::order_stat_fit_only(metric);
                    let maximum_arm =
                        crate::fit::maximum_fit_only(metric) && crate::fit::maximum_gpu_fit_worthwhile(rows, 1, iters);
                    let none = fit_loss.is_none() && rank_loss.is_none() && !order_stat_arm && !maximum_arm;
                    let arm = DeviceArm::for_metric(metric, rows, iters);
                    assert_eq!(arm.is_none(), none, "{metric:?} rows {rows} iters {iters}");
                    let Some(arm) = arm else { continue };
                    let want = OrderStat::for_metric(metric);
                    let inline = if maximum_arm {
                        DeviceArm::Maximum
                    } else if let Some(want) = want {
                        DeviceArm::OrderStat(want)
                    } else {
                        match crate::fit::rank_metric_signed(metric) {
                            Some(signed) => DeviceArm::Rank { signed },
                            None => DeviceArm::Lm(fit_loss.expect("an LM objective")),
                        }
                    };
                    assert_eq!(arm, inline, "{metric:?} rows {rows} iters {iters}");
                    let lm_arm = want.is_none() && !maximum_arm && rank_loss.is_none();
                    for status in 0..10 {
                        let inline_tail = if lm_arm { tail_status_cause(status) } else { None };
                        assert_eq!(arm.tail_cause(status), inline_tail, "{metric:?} status {status}");
                    }
                }
            }
        }
    }

    /// #704: the replay adapter serves the outputs of a recording read from
    /// disk, one call per device step, and refuses a call for other programs.
    #[test]
    fn replay_device_serves_the_recorded_answers() {
        let mut rng = Mt::new(7);
        let trees: Vec<Node> = (0..9).map(|_| random_tree(&mut rng)).collect();
        let plans: Vec<GpuFitPlan> = trees.iter().filter_map(|t| crate::fit::gpu_fit_plan_reason(t).ok()).collect();
        assert!(plans.len() >= 3, "the fixture must plan several trees");
        let refs: Vec<&GpuFitPlan> = plans.iter().collect();
        let scripts: Vec<Script> = refs.iter().map(|_| script(&mut rng)).collect();
        let arm = DeviceArm::Lm(DeviceLoss::Squared);
        let call = recorded(&refs, &scripts, arm, false);
        let scratch = crate::test_scratch::Scratch::new("fit-route-replay");
        std::fs::write(scratch.join("call-0.bin"), call.encode()).expect("write call 0");
        std::fs::write(scratch.join("call-1.bin"), call.encode()).expect("write call 1");
        let recording = crate::fit_record::load_recording(&scratch).expect("load the recording");
        let mut device = ReplayDevice::from_recording(recording);
        let sample = rows(Metric::Squared, 40);
        let settings = settings(Metric::Squared, 0, false);
        let run = device.fit(arm, &refs, &settings, &sample);
        let want = answers_of(&call, &refs).expect("a recorded launch");
        assert_eq!(run.answers.as_ref(), Some(&want));
        for (answer, (plan, s)) in want.iter().zip(refs.iter().zip(&scripts)) {
            assert_eq!(answer.status, s.status);
            assert_eq!(answer.c.len(), plan.nconst);
        }
        assert_eq!(run.ms, 0.5);
        // The second call was recorded for the same programs; a different
        // start constant makes it another launch.
        let mut other: Vec<GpuFitPlan> = trees.iter().filter_map(|t| crate::fit::gpu_fit_plan_reason(t).ok()).collect();
        other[0].c0[0] += 1.0;
        let other_refs: Vec<&GpuFitPlan> = other.iter().collect();
        assert!(device.fit(arm, &other_refs, &settings, &sample).answers.is_none());
        assert!(device.fit(arm, &refs, &settings, &sample).answers.is_none(), "no call is left");
        let mut words = ReplayDevice::new([call.clone()]);
        assert!(words.fit(arm, &refs[1..], &settings, &sample).answers.is_none(), "other programs");
        let mut loss = ReplayDevice::new([call.clone()]);
        assert!(loss.fit(DeviceArm::Lm(DeviceLoss::Absolute), &refs, &settings, &sample).answers.is_none());
        let mut iters = ReplayDevice::new([call]);
        let more = FitSettings { iters: ITERS + 1, ..settings };
        assert!(iters.fit(arm, &refs, &more, &sample).answers.is_none());
    }
}
