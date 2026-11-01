//! #118: GPU accounting shared by the CUDA evaluator, the search report and
//! the benchmarks.
//!
//! The C backend times the full batch path (`stats_begin` → reserve/upload →
//! launch → sync → download) with disjoint host windows. This type accumulates
//! those windows across batches and derives the honest kernel-time duty
//! fraction. It is compiled without CUDA so the arithmetic has tests on every
//! machine.
//!
//! What this is *not*: `duty_cycle` is `kernel_s / wall_s`, a kernel-time
//! fraction. It is not `nvidia-smi`'s 1 Hz "some kernel was resident" sample and
//! it is not SM occupancy. Achieved occupancy needs `ncu`; this type leaves it
//! out entirely rather than inventing it.
//!
//! #284: [`GpuStats`] stays the evaluator-window sub-ledger. [`RunLedger`] is
//! the run-scoped ledger: one wall clock, exclusive stage seconds, and the
//! **union** of device intervals (never their sum). Every derived field is a
//! [`Ratio`] carrying numerator, denominator, units and scope, so a rate cannot
//! silently acquire the wrong denominator.

/// #284: the stages a run can spend wall time in. The names are the JSON keys
/// and match `09-dispatch-architecture.md` §7.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Variation, mutation, crossover and simplification.
    Vary,
    /// Candidate evaluation (including its host glue).
    Eval,
    /// Constant fitting, including the exact CPU fallback and finalist refits.
    Fit,
    /// Selection and promotion.
    Select,
    /// Pareto archive insertion and bookkeeping.
    Archive,
    /// Feature construction (enrichment) when it runs in the same run.
    Features,
    /// P&L / walk-forward work.
    Pnl,
    /// Loading, writing, checkpoints and hashing.
    Io,
    /// Every other measured main-thread window.
    HostOther,
}

impl Stage {
    pub const ALL: [Stage; 9] = [
        Stage::Vary,
        Stage::Eval,
        Stage::Fit,
        Stage::Select,
        Stage::Archive,
        Stage::Features,
        Stage::Pnl,
        Stage::Io,
        Stage::HostOther,
    ];

    /// Stages a search run has no work for by design. Enrichment and P&L are
    /// separate commands with their own ledger ([`RunLedger::phase_json`]); their
    /// value inside a search run is zero by construction, not by omission.
    pub fn structurally_zero_in_search(self) -> bool {
        matches!(self, Stage::Features | Stage::Pnl)
    }

    pub fn label(self) -> &'static str {
        match self {
            Stage::Vary => "vary",
            Stage::Eval => "eval",
            Stage::Fit => "fit",
            Stage::Select => "select",
            Stage::Archive => "archive",
            Stage::Features => "features",
            Stage::Pnl => "pnl",
            Stage::Io => "io",
            Stage::HostOther => "host_other",
        }
    }

    fn index(self) -> usize {
        match self {
            Stage::Vary => 0,
            Stage::Eval => 1,
            Stage::Fit => 2,
            Stage::Select => 3,
            Stage::Archive => 4,
            Stage::Features => 5,
            Stage::Pnl => 6,
            Stage::Io => 7,
            Stage::HostOther => 8,
        }
    }
}

/// #284: a derived value and the four facts needed to audit it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratio {
    pub value: f64,
    pub numerator: f64,
    pub denominator: f64,
    pub units: &'static str,
    pub scope: &'static str,
}

impl Ratio {
    /// A ratio with its own provenance. A zero or non-finite denominator makes
    /// the value `0.0`; the raw numerator and denominator are still reported so
    /// a reader can see why.
    pub fn new(
        numerator: f64,
        denominator: f64,
        units: &'static str,
        scope: &'static str,
    ) -> Self {
        let value = if numerator.is_finite() && denominator.is_finite() && denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
        Self { value, numerator, denominator, units, scope }
    }

    /// A ratio that is explicitly undefined: `value = null` in JSON, with the
    /// numerator/denominator that failed the precondition.
    pub fn undefined(
        numerator: f64,
        denominator: f64,
        units: &'static str,
        scope: &'static str,
    ) -> Self {
        Self { value: f64::NAN, numerator, denominator, units, scope }
    }

    pub fn is_defined(&self) -> bool {
        self.value.is_finite()
    }
}

/// #284: where an interval's placement on the run clock came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalSource {
    /// Start and stop are CUDA event timestamps, read back after the sync.
    Event,
    /// Placed from the host envelope of the call; used when events are not
    /// available.
    HostEnvelope,
}

/// #284: one half-open `[start_s, end_s)` interval on the run clock. Device
/// intervals are merged by [`RunLedger::device_busy_s`]; `stream` is retained
/// so a reader can see that two intervals were concurrent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub stage: Stage,
    pub stream: u32,
    pub start_s: f64,
    pub end_s: f64,
    pub source: IntervalSource,
}

/// #334: allocator activity of the persistent fit and score regions. Counters
/// are totals for the run, copied from the device context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FitAlloc {
    /// Device and pinned allocate and free calls, cold and steady.
    pub alloc_calls: u64,
    /// Growths to a new high-water mark.
    pub growths: u64,
    /// Allocator calls after the first successful batch of the same kind.
    pub steady_alloc_calls: u64,
    /// Fit and score batches that went through the regions.
    pub batches: u64,
}

/// #284: run-scoped telemetry. Stage seconds are exclusive main-thread windows;
/// device intervals are used only for the union. The two measures may differ
/// (device work can overlap host work), which is why the ledger reports both.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunLedger {
    stage_s: [f64; 9],
    intervals: Vec<Interval>,
    /// One clock for every derived field in this ledger.
    wall_s: f64,
    /// Measured evaluator kernel seconds, summed from the device's raw events.
    eval_kernel_s: f64,
    /// Whole evaluator full-path wall, the denominator for the evaluator-window
    /// kernel duty. It includes generation-0 scoring, which can happen before
    /// the generation loop's exclusive eval stage starts.
    eval_window_s: f64,
    /// Whole fit stage: device kernel + host upload/download + CPU fallback.
    fit_host_s: f64,
    /// Fit kernel seconds only, summed from the device's raw events.
    fit_kernel_s: f64,
    /// #496: the part of `fit_kernel_s` apportioned to the polish phase.
    fit_kernel_polish_s: f64,
    /// #498: exact host gate seconds of device proposals inside the fit stage.
    fit_gate_s: f64,
    /// Exact CPU fallback seconds inside the fit stage.
    fit_fallback_s: f64,
    /// Exact CPU f64 finalist refit seconds (a subset of the fit stage).
    refit_s: f64,
    /// Candidates admitted to the fit stage.
    admitted: u64,
    /// Admitted candidates the device fitted.
    device_fits: u64,
    /// Admitted candidates refit by the exact CPU arm.
    fallback_fits: u64,
    /// Fit fallbacks labelled by cause.
    fallback_by_cause: std::collections::BTreeMap<String, usize>,
    /// Fallbacks whose exact CPU refit lowered the exact error, by cause.
    fallback_improved_by_cause: std::collections::BTreeMap<String, usize>,
    /// Candidates scored by the evaluator's CPU fallback pool.
    eval_fallback_programs: u64,
    /// Wall seconds the evaluator's CPU fallback pool spent.
    eval_fallback_s: f64,
    /// Finalist refits performed.
    refits: u64,
    /// #334: allocator activity of the fit and score regions.
    fit_alloc: FitAlloc,
    /// Windows before the search clock starts (data load, preparation). They
    /// are not part of the run wall, so they never enter the closure.
    outside_s: [f64; 9],
}

impl RunLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one exclusive main-thread stage window. Negative and non-finite
    /// durations are ignored: a clock that went backwards is not evidence.
    pub fn record_stage(&mut self, stage: Stage, seconds: f64) {
        if seconds.is_finite() && seconds > 0.0 {
            self.stage_s[stage.index()] += seconds;
        }
    }

    /// #284: move a measured fit sub-window out of the enclosing eval window.
    /// Fit is nested inside eval (screening → fit → rescore), so it is
    /// reclassified rather than added: the stage sum must not grow.
    pub fn reclassify_eval_to_fit(&mut self, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0.0 {
            return;
        }
        let take = seconds.min(self.stage_s[Stage::Eval.index()]);
        self.stage_s[Stage::Eval.index()] -= take;
        self.stage_s[Stage::Fit.index()] += take;
    }

    /// Add one device interval. The interval is clamped to a finite,
    /// non-negative half-open window; an invalid interval is not recorded.
    pub fn record_interval(&mut self, stage: Stage, stream: u32, start_s: f64, end_s: f64) {
        self.record_interval_from(stage, stream, start_s, end_s, IntervalSource::HostEnvelope);
    }

    /// Add one device interval with an explicit placement source.
    pub fn record_interval_from(
        &mut self,
        stage: Stage,
        stream: u32,
        start_s: f64,
        end_s: f64,
        source: IntervalSource,
    ) {
        if !start_s.is_finite() || !end_s.is_finite() || end_s <= start_s {
            return;
        }
        let start_s = start_s.max(0.0);
        if end_s <= start_s {
            return;
        }
        self.intervals.push(Interval { stage, stream, start_s, end_s, source });
    }

    /// Record the evaluator's kernel window from CUDA event timestamps.
    /// `start_s` is the event start on the run clock.
    pub fn record_eval_kernel_event(&mut self, stream: u32, start_s: f64, kernel_s: f64) {
        if kernel_s.is_finite() && kernel_s > 0.0 {
            self.eval_kernel_s += kernel_s;
            self.record_interval_from(
                Stage::Eval,
                stream,
                start_s,
                start_s + kernel_s,
                IntervalSource::Event,
            );
        }
    }

    /// Record the evaluator's measured device kernel window. `start_s` is on the
    /// same run clock as the stage windows; `kernel_s` comes from the raw
    /// device event pair, not from the host envelope.
    pub fn record_eval_kernel(&mut self, stream: u32, start_s: f64, kernel_s: f64) {
        if kernel_s.is_finite() && kernel_s > 0.0 {
            self.eval_kernel_s += kernel_s;
            self.record_interval(Stage::Eval, stream, start_s, start_s + kernel_s);
        }
    }

    /// Record the fit stage totals. `host_s` includes the device call, the CPU
    /// fallback and write-back; `kernel_s` and `fallback_s` are its two
    /// non-overlapping measured leaves.
    pub fn record_fit(&mut self, host_s: f64, kernel_s: f64, fallback_s: f64) {
        if host_s.is_finite() && host_s > 0.0 {
            self.fit_host_s += host_s;
        }
        if kernel_s.is_finite() && kernel_s > 0.0 {
            self.fit_kernel_s += kernel_s;
        }
        if fallback_s.is_finite() && fallback_s > 0.0 {
            self.fit_fallback_s += fallback_s;
        }
    }

    /// #498: record the wall seconds of the exact device-proposal gate.
    pub fn record_fit_gate(&mut self, gate_s: f64) {
        if gate_s.is_finite() && gate_s > 0.0 {
            self.fit_gate_s += gate_s;
        }
    }

    /// #496: record the polish-phase share of the fit kernel seconds.
    pub fn record_fit_polish_split(&mut self, polish_s: f64) {
        if polish_s.is_finite() && polish_s > 0.0 {
            self.fit_kernel_polish_s += polish_s;
        }
    }

    /// Record one fit device kernel interval on the run clock.
    pub fn record_fit_kernel_interval(&mut self, stream: u32, start_s: f64, kernel_s: f64) {
        if kernel_s.is_finite() && kernel_s > 0.0 {
            self.record_interval(Stage::Fit, stream, start_s, start_s + kernel_s);
        }
    }

    /// Record one fit device kernel interval placed from CUDA event timestamps.
    pub fn record_fit_kernel_event(&mut self, stream: u32, start_s: f64, kernel_s: f64) {
        if kernel_s.is_finite() && kernel_s > 0.0 {
            self.record_interval_from(
                Stage::Fit,
                stream,
                start_s,
                start_s + kernel_s,
                IntervalSource::Event,
            );
        }
    }

    pub fn record_refit(&mut self, refits: u64, seconds: f64) {
        self.refits += refits;
        if seconds.is_finite() && seconds > 0.0 {
            self.refit_s += seconds;
        }
    }

    pub fn record_fit_counters(
        &mut self,
        admitted: u64,
        device_fits: u64,
        fallback_fits: u64,
        by_cause: std::collections::BTreeMap<String, usize>,
        improved_by_cause: std::collections::BTreeMap<String, usize>,
    ) {
        self.admitted += admitted;
        self.device_fits += device_fits;
        self.fallback_fits += fallback_fits;
        for (name, count) in by_cause {
            *self.fallback_by_cause.entry(name).or_default() += count;
        }
        for (name, count) in improved_by_cause {
            *self.fallback_improved_by_cause.entry(name).or_default() += count;
        }
    }

    /// #334: store the fit/score allocator counters of one device context. The
    /// counters are totals, so this replaces rather than adds.
    pub fn set_fit_alloc(&mut self, alloc: FitAlloc) {
        self.fit_alloc = alloc;
    }

    pub fn fit_alloc(&self) -> FitAlloc {
        self.fit_alloc
    }

    pub fn record_eval_fallback(&mut self, programs: u64, seconds: f64) {
        self.eval_fallback_programs += programs;
        if seconds.is_finite() && seconds > 0.0 {
            self.eval_fallback_s += seconds;
        }
    }

    /// #284: set the evaluator full-path wall, the scope of the evaluator-window
    /// kernel duty. It can cover generation-0 scoring outside the loop stages.
    pub fn set_eval_window(&mut self, seconds: f64) {
        if seconds.is_finite() && seconds > 0.0 {
            self.eval_window_s = seconds;
        }
    }

    pub fn eval_window_s(&self) -> f64 {
        self.eval_window_s
    }

    /// Merge another ledger's device intervals and canonical sub-ledgers into
    /// this one. Stage windows are **not** merged: the caller owns the single
    /// main-thread decomposition.
    pub fn absorb(&mut self, other: &RunLedger) {
        self.intervals.extend_from_slice(&other.intervals);
        for (mine, theirs) in self.outside_s.iter_mut().zip(other.outside_s) {
            *mine += theirs;
        }
        self.eval_kernel_s += other.eval_kernel_s;
        self.eval_window_s += other.eval_window_s;
        self.fit_host_s += other.fit_host_s;
        self.fit_kernel_s += other.fit_kernel_s;
        self.fit_kernel_polish_s += other.fit_kernel_polish_s;
        self.fit_fallback_s += other.fit_fallback_s;
        self.fit_gate_s += other.fit_gate_s;
        self.refit_s += other.refit_s;
        self.refits += other.refits;
        self.admitted += other.admitted;
        self.device_fits += other.device_fits;
        self.fallback_fits += other.fallback_fits;
        self.eval_fallback_programs += other.eval_fallback_programs;
        self.eval_fallback_s += other.eval_fallback_s;
        self.fit_alloc.alloc_calls += other.fit_alloc.alloc_calls;
        self.fit_alloc.growths += other.fit_alloc.growths;
        self.fit_alloc.steady_alloc_calls += other.fit_alloc.steady_alloc_calls;
        self.fit_alloc.batches += other.fit_alloc.batches;
        for (name, count) in &other.fallback_by_cause {
            *self.fallback_by_cause.entry(name.clone()).or_default() += *count;
        }
        for (name, count) in &other.fallback_improved_by_cause {
            *self
                .fallback_improved_by_cause
                .entry(name.clone())
                .or_default() += *count;
        }
    }

    /// Add a window that ran before the search clock started.
    pub fn record_outside(&mut self, stage: Stage, seconds: f64) {
        if seconds.is_finite() && seconds > 0.0 {
            self.outside_s[stage.index()] += seconds;
        }
    }

    pub fn outside_s(&self, stage: Stage) -> f64 {
        self.outside_s[stage.index()]
    }

    /// Number of device intervals placed from event timestamps and from the host
    /// envelope.
    pub fn interval_counts(&self) -> (usize, usize) {
        let event = self
            .intervals
            .iter()
            .filter(|i| i.source == IntervalSource::Event)
            .count();
        (event, self.intervals.len() - event)
    }

    /// How the device intervals were placed: `events`, `host_envelope`, `mixed`
    /// or `none`.
    pub fn interval_basis(&self) -> &'static str {
        match self.interval_counts() {
            (0, 0) => "none",
            (_, 0) => "events",
            (0, _) => "host_envelope",
            _ => "mixed",
        }
    }

    pub fn set_wall(&mut self, wall_s: f64) {
        if wall_s.is_finite() && wall_s > 0.0 {
            self.wall_s = wall_s;
        }
    }

    pub fn wall_s(&self) -> f64 {
        self.wall_s
    }

    pub fn stage_s(&self, stage: Stage) -> f64 {
        self.stage_s[stage.index()]
    }

    pub fn stage_sum_s(&self) -> f64 {
        self.stage_s.iter().sum()
    }

    pub fn intervals(&self) -> &[Interval] {
        &self.intervals
    }

    pub fn eval_kernel_s(&self) -> f64 {
        self.eval_kernel_s
    }

    pub fn fit_host_s(&self) -> f64 {
        self.fit_host_s
    }

    pub fn fit_kernel_s(&self) -> f64 {
        self.fit_kernel_s
    }

    /// #496: kernel seconds apportioned to the polish phase (re-baseline and
    /// Gauss-Newton steps) by the per-block cycle ratio.
    pub fn fit_kernel_polish_s(&self) -> f64 {
        self.fit_kernel_polish_s.min(self.fit_kernel_s)
    }

    /// #496: kernel seconds apportioned to the LM phase; the two parts sum to
    /// `fit_kernel_s`.
    pub fn fit_kernel_lm_s(&self) -> f64 {
        self.fit_kernel_s - self.fit_kernel_polish_s()
    }

    pub fn fit_gate_s(&self) -> f64 {
        self.fit_gate_s
    }

    pub fn fit_fallback_s(&self) -> f64 {
        self.fit_fallback_s
    }

    pub fn refit_s(&self) -> f64 {
        self.refit_s
    }

    pub fn refits(&self) -> u64 {
        self.refits
    }

    pub fn admitted(&self) -> u64 {
        self.admitted
    }

    pub fn device_fits(&self) -> u64 {
        self.device_fits
    }

    pub fn fallback_fits(&self) -> u64 {
        self.fallback_fits
    }

    pub fn fallback_by_cause(&self) -> &std::collections::BTreeMap<String, usize> {
        &self.fallback_by_cause
    }
    pub fn fallback_improved_by_cause(&self) -> &std::collections::BTreeMap<String, usize> {
        &self.fallback_improved_by_cause
    }
    pub fn fallback_improved(&self) -> u64 {
        self.fallback_improved_by_cause.values().map(|n| *n as u64).sum()
    }

    pub fn eval_fallback_programs(&self) -> u64 {
        self.eval_fallback_programs
    }

    pub fn eval_fallback_s(&self) -> f64 {
        self.eval_fallback_s
    }

    /// Arithmetic sum of every recorded device interval. This is **not** the
    /// duty numerator; it exists only so the ledger can show the double count
    /// that a sum would introduce.
    pub fn device_busy_sum_s(&self) -> f64 {
        self.intervals.iter().map(|i| i.end_s - i.start_s).sum()
    }

    /// Measure of the union of all device intervals, across every stream.
    pub fn device_busy_s(&self) -> f64 {
        union_len(&self.intervals)
    }

    /// Seconds counted twice by a naive sum: `sum - union`.
    pub fn device_overlap_s(&self) -> f64 {
        (self.device_busy_sum_s() - self.device_busy_s()).max(0.0)
    }

    /// Wall not attributed to any exclusive stage window.
    pub fn unaccounted_s(&self) -> f64 {
        (self.wall_s - self.stage_sum_s()).max(0.0)
    }

    /// `Σ stage_s + unaccounted = wall_run`.
    pub fn closure_error_s(&self) -> f64 {
        self.stage_sum_s() + self.unaccounted_s() - self.wall_s
    }

    /// The unaccounted share of run wall: `unaccounted_s / wall_s`. The
    /// acceptance threshold is `0.01` (1 %), so a reader sees the gap rather
    /// than a ratio that the `max(0)` in [`Self::unaccounted_s`] always zeroes.
    pub fn closure_residual(&self) -> Ratio {
        Ratio::new(self.unaccounted_s(), self.wall_s, "fraction", "run")
    }

    pub fn duty_run(&self) -> Ratio {
        Ratio::new(self.device_busy_s(), self.wall_s, "fraction", "run")
    }

    pub fn eval_kernel_duty(&self) -> Ratio {
        // The evaluator window, not the loop's exclusive eval stage: generation
        // 0 can be scored before the loop and still belongs to this duty.
        let denominator = if self.eval_window_s > 0.0 {
            self.eval_window_s
        } else {
            self.stage_s(Stage::Eval)
        };
        Ratio::new(self.eval_kernel_s, denominator, "fraction", "evaluator_window")
    }

    pub fn fit_kernel_duty(&self) -> Ratio {
        Ratio::new(self.fit_kernel_s, self.fit_host_s, "fraction", "fit_stage")
    }

    pub fn fallback_fraction(&self) -> Ratio {
        Ratio::new(
            self.fallback_fits as f64,
            self.admitted as f64,
            "fraction",
            "admitted_fits",
        )
    }

    /// #284: the ledger of a command that is not a search (`enrich`,
    /// `walkforward`, `ladder`). Same keys as the search ledger's run-clock
    /// block; `zero_stages` names the stages the command has no work for by
    /// design.
    pub fn phase_json(&self, command: &str, zero_stages: &[Stage]) -> String {
        fn num(v: f64) -> String {
            if v.is_finite() {
                format!("{v}")
            } else {
                "null".to_string()
            }
        }
        fn quote(s: &str) -> String {
            format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        let ratio = |r: Ratio| {
            format!(
                "{{\"value\":{},\"numerator\":{},\"denominator\":{},\"units\":{},\"scope\":{}}}",
                num(r.value),
                num(r.numerator),
                num(r.denominator),
                quote(r.units),
                quote(r.scope)
            )
        };
        let stages = Stage::ALL
            .iter()
            .map(|s| format!("{}:{}", quote(&format!("{}_s", s.label())), num(self.stage_s(*s))))
            .collect::<Vec<_>>()
            .join(",");
        let zero = zero_stages
            .iter()
            .map(|s| quote(s.label()))
            .collect::<Vec<_>>()
            .join(",");
        let (by_event, by_host) = self.interval_counts();
        format!(
            "{{\"scope\":\"run\",\"command\":{},\"wall_s\":{},\"stages\":{{{}}},\"stage_sum_s\":{},\"unaccounted_s\":{},\"closure\":{},\"device_busy_s\":{},\"device_busy_sum_s\":{},\"device_overlap_s\":{},\"device_intervals\":{},\"device_interval_basis\":{{\"basis\":{},\"events\":{},\"host_envelope\":{}}},\"structural_zero_stages\":[{}],\"duty_run\":{}}}",
            quote(command),
            num(self.wall_s()),
            stages,
            num(self.stage_sum_s()),
            num(self.unaccounted_s()),
            ratio(self.closure_residual()),
            num(self.device_busy_s()),
            num(self.device_busy_sum_s()),
            num(self.device_overlap_s()),
            self.intervals().len(),
            quote(self.interval_basis()),
            by_event,
            by_host,
            zero,
            ratio(self.duty_run())
        )
    }

    /// Decades of error reduction per wall-hour, using **like units** for the
    /// two errors (both MSE or both SSE, never one of each).
    pub fn quality_per_gpu_hour(&self, best_error: f64, baseline_error: f64) -> Ratio {
        let decades = if best_error.is_finite()
            && best_error > 0.0
            && baseline_error.is_finite()
            && baseline_error > 0.0
            && self.wall_s > 0.0
        {
            -(best_error / baseline_error).log10()
        } else {
            f64::NAN
        };
        let hours = self.wall_s / 3600.0;
        if decades.is_finite() {
            Ratio::new(decades, hours, "decades_per_gpu_hour", "run")
        } else {
            Ratio::undefined(decades, hours, "decades_per_gpu_hour", "run")
        }
    }
}

/// #284: length of the union of half-open intervals. Sorting by start and
/// merging on `start <= current_end` is the standard sweep; equal endpoints do
/// not double count a zero-length gap.
fn union_len(intervals: &[Interval]) -> f64 {
    if intervals.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<&Interval> = intervals.iter().collect();
    sorted.sort_by(|a, b| {
        a.start_s
            .partial_cmp(&b.start_s)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.end_s.partial_cmp(&b.end_s).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut total = 0.0;
    let mut cur_start = sorted[0].start_s;
    let mut cur_end = sorted[0].end_s;
    for iv in sorted.iter().skip(1) {
        if iv.start_s <= cur_end {
            cur_end = cur_end.max(iv.end_s);
        } else {
            total += cur_end - cur_start;
            cur_start = iv.start_s;
            cur_end = iv.end_s;
        }
    }
    total + (cur_end - cur_start)
}

/// #118: accumulated GPU accounting for a run, in seconds and bytes.
///
/// `wall_s` is measured by the Rust caller around the whole evaluator call, so
/// it includes the C host windows and the small Rust glue between them.
/// `kernel_s` is GPU-event time inside that wall and is deliberately not part of
/// [`GpuStats::accounted_s`]: it overlaps the launch/sync windows and would
/// double count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuStats {
    /// GPU-event time between the launch and completion events.
    pub kernel_s: f64,
    /// Host time in the kernel launch calls (including event records).
    pub launch_s: f64,
    /// Host time blocked in `cudaEventSynchronize`.
    pub sync_s: f64,
    /// Host time in `cudaMalloc`/`cudaFree`/`cudaMallocManaged` (synchronising).
    pub alloc_s: f64,
    /// Host time in the host→device program upload.
    pub h2d_s: f64,
    /// Host time in the device→host SSE download.
    pub d2h_s: f64,
    /// Host glue outside the measured C windows: Rust-side allocation and the
    /// gaps between reserve/upload/eval calls. It is the difference between the
    /// independently measured outer wall and the C envelope, so the ledger
    /// closes to wall instead of hiding an `other` bucket.
    pub host_s: f64,
    /// Independently measured wall of every evaluator call.
    pub wall_s: f64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    /// Number of evaluator batches (generation evaluations).
    pub batches: u64,
    /// Number of compiled programs scored.
    pub programs: u64,
    /// Number of rows in the training/evaluation sample.
    pub rows: u64,
    /// Candidate evaluations: one per program scored, i.e. `programs`.
    pub evals: u64,
    /// Row evaluations: `programs × rows`.
    pub row_evals: u64,
    /// Batches that returned an error after being timed.
    pub failed_batches: u64,
}

impl GpuStats {
    /// Add one batch's ledger. `wall_ms` is the independently measured envelope
    /// of the whole evaluator call. `failed` marks a batch that returned an
    /// error: its wall and windows are real, but it contributes no evaluations,
    /// so a failed batch can never inflate `evals/s`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_batch(
        &mut self,
        kernel_ms: f64,
        launch_ms: f64,
        sync_ms: f64,
        alloc_ms: f64,
        h2d_ms: f64,
        d2h_ms: f64,
        host_ms: f64,
        wall_ms: f64,
        h2d_bytes: u64,
        d2h_bytes: u64,
        programs: u64,
        rows: u64,
        failed: bool,
    ) {
        // A failed launch can return a negative kernel time; the event pair did
        // not produce a reading, so keep the accumulated time honest.
        self.kernel_s += kernel_ms.max(0.0) / 1e3;
        self.launch_s += launch_ms.max(0.0) / 1e3;
        self.sync_s += sync_ms.max(0.0) / 1e3;
        self.alloc_s += alloc_ms.max(0.0) / 1e3;
        self.h2d_s += h2d_ms.max(0.0) / 1e3;
        self.d2h_s += d2h_ms.max(0.0) / 1e3;
        self.host_s += host_ms.max(0.0) / 1e3;
        self.wall_s += wall_ms.max(0.0) / 1e3;
        self.h2d_bytes += h2d_bytes;
        self.d2h_bytes += d2h_bytes;
        self.batches += 1;
        if failed {
            self.failed_batches += 1;
        } else {
            self.programs += programs;
            self.rows = rows;
            self.evals += programs;
            self.row_evals += programs * rows;
        }
    }

    /// Subtract an earlier snapshot, for timing a benchmark window.
    pub fn since(&self, earlier: &GpuStats) -> GpuStats {
        GpuStats {
            kernel_s: self.kernel_s - earlier.kernel_s,
            launch_s: self.launch_s - earlier.launch_s,
            sync_s: self.sync_s - earlier.sync_s,
            alloc_s: self.alloc_s - earlier.alloc_s,
            h2d_s: self.h2d_s - earlier.h2d_s,
            d2h_s: self.d2h_s - earlier.d2h_s,
            host_s: self.host_s - earlier.host_s,
            wall_s: self.wall_s - earlier.wall_s,
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            batches: self.batches.saturating_sub(earlier.batches),
            programs: self.programs.saturating_sub(earlier.programs),
            rows: self.rows,
            evals: self.evals.saturating_sub(earlier.evals),
            row_evals: self.row_evals.saturating_sub(earlier.row_evals),
            failed_batches: self.failed_batches.saturating_sub(earlier.failed_batches),
        }
    }

    /// Both copies, the way a report usually wants them.
    pub fn transfer_s(&self) -> f64 {
        self.h2d_s + self.d2h_s
    }

    /// The measured C windows alone. `wall_s − window_s` is the host glue that
    /// [`Self::host_s`] records.
    pub fn window_s(&self) -> f64 {
        self.alloc_s + self.h2d_s + self.launch_s + self.sync_s + self.d2h_s
    }

    /// Disjoint phases that should add up to `wall_s` (kernel time overlaps and
    /// is excluded). `host_s` is the residue outside the C windows, measured
    /// against the independent outer wall rather than guessed.
    pub fn accounted_s(&self) -> f64 {
        self.window_s() + self.host_s
    }

    /// Wall not spent inside the kernel: host overhead plus GPU idle.
    pub fn idle_s(&self) -> f64 {
        (self.wall_s - self.kernel_s).max(0.0)
    }

    /// Kernel-time fraction of the measured full path. A duty estimate, not
    /// occupancy.
    pub fn duty_cycle(&self) -> f64 {
        if self.wall_s > 0.0 {
            (self.kernel_s / self.wall_s).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Candidate evaluations per full-path second.
    pub fn evals_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.evals as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// Row evaluations (`programs × rows`) per full-path second.
    pub fn row_evals_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.row_evals as f64 / self.wall_s
        } else {
            0.0
        }
    }

    pub fn generations_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.batches as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// Decades of error reduction against the constant-mean baseline per
    /// evaluator-window GPU-hour: `-log10(best_error / baseline_error) / hours`.
    /// Positive means the searched model beats the trivial mean predictor.
    /// `best_error` and `baseline_error` **must be the same unit and the same
    /// rows** (both MSE or both SSE); the caller owns that contract. `None`
    /// when either error is not finite/positive or no GPU time was measured.
    /// #284 adds the run-scoped, provenance-carrying
    /// [`RunLedger::quality_per_gpu_hour`].
    pub fn quality_per_gpu_hour(&self, best_error: f64, baseline_error: f64) -> Option<f64> {
        if !best_error.is_finite()
            || best_error <= 0.0
            || !baseline_error.is_finite()
            || baseline_error <= 0.0
            || !(self.wall_s > 0.0)
        {
            return None;
        }
        Some((-(best_error / baseline_error).log10()) / (self.wall_s / 3600.0))
    }

    /// The additive host decomposition agrees with the independently measured
    /// wall to within `tol` (relative). Used as an accounting test and reported
    /// as `residual_fraction` so a reader can see the unmeasured glue.
    pub fn decomposition_within(&self, tol: f64) -> bool {
        self.wall_s > 0.0 && (self.accounted_s() - self.wall_s).abs() <= tol * self.wall_s
    }

    /// Signed relative residual `(accounted − wall) / wall`.
    pub fn residual_fraction(&self) -> f64 {
        if self.wall_s > 0.0 {
            (self.accounted_s() - self.wall_s) / self.wall_s
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #118: the host windows are disjoint and add up to wall; kernel time is
    /// overlapping and only feeds duty/idle, never the additive sum.
    #[test]
    fn gpu_accounting_decomposes_wall_and_reports_honest_duty() {
        let mut st = GpuStats::default();
        st.add_batch(2.0, 0.1, 3.0, 1.0, 0.4, 0.5, 0.0, 5.0, 800, 400, 10, 128, false);
        assert!(st.decomposition_within(1e-12));
        assert!((st.accounted_s() - 0.005).abs() < 1e-12);
        assert!((st.wall_s - 0.005).abs() < 1e-12);
        assert_eq!(st.evals, 10);
        assert_eq!(st.programs, 10);
        assert_eq!(st.row_evals, 1280);
        assert_eq!(st.rows, 128);
        assert_eq!(st.batches, 1);
        assert_eq!(st.h2d_bytes + st.d2h_bytes, 1200);
        assert!((st.evals_per_s() - 10.0 / 0.005).abs() < 1e-6);
        assert!((st.row_evals_per_s() - 1280.0 / 0.005).abs() < 1e-6);
        // Kernel is 2 ms of a 5 ms path: 40% duty, 3 ms idle.
        assert!((st.duty_cycle() - 0.4).abs() < 1e-12);
        assert!((st.idle_s() - 0.003).abs() < 1e-12);
        // 1e-6 against a 1.0 baseline is six decades removed.
        assert!(
            (st.quality_per_gpu_hour(1e-6, 1.0).unwrap() - 6.0 / (0.005 / 3600.0)).abs() < 1e-6
        );
        assert!(st.quality_per_gpu_hour(0.0, 1.0).is_none());

        let mut twice = st;
        twice.add_batch(2.0, 0.1, 3.0, 1.0, 0.4, 0.5, 0.0, 5.0, 800, 400, 10, 128, false);
        assert_eq!(twice.batches, 2);
        assert_eq!(twice.evals, 20);
        assert!((twice.wall_s - 2.0 * st.wall_s).abs() < 1e-12);
    }

    /// #118: a failed batch keeps its measured wall and transfer bytes but adds
    /// no evaluations, so it cannot make throughput or quality look better.
    #[test]
    fn failed_batch_is_timed_but_adds_no_evals() {
        let mut st = GpuStats::default();
        st.add_batch(-1.0, 0.2, 0.3, 0.4, 0.1, 0.0, 0.0, 1.0, 64, 0, 8, 1000, true);
        assert_eq!(st.batches, 1);
        assert_eq!(st.failed_batches, 1);
        assert_eq!(st.evals, 0);
        assert_eq!(st.row_evals, 0);
        assert_eq!(st.evals_per_s(), 0.0);
        assert!((st.wall_s - 0.001).abs() < 1e-12);
        assert!((st.accounted_s() - 0.001).abs() < 1e-12);
        assert!(st.decomposition_within(1e-9));

        st.add_batch(2.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 4.0, 64, 4, 8, 1000, false);
        assert_eq!(st.evals, 8);
        assert_eq!(st.failed_batches, 1);
    }

    /// #118: `host_s` is the residue between the outer wall and the C envelope;
    /// adding it makes the phase sum close without an opaque `other` bucket.
    #[test]
    fn host_glue_closes_the_phase_ledger() {
        let mut st = GpuStats::default();
        st.add_batch(1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 8, 0, 2, 10, false);
        assert!((st.window_s() - 0.0005).abs() < 1e-12);
        assert!((st.host_s - 0.0001).abs() < 1e-12);
        assert!((st.accounted_s() - 0.0006).abs() < 1e-12);
        assert!(st.decomposition_within(1e-12));
        assert!(st.residual_fraction().abs() < 1e-12);
    }

    /// #284: device-busy is the interval **union**, not the sum. Two streams
    /// overlap for 1 s; the sum says 4 s, the union says 3 s, duty cannot exceed
    /// one, and the double-count is reported rather than hidden.
    #[test]
    fn run_ledger_unions_overlapping_streams() {
        let mut l = RunLedger::new();
        l.record_interval(Stage::Eval, 0, 0.0, 2.0);
        l.record_interval(Stage::Fit, 1, 1.0, 3.0);
        l.set_wall(5.0);
        assert!((l.device_busy_sum_s() - 4.0).abs() < 1e-12);
        assert!((l.device_busy_s() - 3.0).abs() < 1e-12);
        assert!((l.device_overlap_s() - 1.0).abs() < 1e-12);
        let duty = l.duty_run();
        assert!((duty.value - 0.6).abs() < 1e-12);
        assert!(duty.value <= 1.0);
        // No stage windows were recorded, so the closure residual is the whole
        // run wall rather than a fabricated zero.
        assert!((l.closure_residual().value - 1.0).abs() < 1e-12);
        assert_eq!(l.closure_residual().numerator, 5.0);
        assert_eq!(duty.numerator, 3.0);
        assert_eq!(duty.denominator, 5.0);
        assert_eq!(duty.units, "fraction");
        assert_eq!(duty.scope, "run");
        // Nested intervals are one union member, not two.
        l.record_interval(Stage::Eval, 2, 0.5, 1.5);
        assert!((l.device_busy_s() - 3.0).abs() < 1e-12);
    }

    /// #284: the ledger closes to the run wall and reports an explicit residual
    /// instead of inventing an `other` bucket. A 1,000-generation replay is the
    /// acceptance shape: 1 % of wall is the threshold.
    #[test]
    fn run_ledger_closes_over_a_thousand_generation_replay() {
        let mut l = RunLedger::new();
        let mut t = 0.0;
        for gen in 0..1_000 {
            let prep = 0.0001;
            let vary = 0.01;
            let eval = 0.02;
            let select = 0.005;
            let archive = 0.002;
            l.record_stage(Stage::HostOther, prep);
            l.record_stage(Stage::Vary, vary);
            l.record_stage(Stage::Eval, eval);
            l.record_stage(Stage::Select, select);
            l.record_stage(Stage::Archive, archive);
            let fit = 0.004;
            l.record_fit(fit, 0.003, 0.0005);
            // Fit is a sub-window of eval, so the exclusive eval stage is the
            // measured eval window minus the nested fit window.
            l.record_stage(Stage::Eval, -fit); // ignored by design
            l.record_fit_counters(1, if gen % 2 == 0 { 1 } else { 0 }, if gen % 2 == 0 { 0 } else { 1 }, std::collections::BTreeMap::new(), std::collections::BTreeMap::new());
            l.record_interval(Stage::Eval, 0, t, t + 0.015);
            l.record_interval(Stage::Fit, 1, t + 0.012, t + 0.016);
            t += prep + vary + eval + select + archive;
        }
        l.set_wall(t);
        assert!(l.closure_residual().value.abs() < 0.01, "closure {:?}", l.closure_residual());
        assert!(l.duty_run().value <= 1.0);
        assert_eq!(l.admitted(), 1_000);
        assert_eq!(l.device_fits() + l.fallback_fits(), 1_000);
    }

    /// #284: quality per GPU-hour takes two errors in the same unit. Using an
    /// SSE baseline against an MSE best is the #256 defect; here the two are
    /// both MSE and the answer is the documented 4.844205.
    #[test]
    fn quality_per_gpu_hour_requires_like_units() {
        let mut l = RunLedger::new();
        l.set_wall(7.467_618_67);
        let q = l.quality_per_gpu_hour(0.103_464_003_567_670_26, 0.105_885_819_341_126_06);
        assert!((q.value - 4.844_205).abs() < 1e-5, "{:?}", q);
        assert_eq!(q.units, "decades_per_gpu_hour");
        assert_eq!(q.scope, "run");
        assert!((q.denominator - 7.467_618_67 / 3600.0).abs() < 1e-15);
    }

    /// #265: improved refits are kept per cause, summed, and merged.
    #[test]
    fn ledger_counts_improved_fallbacks_per_cause() {
        let causes = |pairs: &[(&str, usize)]| -> std::collections::BTreeMap<String, usize> {
            pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
        };
        let mut a = RunLedger::new();
        a.record_fit_counters(10, 5, 5, causes(&[("worse", 3), ("launch", 2)]), causes(&[("worse", 2)]));
        let mut b = RunLedger::new();
        b.record_fit_counters(4, 2, 2, causes(&[("worse", 2)]), causes(&[("worse", 1), ("launch", 1)]));
        a.absorb(&b);
        assert_eq!(a.fallback_improved(), 4);
        assert_eq!(a.fallback_improved_by_cause(), &causes(&[("launch", 1), ("worse", 3)]));
        for (cause, improved) in a.fallback_improved_by_cause() {
            assert!(improved <= &a.fallback_by_cause()[cause]);
        }
    }

    /// #284: `since()` differences a benchmark window without losing bytes.
    #[test]
    fn since_subtracts_a_snapshot() {
        let mut st = GpuStats::default();
        st.add_batch(1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 2.0, 10, 20, 4, 100, false);
        let before = st;
        st.add_batch(2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 4.0, 30, 40, 8, 100, false);
        let d = st.since(&before);
        assert_eq!(d.batches, 1);
        assert_eq!(d.evals, 8);
        assert_eq!(d.h2d_bytes, 30);
        assert_eq!(d.d2h_bytes, 40);
        assert!((d.wall_s - 0.004).abs() < 1e-12);
        assert!((d.accounted_s() - 0.001).abs() < 1e-12); // five 0.2 ms windows
    }

    /// #284: an interval placed from event timestamps is labelled as such; the
    /// host envelope stays the labelled fallback and a mix says so.
    #[test]
    fn run_ledger_labels_event_and_host_envelope_intervals() {
        let mut l = RunLedger::new();
        assert_eq!(l.interval_basis(), "none");
        l.record_eval_kernel_event(0, 1.0, 0.5);
        l.record_fit_kernel_event(1, 1.25, 0.5);
        assert_eq!(l.interval_basis(), "events");
        assert_eq!(l.interval_counts(), (2, 0));
        assert!((l.device_busy_s() - 0.75).abs() < 1e-12);
        l.record_eval_kernel(0, 4.0, 0.5);
        assert_eq!(l.interval_basis(), "mixed");
        assert_eq!(l.interval_counts(), (2, 1));
        let mut host = RunLedger::new();
        host.record_fit_kernel_interval(0, 0.0, 1.0);
        assert_eq!(host.interval_basis(), "host_envelope");
        l.absorb(&host);
        assert_eq!(l.interval_counts(), (2, 2));
    }

    /// #284: stages with no work in a search run are named, and windows before
    /// the run clock stay outside the closure.
    #[test]
    fn run_ledger_names_structural_zero_stages_and_keeps_outside_windows_out_of_closure() {
        let zero: Vec<&str> = Stage::ALL
            .iter()
            .filter(|s| s.structurally_zero_in_search())
            .map(|s| s.label())
            .collect();
        assert_eq!(zero, ["features", "pnl"]);
        let mut l = RunLedger::new();
        l.record_outside(Stage::Io, 2.0);
        l.record_outside(Stage::Features, 1.5);
        l.record_stage(Stage::Eval, 1.0);
        l.set_wall(1.0);
        assert_eq!(l.outside_s(Stage::Io), 2.0);
        assert_eq!(l.outside_s(Stage::Features), 1.5);
        assert_eq!(l.stage_s(Stage::Io), 0.0);
        assert!(l.closure_residual().value.abs() < 1e-12);
        let mut total = RunLedger::new();
        total.absorb(&l);
        assert_eq!(total.outside_s(Stage::Io), 2.0);
    }
}
