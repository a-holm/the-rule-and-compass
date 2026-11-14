//! #1081: the island policy. One module decides, for both generation loops,
//! where each island draws its parents, which islands receive a migrant and
//! from where, and which islands restart. The owned-tree islands of
//! `run_search` and the flat postfix `Arena` of `run_postfix_search` are
//! population adapters: they apply a decision and decide nothing.
//!
//! No decision draws from an RNG. #1029: the tournament migrant of
//! `--migration-policy ring-tournament` is resolved by the population adapter
//! with the destination island's generator. A restart draws its trees from the
//! population adapter's own generator, so the node path keeps its `Mt` stream
//! per island and the flat path its positional `Philox` draws.
//!
//! #1100: with `--explore-islands K` the highest `K` islands form an explore
//! tier. They take no front parent and no `--qd-policy` parent, send and
//! receive no migrant, and restart at the end of each lifetime of a Luby
//! schedule that depends on the generation alone. The other islands (the
//! exploit tier) keep the restart rule and a ring of their own; explore
//! results reach them only through the archive.
//!
//! #1070: with `--front-scope island` each island draws its front parents from
//! its own error/complexity front instead of the global one. A front travels
//! only along a migration move, where the destination front becomes the
//! canonical front of both; a restart empties the island's front.

use super::{
    afp_worst, canonical_front, defers_fit_to_gpu, draw_row_dependent, fit_constants, gate_fit, population_best,
    FitAcceptCounts, FitPath,
    pre_fit_tree, qd_descriptor_of, tournament, Config, EvalSample, FrontScope, Ind, MigrationPolicy, RestartPolicy, Selection,
    FRONT_PARENT_P, MIGRANT_TOURNAMENT,
};
use crate::checkpoint::IslandStall;
use crate::data::Sample;
use crate::expr::{self, random_tree_with, AggTable, Node};
use crate::pareto::{Archive, Sol};
#[cfg(feature = "cuda")]
use crate::postfix::{Arena, Variation};
use crate::qd::{self, QdArchive};
use crate::rng::Mt;

/// Where one island draws its parents in one generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ParentSource {
    /// Probability that a parent draw takes a uniform member of the global
    /// front (#76).
    pub front_p: f64,
    /// Whether the island may draw from the pools `--qd-policy` shares across
    /// islands: its global-front share and the QD archive (#120).
    pub shared: bool,
}

/// One migration: the best member of island `from` replaces one member of
/// island `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Move {
    pub from: usize,
    pub to: usize,
}

/// #1029: what one migration step did: migrants copied, and migrants skipped
/// because the destination already held the migrant's error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Migrants {
    pub sent: usize,
    pub skipped: usize,
}

/// Read side of a population adapter.
pub(crate) trait IslandBests {
    fn island_count(&self) -> usize;
    /// The island's lowest error. A `NaN` (an unscored member) is skipped; an
    /// island without a non-`NaN` error gives `+inf`.
    fn best_error(&self, island: usize) -> f64;
}

/// #1070: the members of a population adapter, for the island fronts.
pub(crate) trait IslandMembers: IslandBests {
    /// Offer every member of `island` to `front`, in population order, as the
    /// global archive update does. `keep` decides whether a tree the front
    /// admits may enter (#848).
    fn offer(&self, island: usize, front: &mut Archive, keep: &dyn Fn(&Node) -> bool);
}

/// Write side of a population adapter.
pub(crate) trait IslandPopulation: IslandBests {
    /// Copy the migrant of each move's source island over one member of its
    /// destination: the best member under `ring`, a tournament winner drawn by
    /// the adapter with the destination's own generator under
    /// `ring-tournament` (#1029). Every migrant and every victim is chosen
    /// before the first copy.
    fn migrate(&mut self, moves: &[Move], policy: MigrationPolicy) -> Migrants;
    /// Replace every member of `island` with fresh random material.
    fn restart(&mut self, island: usize);
}

/// #1022: per-island restart bookkeeping under `--restart-policy island`.
/// Every counter is a pure function of the trajectory; nothing here draws from
/// an RNG.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct RestartState {
    /// Generations since each island's best error last improved.
    since_improve: Vec<usize>,
    /// Generations of isolation left after a restart.
    isolated: Vec<usize>,
    /// Each island's best error so far (`+inf` at the start and after a restart).
    best_prev: Vec<f64>,
}

impl RestartState {
    fn new(islands: usize) -> Self {
        RestartState {
            since_improve: vec![0; islands],
            isolated: vec![0; islands],
            best_prev: vec![f64::INFINITY; islands],
        }
    }

    /// The saved counters of a resume; a checkpoint written under the archive
    /// policy has none.
    fn from_saved(saved: &[Option<IslandStall>]) -> Result<Self, String> {
        let mut state = RestartState::new(saved.len());
        for (i, stall) in saved.iter().enumerate() {
            let stall = stall.ok_or_else(|| {
                "checkpoint has no island_stall records; it was written under --restart-policy archive"
                    .to_string()
            })?;
            state.since_improve[i] = stall.since_improve;
            state.isolated[i] = stall.isolated;
            state.best_prev[i] = stall.best_prev;
        }
        Ok(state)
    }

    fn saved(&self, island: usize) -> IslandStall {
        IslandStall {
            since_improve: self.since_improve[island],
            isolated: self.isolated[island],
            best_prev: self.best_prev[island],
        }
    }

    /// One generation's stall update, after the archive update. `island_best`
    /// is each island's best error.
    fn tick(&mut self, island_best: &[f64]) {
        for (ii, &cur) in island_best.iter().enumerate() {
            if cur.is_finite() && !crate::progress_stats::no_improvement(self.best_prev[ii], cur) {
                self.since_improve[ii] = 0;
            } else {
                self.since_improve[ii] += 1;
            }
            self.best_prev[ii] = self.best_prev[ii].min(cur);
            self.isolated[ii] = self.isolated[ii].saturating_sub(1);
        }
    }

    /// The lowest island whose best error equals the archive's; it is never
    /// restarted. `None` when no island holds the archive best.
    fn keeper(island_best: &[f64], arch_best: f64) -> Option<usize> {
        island_best.iter().position(|&b| b == arch_best)
    }

    fn due(&self, island: usize, keeper: Option<usize>, interval: usize) -> bool {
        self.since_improve[island] >= interval && Some(island) != keeper
    }

    fn restarted(&mut self, island: usize, isolation: usize) {
        self.since_improve[island] = 0;
        self.best_prev[island] = f64::INFINITY;
        self.isolated[island] = isolation;
    }

    fn is_isolated(&self, island: usize) -> bool {
        self.isolated[island] > 0
    }

    /// The global-front parent probability of `island` this generation.
    fn front_p(&self, island: usize, base: f64) -> f64 {
        if self.is_isolated(island) {
            0.0
        } else {
            base
        }
    }

    fn isolated_count(&self) -> usize {
        self.isolated.iter().filter(|&&w| w > 0).count()
    }
}

/// #1022: the stall rule that restarts an island.
#[derive(Clone, Debug, PartialEq)]
enum RestartRule {
    /// Two checks in a row in which the island's best error is worse than the
    /// archive best; `IslandPolicy::stagnant` counts the checks.
    ArchiveStall,
    /// The island's own stall; the keeper is spared and a restarted island is
    /// isolated.
    IslandStall(RestartState),
}

fn due(interval: usize, gen: usize) -> bool {
    interval > 0 && (gen + 1) % interval == 0
}

/// #1100: the `n`-th term (`n >= 1`) of the Luby sequence 1, 1, 2, 1, 1, 2,
/// 4, ... (Luby, Sinclair and Zuckerman 1993).
fn luby(mut n: u64) -> u64 {
    loop {
        let mut k = 1;
        while (1u64 << k) - 1 < n {
            k += 1;
        }
        if n == (1u64 << k) - 1 {
            return 1u64 << (k - 1);
        }
        n -= (1u64 << (k - 1)) - 1;
    }
}

/// #1100: the explore slots (`0..k`) whose lifetime ends after `units`
/// lifetime units. One Luby sequence is list-scheduled onto the slots: each
/// next term goes to the slot with the least accumulated length, ties to the
/// lowest slot (Cire, Kadioglu and Sellmann 2014). Every slot starts its first
/// lifetime at unit 0.
fn explore_restarts(k: usize, units: u64) -> Vec<usize> {
    if k == 0 || units == 0 {
        return Vec::new();
    }
    let mut ends = vec![0u64; k];
    let mut n = 0;
    loop {
        let slot = (0..k).min_by_key(|&s| (ends[s], s)).expect("k > 0");
        if ends[slot] >= units {
            break;
        }
        n += 1;
        ends[slot] += luby(n);
    }
    (0..k).filter(|&s| ends[s] == units).collect()
}

/// The island policy of one run: parent sources at the start of a generation,
/// the stall update after the archive update, and the migration and restart
/// decisions at the generation barrier.
///
/// Call order per generation: [`IslandPolicy::parent_source`] and
/// [`IslandPolicy::front`] for each island, then [`IslandPolicy::offer`] and
/// [`IslandPolicy::observe`] after the archive update, then
/// [`IslandPolicy::barrier`]. A run that stops early skips the barrier of its
/// last generation, never the observation.
#[derive(Clone, Debug)]
pub(crate) struct IslandPolicy {
    migration_interval: usize,
    migration_policy: MigrationPolicy,
    restart_interval: usize,
    isolation: usize,
    front_p: f64,
    /// Archive-stall checks per island. Saved in every checkpoint; all zero
    /// under the island rule.
    stagnant: Vec<usize>,
    rule: RestartRule,
    /// Restarts since this process started the run (not saved).
    restarts_total: u64,
    /// #1100: islands `explore_from..` form the explore tier.
    explore_from: usize,
    /// #1070: one front per island under `--front-scope island`. Saved in the
    /// checkpoint as `island_front` records.
    fronts: Option<Vec<Archive>>,
    /// #1029: the migration step of the current generation.
    migrants: Migrants,
}

/// #1070: the cap of an island front, the cap of the global archive of
/// `run_search`.
const ISLAND_FRONT_CAP: usize = 4000;

impl IslandPolicy {
    pub(crate) fn new(cfg: &Config) -> Self {
        let islands = cfg.islands.max(1);
        let state = (cfg.restart_policy == RestartPolicy::Island).then(|| RestartState::new(islands));
        IslandPolicy::with_state(cfg, vec![0; islands], state)
    }

    /// The policy of a resumed run: the saved archive-stall checks, under
    /// `--restart-policy island` the saved `island_stall` records, and under
    /// `--front-scope island` the saved members of each island front.
    pub(crate) fn resume(
        cfg: &Config,
        stagnant: Vec<usize>,
        stalls: &[Option<IslandStall>],
        fronts: Vec<Vec<Sol>>,
    ) -> Result<Self, String> {
        let state = if cfg.restart_policy == RestartPolicy::Island {
            Some(RestartState::from_saved(stalls)?)
        } else {
            None
        };
        let mut policy = IslandPolicy::with_state(cfg, stagnant, state);
        if let Some(own) = policy.fronts.as_mut() {
            for (front, items) in own.iter_mut().zip(fronts) {
                front.items = items;
            }
        }
        Ok(policy)
    }

    fn with_state(cfg: &Config, stagnant: Vec<usize>, state: Option<RestartState>) -> Self {
        IslandPolicy {
            migration_interval: cfg.migration_interval,
            migration_policy: cfg.migration_policy,
            restart_interval: cfg.restart_interval,
            isolation: cfg.restart_isolation,
            front_p: if cfg.front_parents { FRONT_PARENT_P } else { 0.0 },
            stagnant,
            rule: state.map_or(RestartRule::ArchiveStall, RestartRule::IslandStall),
            restarts_total: 0,
            explore_from: cfg.islands.max(1).saturating_sub(cfg.explore_islands),
            fronts: (cfg.front_scope == FrontScope::Island)
                .then(|| vec![Archive::new(ISLAND_FRONT_CAP); cfg.islands.max(1)]),
            migrants: Migrants::default(),
        }
    }

    /// #1070: the front `island` draws its front parents from this generation:
    /// its own under `--front-scope island`, else `global`.
    pub(crate) fn front<'a>(&'a self, island: usize, global: &'a [Sol]) -> &'a [Sol] {
        match &self.fronts {
            Some(fronts) => &fronts[island].items,
            None => global,
        }
    }

    /// #1070: the island front for the checkpoint; empty under
    /// `--front-scope global`.
    pub(crate) fn island_front(&self, island: usize) -> &[Sol] {
        self.fronts.as_ref().map_or(&[], |fronts| &fronts[island].items)
    }

    /// #1070: offer every member of every island to its own front, after the
    /// archive update. Nothing happens under `--front-scope global`.
    pub(crate) fn offer(&mut self, population: &(impl IslandMembers + ?Sized), keep: &dyn Fn(&Node) -> bool) {
        if let Some(fronts) = self.fronts.as_mut() {
            for (island, front) in fronts.iter_mut().enumerate().take(population.island_count()) {
                population.offer(island, front, keep);
            }
        }
    }

    /// #1070: each move's destination front becomes the canonical front of
    /// its own and the source's members, both read before the first change.
    fn exchange_fronts(&mut self, moves: &[Move]) {
        if let Some(fronts) = self.fronts.as_mut() {
            let before = fronts.clone();
            for &Move { from, to } in moves {
                let union: Vec<Sol> = before[to].items.iter().chain(&before[from].items).cloned().collect();
                fronts[to] = canonical_front(before[to].cap, union);
            }
        }
    }

    /// #1100: whether `island` belongs to the explore tier.
    fn explores(&self, island: usize) -> bool {
        island >= self.explore_from
    }

    fn isolated(&self, island: usize) -> bool {
        self.explores(island)
            || match &self.rule {
                RestartRule::ArchiveStall => false,
                RestartRule::IslandStall(s) => s.is_isolated(island),
            }
    }

    /// Where `island` draws its parents this generation. An isolated island
    /// and an explore island breed from their own members only (#1022, #1067,
    /// #1100).
    pub(crate) fn parent_source(&self, island: usize) -> ParentSource {
        if self.explores(island) {
            return ParentSource { front_p: 0.0, shared: false };
        }
        match &self.rule {
            RestartRule::ArchiveStall => ParentSource { front_p: self.front_p, shared: true },
            RestartRule::IslandStall(s) => ParentSource {
                front_p: s.front_p(island, self.front_p),
                shared: !s.is_isolated(island),
            },
        }
    }

    /// One generation's stall update, after the archive update.
    pub(crate) fn observe(&mut self, population: &(impl IslandBests + ?Sized)) {
        self.migrants = Migrants::default();
        if let RestartRule::IslandStall(s) = &mut self.rule {
            let bests: Vec<f64> = (0..population.island_count()).map(|i| population.best_error(i)).collect();
            s.tick(&bests);
        }
    }

    /// The generation barrier: migration every `--migration-interval`
    /// generations, then restarts every `--restart-interval` generations,
    /// decided from the island bests after migration and the best error of
    /// `archive`. Returns the restarted islands in island order.
    pub(crate) fn barrier(&mut self, gen: usize, archive: &[Sol], population: &mut impl IslandPopulation) -> Vec<usize> {
        let islands = population.island_count();
        if islands > 1 && due(self.migration_interval, gen) {
            let moves = self.ring(islands);
            self.exchange_fronts(&moves);
            self.migrants = population.migrate(&moves, self.migration_policy);
        }
        if !due(self.restart_interval, gen) {
            return Vec::new();
        }
        let archive_best = archive.iter().map(|s| s.error).fold(f64::INFINITY, f64::min);
        let bests: Vec<f64> = (0..islands).map(|i| population.best_error(i)).collect();
        let restarts = self.restart_set(gen, &bests, archive_best);
        for &island in &restarts {
            population.restart(island);
            if let Some(fronts) = self.fronts.as_mut() {
                fronts[island].items.clear();
            }
        }
        restarts
    }

    /// #1029: the migrants the last barrier copied and skipped; zero from the
    /// next observation on.
    pub(crate) fn migrants(&self) -> Migrants {
        self.migrants
    }

    /// The directed ring of the exploit tier: every exploit island that is not
    /// isolated receives the best member of the next exploit island.
    fn ring(&self, islands: usize) -> Vec<Move> {
        let exploit = islands.min(self.explore_from);
        if exploit < 2 {
            return Vec::new();
        }
        (0..exploit)
            .filter(|&to| !self.isolated(to))
            .map(|to| Move { from: (to + 1) % exploit, to })
            .collect()
    }

    /// The islands to restart at the restart check of generation `gen`, in
    /// island order, with their counters updated. An explore island restarts
    /// exactly at the end of its lifetime; the restart rule decides the rest.
    fn restart_set(&mut self, gen: usize, bests: &[f64], archive_best: f64) -> Vec<usize> {
        let keeper = match &self.rule {
            RestartRule::ArchiveStall => None,
            RestartRule::IslandStall(_) => RestartState::keeper(bests, archive_best),
        };
        let explore_from = self.explore_from.min(bests.len());
        let units = ((gen + 1) / self.restart_interval.max(1)) as u64;
        let lifetime_ends: Vec<usize> =
            explore_restarts(bests.len() - explore_from, units).into_iter().map(|slot| explore_from + slot).collect();
        let mut set = Vec::new();
        for (island, &best) in bests.iter().enumerate() {
            let explore = self.explores(island);
            let restart = if explore {
                lifetime_ends.contains(&island)
            } else {
                match &self.rule {
                    RestartRule::IslandStall(s) => s.due(island, keeper, self.restart_interval),
                    RestartRule::ArchiveStall => {
                        if best > archive_best + 1e-9 {
                            self.stagnant[island] += 1;
                        } else {
                            self.stagnant[island] = 0;
                        }
                        self.stagnant[island] >= 2
                    }
                }
            };
            if restart {
                self.stagnant[island] = 0;
                // #1100: an explore island is isolated by its tier, not by a
                // window, so `isolated` keeps counting exploit windows only.
                let window = if explore { 0 } else { self.isolation };
                if let RestartRule::IslandStall(s) = &mut self.rule {
                    s.restarted(island, window);
                }
                self.restarts_total += 1;
                set.push(island);
            }
        }
        set
    }

    /// The archive-stall checks for the checkpoint's `stagnant` record.
    pub(crate) fn stagnant(&self) -> &[usize] {
        &self.stagnant
    }

    /// The `island_stall` record of `island`; `None` under the archive rule.
    pub(crate) fn stall(&self, island: usize) -> Option<IslandStall> {
        match &self.rule {
            RestartRule::ArchiveStall => None,
            RestartRule::IslandStall(s) => Some(s.saved(island)),
        }
    }

    /// Islands inside their isolation window.
    pub(crate) fn isolated_count(&self) -> usize {
        match &self.rule {
            RestartRule::ArchiveStall => 0,
            RestartRule::IslandStall(s) => s.isolated_count(),
        }
    }

    pub(crate) fn restarts_total(&self) -> u64 {
        self.restarts_total
    }
}

impl IslandBests for [Vec<Ind>] {
    fn island_count(&self) -> usize {
        self.len()
    }

    fn best_error(&self, island: usize) -> f64 {
        population_best(&self[island])
    }
}

impl IslandMembers for [Vec<Ind>] {
    fn offer(&self, island: usize, front: &mut Archive, keep: &dyn Fn(&Node) -> bool) {
        for ind in &self[island] {
            if front.admits(ind.error, ind.complexity) && !keep(&ind.tree) {
                continue;
            }
            front.try_insert_view(&ind.tree, ind.error, expr::complexity(&ind.tree), ind.complexity);
        }
    }
}

/// The owned-tree population of `run_search`, with what its migration and
/// restart need from the generation.
pub(super) struct TreeIslands<'a> {
    pub islands: &'a mut [Vec<Ind>],
    pub rngs: &'a mut [Mt],
    pub cfg: &'a Config,
    pub nvars: usize,
    pub aggs: &'a AggTable,
    /// This generation's scoring rows and their column view.
    pub gen_train: &'a Sample,
    pub gen_data: &'a EvalSample<'a>,
    /// This generation's constant-fitting rows (#85).
    pub fit_train: &'a Sample,
    /// Migrants and restarted trees wait for the next device batch (#26).
    pub defer_gpu: bool,
    pub qd: Option<&'a QdArchive>,
    /// Whether a member's cached descriptor describes `gen_train` (#120).
    pub cached_descriptors_valid: bool,
    /// #1117: where the restart fill counts its fit decisions.
    pub fit_counts: Option<&'a FitAcceptCounts>,
}

impl IslandBests for TreeIslands<'_> {
    fn island_count(&self) -> usize {
        self.islands.len()
    }

    fn best_error(&self, island: usize) -> f64 {
        population_best(&self.islands[island])
    }
}

impl TreeIslands<'_> {
    /// #120: the least-visited descriptor cell under `--qd-policy`.
    fn qd_target(&self) -> Option<qd::Descriptor> {
        if self.cfg.qd_policy {
            self.qd.and_then(|q| q.least_occupied_descriptor())
        } else {
            None
        }
    }

    /// The member of `island` a migrant replaces.
    fn victim(&self, island: usize) -> Option<usize> {
        // #120: stratified migration replaces the individual in the
        // least-visited descriptor cell, not the globally worst error.
        let victim = self.qd_target().and_then(|target| {
            // #120: reuse the descriptors the scoring pass already accumulated
            // when they describe `gen_train`; only the GPU or subset paths pay
            // for one here.
            let descriptors: Vec<qd::Descriptor> = self.islands[island]
                .iter()
                .map(|ind| {
                    if self.cached_descriptors_valid {
                        if let Some(d) = ind.descriptor {
                            return d;
                        }
                    }
                    qd_descriptor_of(&ind.tree, self.gen_data, self.cfg) // #1084
                })
                .collect();
            descriptors
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let score = |d: &qd::Descriptor| (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2);
                    score(a).partial_cmp(&score(b)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
        });
        victim.or_else(|| {
            if self.cfg.selection == Selection::Afp {
                return afp_worst(&self.islands[island]);
            }
            self.islands[island]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| crate::error_order::cmp_error(a.error, b.error))
                .map(|(i, _)| i)
        })
    }
}

/// #1029: whether `island` already holds the migrant's book. Equal finite error
/// bits stand for the same book; #1082's `Ind.book` replaces the comparison
/// here.
fn holds_book(island: &[Ind], migrant: &Ind) -> bool {
    migrant.error.is_finite() && island.iter().any(|m| m.error.to_bits() == migrant.error.to_bits())
}

impl IslandPopulation for TreeIslands<'_> {
    fn migrate(&mut self, moves: &[Move], policy: MigrationPolicy) -> Migrants {
        fn best(pop: &[Ind]) -> Option<&Ind> {
            pop.iter().min_by(|a, b| crate::error_order::cmp_error(a.error, b.error))
        }
        // Every migrant is chosen before the first copy. `None` sends nothing;
        // `skip` marks a migrant the destination already holds.
        struct Pick {
            tree: Node,
            age: u32,
            skip: bool,
        }
        let picks: Vec<Option<Pick>> = moves
            .iter()
            .map(|&Move { from, to }| match policy {
                MigrationPolicy::Ring => {
                    best(&self.islands[from]).map(|i| Pick { tree: i.tree.clone(), age: i.age, skip: false })
                }
                MigrationPolicy::RingTournament => {
                    // The destination's generator draws, so the source island's
                    // own stream does not depend on who migrates from it.
                    if !self.islands[from].iter().any(|i| i.error.is_finite()) {
                        return None;
                    }
                    let winner = &self.islands[from][tournament(&self.islands[from], &mut self.rngs[to], MIGRANT_TOURNAMENT)];
                    Some(Pick {
                        tree: winner.tree.clone(),
                        age: winner.age,
                        skip: holds_book(&self.islands[to], winner),
                    })
                }
            })
            .collect();
        let mut count = Migrants::default();
        for (&Move { to, .. }, pick) in moves.iter().zip(&picks) {
            let Some(pick) = pick else { continue };
            if pick.skip {
                count.skipped += 1;
                continue;
            }
            let slot = self.victim(to);
            if let Some(worst) = slot.and_then(|i| self.islands[to].get_mut(i)) {
                // #26: batch the migrant with the next GPU call.
                *worst = if self.defer_gpu {
                    Ind::pending(pick.tree.clone(), self.cfg)
                } else {
                    Ind::evaluated(pick.tree.clone(), self.gen_data, self.cfg) // #25
                };
                worst.age = pick.age;
                count.sent += 1;
            }
        }
        count
    }

    fn restart(&mut self, island: usize) {
        let (cfg, nvars, aggs, gen_train, gen_data) = (self.cfg, self.nvars, self.aggs, self.gen_train, self.gen_data);
        let mut r = self.rngs[island].clone();
        // #120: aim the restart at the least-visited region by rejection
        // sampling a few cheap descriptor probes.
        let qd_target = self.qd_target();
        let mut have_const = false; // #477
        for ind in self.islands[island].iter_mut() {
            let hc = have_const; // #477
            let mut t = draw_row_dependent(cfg, nvars, &mut have_const, || {
                let mut t = random_tree_with(&mut r, nvars, &cfg.ops, cfg.init_depth, Some(aggs));
                if let Some(target) = qd_target {
                    let mut best = t.clone();
                    let best_d = qd_descriptor_of(&t, gen_data, cfg); // #1084
                    let mut best_score = if hc && !expr::row_dependent(&t) {
                        f64::INFINITY
                    } else {
                        (best_d.usage - target.usage).powi(2) + (best_d.skew - target.skew).powi(2)
                    };
                    for _ in 0..3 {
                        let cand = random_tree_with(&mut r, nvars, &cfg.ops, cfg.init_depth, Some(aggs));
                        let d = qd_descriptor_of(&cand, gen_data, cfg); // #1084
                        let score = (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2);
                        if score < best_score && !(hc && !expr::row_dependent(&cand)) {
                            best = cand;
                            best_score = score;
                        }
                    }
                    t = best;
                }
                t
            });
            // #15: multi-output fits per output inside the evaluator.
            let mut fitted = None;
            let mut before = None; // #1078
            if cfg.fit_iters > 0 && !gen_train.is_multi() && !defers_fit_to_gpu(cfg, self.defer_gpu) {
                before = pre_fit_tree(cfg, &t);
                fitted = fit_constants(&mut t, cfg, self.fit_train, gen_train); // #85 #87
            }
            // #26: restart trees are batched with the next GPU call too, before
            // the checkpoint is taken.
            *ind = if self.defer_gpu && before.is_none() {
                Ind::pending(t, cfg)
            } else {
                gate_fit(
                    None,
                    self.fit_counts,
                    FitPath::Restart,
                    before,
                    Ind::evaluated_fitted(t, fitted, self.gen_data, cfg),
                    self.gen_data,
                    cfg,
                ) // #87 #1078
            };
        }
        self.rngs[island] = r;
    }
}

#[cfg(feature = "cuda")]
impl IslandBests for Arena {
    fn island_count(&self) -> usize {
        self.islands
    }

    fn best_error(&self, island: usize) -> f64 {
        (0..self.pop).map(|j| self.err[self.slot_index(island, j)]).fold(f64::INFINITY, f64::min)
    }
}

#[cfg(feature = "cuda")]
impl IslandMembers for Arena {
    fn offer(&self, island: usize, front: &mut Archive, keep: &dyn Fn(&Node) -> bool) {
        for j in 0..self.pop {
            let idx = self.slot_index(island, j);
            let (error, cost) = (self.err[idx], self.cost[idx] as f64);
            if !front.admits(error, cost) {
                continue;
            }
            let Some(tree) = self.decode(idx) else { continue };
            if keep(&tree) {
                front.try_insert_view(&tree, error, expr::complexity(&tree), cost);
            }
        }
    }
}

/// The flat postfix population of `run_postfix_search` in generation `gen`.
#[cfg(feature = "cuda")]
pub(super) struct ArenaIslands<'a> {
    pub arena: &'a mut Arena,
    pub variation: &'a Variation,
    pub gen: u32,
}

#[cfg(feature = "cuda")]
impl IslandBests for ArenaIslands<'_> {
    fn island_count(&self) -> usize {
        self.arena.island_count()
    }

    fn best_error(&self, island: usize) -> f64 {
        self.arena.best_error(island)
    }
}

#[cfg(feature = "cuda")]
impl IslandPopulation for ArenaIslands<'_> {
    fn migrate(&mut self, moves: &[Move], policy: MigrationPolicy) -> Migrants {
        let moves: Vec<(usize, usize)> = moves.iter().map(|m| (m.from, m.to)).collect();
        let (sent, skipped) = self.arena.migrate_with(&moves, policy, self.variation.seed, self.gen);
        Migrants { sent, skipped }
    }

    fn restart(&mut self, island: usize) {
        self.variation.restart_island(self.arena, island, self.gen);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{pick_parent_aged, run_resumable, Outcome, Parents};
    use super::*;
    use crate::checkpoint::Checkpoint;
    use crate::expr::{constant, input, node, Op};

    fn sol(error: f64) -> Sol {
        Sol { tree: input(0), error, complexity: 1, cost: 1.0 }
    }

    /// A population of island bests that records what the policy asks of it.
    #[derive(Default)]
    struct Bests {
        best: Vec<f64>,
        moves: Vec<Move>,
        restarts: Vec<usize>,
    }

    impl IslandBests for Bests {
        fn island_count(&self) -> usize {
            self.best.len()
        }

        fn best_error(&self, island: usize) -> f64 {
            self.best[island]
        }
    }

    impl IslandPopulation for Bests {
        fn migrate(&mut self, moves: &[Move], _policy: MigrationPolicy) -> Migrants {
            self.moves.extend_from_slice(moves);
            Migrants::default()
        }

        fn restart(&mut self, island: usize) {
            self.restarts.push(island);
        }
    }

    /// Scored island members for the island-front tests; records moves and
    /// restarts like [`Bests`].
    #[derive(Default)]
    struct Members {
        islands: Vec<Vec<Sol>>,
        moves: Vec<Move>,
        restarts: Vec<usize>,
    }

    impl IslandBests for Members {
        fn island_count(&self) -> usize {
            self.islands.len()
        }

        fn best_error(&self, island: usize) -> f64 {
            self.islands[island].iter().map(|s| s.error).fold(f64::INFINITY, f64::min)
        }
    }

    impl IslandMembers for Members {
        fn offer(&self, island: usize, front: &mut Archive, keep: &dyn Fn(&Node) -> bool) {
            for s in &self.islands[island] {
                if front.admits(s.error, s.cost) && !keep(&s.tree) {
                    continue;
                }
                front.try_insert_view(&s.tree, s.error, expr::complexity(&s.tree), s.cost);
            }
        }
    }

    impl IslandPopulation for Members {
        fn migrate(&mut self, moves: &[Move], _policy: MigrationPolicy) -> Migrants {
            self.moves.extend_from_slice(moves);
            Migrants::default()
        }

        fn restart(&mut self, island: usize) {
            self.restarts.push(island);
        }
    }

    fn policy_cfg(islands: usize, rule: RestartPolicy, migration: usize, restart: usize, isolation: usize) -> Config {
        Config {
            islands,
            restart_policy: rule,
            migration_interval: migration,
            restart_interval: restart,
            restart_isolation: isolation,
            ..Default::default()
        }
    }

    const OPEN: ParentSource = ParentSource { front_p: FRONT_PARENT_P, shared: true };
    const ISOLATED: ParentSource = ParentSource { front_p: 0.0, shared: false };

    #[test]
    fn an_isolated_island_receives_no_migrant_and_still_sends_one() {
        let mut policy = IslandPolicy::new(&policy_cfg(3, RestartPolicy::Island, 1, 2, 3));
        let archive = [sol(0.1)];
        let mut pop = Bests { best: vec![0.1, 0.5, 0.9], ..Default::default() };
        let ring = vec![Move { from: 1, to: 0 }, Move { from: 2, to: 1 }, Move { from: 0, to: 2 }];
        // Island 0 keeps the archive best, island 1 stalls, island 2 improves.
        for gen in 0..4 {
            policy.observe(&pop);
            pop.moves.clear();
            let restarts = policy.barrier(gen, &archive, &mut pop);
            assert_eq!(pop.moves, ring, "generation {gen}: nobody is isolated yet");
            assert_eq!(restarts, if gen == 3 { vec![1] } else { vec![] }, "generation {gen}");
            pop.best[2] -= 0.1;
        }
        assert_eq!(pop.restarts, vec![1]);
        assert_eq!((policy.parent_source(0), policy.parent_source(1)), (OPEN, ISOLATED));
        assert_eq!(policy.isolated_count(), 1);
        for gen in 4..6 {
            policy.observe(&pop);
            pop.moves.clear();
            policy.barrier(gen, &archive, &mut pop);
            assert_eq!(pop.moves, vec![Move { from: 1, to: 0 }, Move { from: 0, to: 2 }], "generation {gen}");
            assert_eq!(policy.parent_source(1), ISOLATED, "generation {gen}");
        }
        policy.observe(&pop);
        assert_eq!(policy.parent_source(1), OPEN, "the window of 3 generations has closed");
        pop.moves.clear();
        policy.barrier(6, &archive, &mut pop);
        assert_eq!(pop.moves, ring);
    }

    #[test]
    fn the_keeper_is_never_restarted() {
        let mut policy = IslandPolicy::new(&policy_cfg(4, RestartPolicy::Island, 0, 1, 0));
        // Islands 1 and 2 hold the archive best; the lowest of them keeps.
        let mut pop = Bests { best: vec![0.3, 0.1, 0.1, 0.2], ..Default::default() };
        for gen in 0..20 {
            policy.observe(&pop);
            policy.barrier(gen, &[sol(0.1)], &mut pop);
        }
        assert!(pop.moves.is_empty(), "migration is off");
        assert!(!pop.restarts.contains(&1), "the keeper restarted: {:?}", pop.restarts);
        for island in [0, 2, 3] {
            assert!(pop.restarts.contains(&island), "island {island} stalls and restarts: {:?}", pop.restarts);
        }
        assert_eq!(policy.restarts_total(), pop.restarts.len() as u64);
    }

    #[test]
    fn one_island_receives_no_migrant_and_holds_the_archive_best() {
        for rule in [RestartPolicy::Archive, RestartPolicy::Island] {
            let mut policy = IslandPolicy::new(&policy_cfg(1, rule, 1, 1, 5));
            let mut pop = Bests { best: vec![0.5], ..Default::default() };
            for gen in 0..10 {
                policy.observe(&pop);
                policy.barrier(gen, &[sol(0.5)], &mut pop);
            }
            assert!(pop.moves.is_empty() && pop.restarts.is_empty(), "{rule:?}: {:?} {:?}", pop.moves, pop.restarts);
        }
    }

    #[test]
    fn no_front_parents_keeps_the_shared_pools() {
        for rule in [RestartPolicy::Archive, RestartPolicy::Island] {
            let cfg = Config { front_parents: false, ..policy_cfg(2, rule, 25, 60, 25) };
            let policy = IslandPolicy::new(&cfg);
            assert_eq!(policy.parent_source(1), ParentSource { front_p: 0.0, shared: true }, "{rule:?}");
        }
    }

    #[test]
    fn the_luby_list_schedule_is_pinned() {
        let terms: Vec<u64> = (1..=31).map(luby).collect();
        assert_eq!(terms, [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 16]);
        let mut class = [0u64; 7];
        for n in 1..=63 {
            class[luby(n).trailing_zeros() as usize] += luby(n);
        }
        assert_eq!(class, [32, 32, 32, 32, 32, 32, 0], "every length class gets the same compute");

        let ends = |k: usize, upto: u64| -> Vec<Vec<u64>> {
            let mut out = vec![Vec::new(); k];
            for units in 1..=upto {
                for slot in explore_restarts(k, units) {
                    out[slot].push(units);
                }
            }
            out
        };
        let prefix: Vec<u64> = (1..=15).map(luby).scan(0, |s, t| {
            *s += t;
            Some(*s)
        }).collect();
        assert_eq!(ends(1, 32), [prefix], "one slot runs the sequence itself");
        let six = ends(6, 7);
        let want: [&[u64]; 6] = [&[1, 5, 7], &[1, 2, 3], &[2, 3, 4, 6, 7], &[1, 2, 4, 5], &[1, 3, 4, 5, 6], &[2, 6, 7]];
        for (slot, want) in want.iter().enumerate() {
            assert_eq!(six[slot], *want, "slot {slot}");
        }
        assert_eq!(ends(6, 11)[1], [1, 2, 3, 11], "slot 1 then runs 8 units");
        assert_eq!(ends(6, 9)[3], [1, 2, 4, 5, 9]);
        assert!(explore_restarts(0, 5).is_empty() && explore_restarts(3, 0).is_empty());
    }

    #[test]
    fn the_migration_interval_sets_the_migration_generations() {
        let archive = [sol(0.1)];
        for (interval, want) in [(4usize, (3..36).step_by(4).collect::<Vec<usize>>()), (1, (0..36).collect()), (0, Vec::new())] {
            let mut policy = IslandPolicy::new(&policy_cfg(3, RestartPolicy::Archive, interval, 0, 0));
            let mut pop = Bests { best: vec![0.1, 0.5, 0.9], ..Default::default() };
            let mut at = Vec::new();
            for gen in 0..36 {
                policy.observe(&pop);
                pop.moves.clear();
                policy.barrier(gen, &archive, &mut pop);
                if !pop.moves.is_empty() {
                    at.push(gen);
                }
            }
            assert_eq!(at, want, "interval {interval}");
        }
        let mut policy = IslandPolicy::new(&policy_cfg(1, RestartPolicy::Archive, 4, 0, 0));
        let mut pop = Bests { best: vec![0.1], ..Default::default() };
        for gen in 0..36 {
            policy.observe(&pop);
            policy.barrier(gen, &archive, &mut pop);
        }
        assert!(pop.moves.is_empty(), "one island never migrates");
    }

    #[test]
    fn the_restart_interval_sets_the_explore_lifetime_unit() {
        let (n, k, unit) = (4usize, 2usize, 3usize);
        let cfg = Config { explore_islands: k, ..policy_cfg(n, RestartPolicy::Archive, 0, unit, 0) };
        let mut policy = IslandPolicy::new(&cfg);
        let mut pop = Bests { best: vec![0.1, 0.5, 0.7, 0.9], ..Default::default() };
        let archive = [sol(0.1)];
        let mut got: Vec<(usize, usize)> = Vec::new();
        for gen in 0..36 {
            policy.observe(&pop);
            for island in policy.barrier(gen, &archive, &mut pop) {
                if island >= n - k {
                    got.push((gen, island));
                }
            }
        }
        let mut want: Vec<(usize, usize)> = Vec::new();
        for b in 1..=12u64 {
            for slot in explore_restarts(k, b) {
                want.push((unit * b as usize - 1, n - k + slot));
            }
        }
        assert_eq!(got, want);
        assert!(got.len() >= 12, "{got:?}");
        assert!(got.iter().all(|&(gen, _)| (gen + 1) % unit == 0), "{got:?}");
    }

    #[test]
    fn the_restart_interval_sets_the_island_stall_length() {
        for (unit, want) in [(3usize, vec![5usize, 11, 17, 23]), (4, vec![7, 15, 23])] {
            let cfg = policy_cfg(3, RestartPolicy::Island, 0, unit, 0);
            let mut policy = IslandPolicy::new(&cfg);
            let mut pop = Bests { best: vec![0.1, 0.5, 0.5], ..Default::default() };
            let archive = [sol(0.1)];
            let mut at = Vec::new();
            for gen in 0..24 {
                policy.observe(&pop);
                let restarts = policy.barrier(gen, &archive, &mut pop);
                if !restarts.is_empty() {
                    assert_eq!(restarts, [1, 2], "interval {unit} generation {gen}: the keeper is spared");
                    at.push(gen);
                }
            }
            assert_eq!(at, want, "interval {unit}");
        }
        let mut policy = IslandPolicy::new(&policy_cfg(3, RestartPolicy::Island, 0, 0, 0));
        let mut pop = Bests { best: vec![0.1, 0.5, 0.5], ..Default::default() };
        for gen in 0..24 {
            policy.observe(&pop);
            assert!(policy.barrier(gen, &[sol(0.1)], &mut pop).is_empty(), "interval 0 restarts nothing");
        }
    }

    #[test]
    fn an_explore_island_takes_no_front_parent_and_exchanges_no_migrant() {
        let (n, k, unit) = (5usize, 2usize, 2usize);
        for rule in [RestartPolicy::Archive, RestartPolicy::Island] {
            let cfg = Config { explore_islands: k, ..policy_cfg(n, rule, 1, unit, 3) };
            let mut policy = IslandPolicy::new(&cfg);
            // Island 0 holds the archive best; the others stall at worse errors.
            let mut pop = Bests { best: vec![0.1, 0.5, 0.7, 0.9, 0.3], ..Default::default() };
            let archive = [sol(0.1)];
            let mut explore_seen = 0;
            for gen in 0..48 {
                for island in n - k..n {
                    assert_eq!(policy.parent_source(island), ISOLATED, "{rule:?} generation {gen} island {island}");
                }
                policy.observe(&pop);
                pop.moves.clear();
                let restarts = policy.barrier(gen, &archive, &mut pop);
                let ring = [Move { from: 1, to: 0 }, Move { from: 2, to: 1 }, Move { from: 0, to: 2 }];
                assert!(pop.moves.iter().all(|m| ring.contains(m)), "{rule:?} generation {gen}: {:?}", pop.moves);
                if gen == 0 {
                    assert_eq!(pop.moves, ring, "{rule:?}: the exploit ring");
                }
                let want: Vec<usize> = if (gen + 1) % unit == 0 {
                    explore_restarts(k, ((gen + 1) / unit) as u64).into_iter().map(|s| n - k + s).collect()
                } else {
                    Vec::new()
                };
                let got: Vec<usize> = restarts.iter().copied().filter(|&i| i >= n - k).collect();
                assert_eq!(got, want, "{rule:?} generation {gen}: explore restarts");
                explore_seen += got.len();
                assert_eq!(&policy.stagnant()[n - k..], &[0, 0], "{rule:?} generation {gen}");
            }
            assert!(explore_seen >= 10, "{rule:?}: {explore_seen} explore restarts");
            assert!(pop.restarts.iter().any(|&i| i < n - k), "{rule:?}: the exploit rule still restarts");
        }
    }

    #[test]
    fn one_exploit_island_has_no_migration() {
        let cfg = Config { explore_islands: 2, ..policy_cfg(3, RestartPolicy::Archive, 1, 1, 0) };
        let mut policy = IslandPolicy::new(&cfg);
        let mut pop = Bests { best: vec![0.2, 0.1, 0.3], ..Default::default() };
        for gen in 0..12 {
            policy.observe(&pop);
            policy.barrier(gen, &[sol(0.1)], &mut pop);
        }
        assert!(pop.moves.is_empty(), "{:?}", pop.moves);
        assert_eq!(policy.parent_source(0), OPEN);
        assert!(pop.restarts.contains(&1) && pop.restarts.contains(&2));
    }

    /// Member `k` of `island`: `x0` plus `k` constants unique to the island.
    fn member(island: usize, k: usize, error: f64) -> Sol {
        let mut tree = input(0);
        for j in 0..k {
            tree = node(Op::Add, vec![tree, constant((island * 100 + j) as f64 + 0.5)]);
        }
        let complexity = expr::complexity(&tree);
        Sol { tree, error, complexity, cost: complexity as f64 }
    }

    fn front_view(front: &[Sol]) -> Vec<(u64, u64, usize)> {
        front.iter().map(|s| (expr::structural_hash(&s.tree), s.error.to_bits(), s.complexity)).collect()
    }

    #[test]
    fn island_fronts_follow_the_migration_moves() {
        let cfg = Config { front_scope: FrontScope::Island, explore_islands: 1, ..policy_cfg(4, RestartPolicy::Archive, 2, 3, 0) };
        let global = [sol(0.0)];
        let mut pop = Members {
            islands: (0..4).map(|i| (0..3).map(|k| member(i, k, 1.0 / (k + 1) as f64 + 0.01 * i as f64)).collect()).collect(),
            ..Default::default()
        };
        let mut blocked = IslandPolicy::new(&cfg);
        blocked.offer(&pop, &|_| false);
        assert!((0..4).all(|i| blocked.island_front(i).is_empty()), "a tree the filter refuses never enters");

        let mut policy = IslandPolicy::new(&cfg);
        let ring = [Move { from: 1, to: 0 }, Move { from: 2, to: 1 }, Move { from: 0, to: 2 }];
        let (mut exchanges, mut emptied) = (0, 0);
        for gen in 0..12 {
            for (i, island) in pop.islands.iter_mut().enumerate() {
                island.push(member(i, 3 + gen, 0.3 / (gen + 2) as f64 + 0.01 * i as f64));
            }
            let before: Vec<Vec<Sol>> = (0..4).map(|i| policy.island_front(i).to_vec()).collect();
            policy.offer(&pop, &|_| true);
            for i in 0..4 {
                let own: std::collections::HashSet<u64> =
                    pop.islands[i].iter().chain(&before[i]).map(|s| expr::structural_hash(&s.tree)).collect();
                let front = policy.island_front(i);
                assert!(!front.is_empty(), "generation {gen}: island {i}");
                assert!(front.iter().all(|s| own.contains(&expr::structural_hash(&s.tree))), "generation {gen}: island {i} took foreign material");
                assert_eq!(policy.front(i, &global).as_ptr(), front.as_ptr(), "island {i} draws from its own front");
            }
            let offered: Vec<Vec<Sol>> = (0..4).map(|i| policy.island_front(i).to_vec()).collect();
            policy.observe(&pop);
            pop.moves.clear();
            pop.restarts.clear();
            policy.barrier(gen, &global, &mut pop);
            let moved: Vec<usize> = pop.moves.iter().map(|m| m.to).collect();
            if (gen + 1) % 2 == 0 {
                assert_eq!(pop.moves, ring, "generation {gen}: the explore island takes no front");
                exchanges += 1;
            } else {
                assert!(pop.moves.is_empty());
            }
            for i in 0..4 {
                let got = front_view(policy.island_front(i));
                if pop.restarts.contains(&i) {
                    assert!(got.is_empty(), "generation {gen}: a restart empties island {i}'s front");
                    emptied += 1;
                } else if let Some(m) = pop.moves.iter().find(|m| m.to == i) {
                    let union: Vec<Sol> = offered[m.to].iter().chain(&offered[m.from]).cloned().collect();
                    assert_eq!(got, front_view(&canonical_front(ISLAND_FRONT_CAP, union).items), "generation {gen}: island {i}");
                } else {
                    assert!(!moved.contains(&i));
                    assert_eq!(got, front_view(&offered[i]), "generation {gen}: island {i} keeps its front");
                }
            }
        }
        assert!(exchanges >= 5 && emptied >= 2, "{exchanges} exchanges, {emptied} emptied fronts");
    }

    #[test]
    fn the_global_scope_hands_every_island_the_global_front() {
        let mut policy = IslandPolicy::new(&policy_cfg(3, RestartPolicy::Archive, 1, 1, 0));
        let global = [sol(0.1), sol(0.2)];
        let mut pop = Members { islands: (0..3).map(|i| vec![member(i, 1, 0.5)]).collect(), ..Default::default() };
        policy.offer(&pop, &|_| true);
        policy.barrier(0, &global, &mut pop);
        for i in 0..3 {
            assert_eq!(policy.front(i, &global).as_ptr(), global.as_ptr(), "island {i}");
            assert!(policy.island_front(i).is_empty(), "island {i}");
        }
    }

    #[test]
    fn restart_state_counts_the_islands_own_stall_and_spares_the_keeper() {
        let mut s = RestartState::new(3);
        s.tick(&[1.0, 1.0, 2.0]);
        assert_eq!(s.since_improve, vec![0, 0, 0], "the first finite best is an improvement");
        s.tick(&[1.0, 0.5, 2.0]);
        assert_eq!(s.since_improve, vec![1, 0, 1]);
        s.tick(&[1.0 - 1e-12, 0.5, f64::INFINITY]);
        assert_eq!(s.since_improve, vec![2, 1, 2], "a 1e-12 gain and a non-finite best are stalls");
        assert_eq!(s.best_prev[2], 2.0, "a non-finite best never lowers best_prev");

        assert_eq!(RestartState::keeper(&[1.0, 0.5, 0.5], 0.5), Some(1), "the lowest holder keeps");
        assert_eq!(RestartState::keeper(&[1.0, 0.7], 0.5), None);
        assert_eq!(RestartState::keeper(&[0.5], 0.5), Some(0), "one island is its own keeper");
        assert!(s.due(0, Some(1), 2));
        assert!(!s.due(1, Some(1), 1), "the keeper is never due");
        assert!(s.due(1, None, 1));
        assert!(!s.due(1, None, 2), "not stalled for a whole interval yet");
    }

    #[test]
    fn restart_state_isolation_zeroes_front_parents_and_counts_down() {
        let mut s = RestartState::new(2);
        s.tick(&[1.0, 1.0]);
        s.restarted(1, 3);
        assert_eq!(s.best_prev[1], f64::INFINITY);
        assert_eq!(s.since_improve[1], 0);
        assert_eq!((s.front_p(0, 0.15), s.front_p(1, 0.15)), (0.15, 0.0));
        assert_eq!(s.isolated_count(), 1);
        for left in [2usize, 1, 0] {
            s.tick(&[1.0, 5.0]);
            assert_eq!(s.isolated[1], left);
            if left == 2 {
                assert_eq!(s.since_improve[1], 0, "the first generation after a restart is an improvement");
            }
        }
        assert!(!s.is_isolated(1));
        assert_eq!(s.front_p(1, 0.15), 0.15);
        s.restarted(0, 0);
        assert!(!s.is_isolated(0), "a window of 0 turns the isolation off");
    }

    #[test]
    fn restart_state_survives_a_checkpoint_round_trip_and_refuses_a_missing_record() {
        let mut s = RestartState::new(2);
        s.tick(&[2.0, 3.0]);
        s.tick(&[2.0, 2.5]);
        s.restarted(0, 7);
        let saved: Vec<_> = (0..2).map(|i| Some(s.saved(i))).collect();
        assert_eq!(RestartState::from_saved(&saved).unwrap(), s);
        let err = RestartState::from_saved(&[Some(s.saved(0)), None]).unwrap_err();
        assert!(err.contains("island_stall"), "{err}");
    }

    fn golden_sample(n: usize) -> Sample {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = (i as f64) / (n as f64) * 4.0 - 2.0;
            let b = (((i * 13) % n) as f64) / (n as f64) * 2.0 - 1.0;
            x.push(vec![a, b]);
            y.push(a.sin() + 0.5 * b * b);
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
    }

    fn golden_config() -> Config {
        Config {
            generations: 40,
            islands: 4,
            pop: 16,
            max_complexity: 10,
            fit_iters: 2,
            migration_interval: 6,
            restart_interval: 2,
            patience: 0,
            seed: 11,
            gpu: false,
            ..Default::default()
        }
    }

    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }

    /// The outcome, the exit-checkpoint text and the restarts of one run.
    fn checkpointed(cfg: &Config, resume: Option<Checkpoint>) -> (Outcome, String, u64) {
        let (train, test) = (golden_sample(240), golden_sample(80));
        let restarts = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let sink = restarts.clone();
        let cfg = Config {
            dev_log: Some(std::sync::Arc::new(move |r: &crate::devlog::GenRecord| {
                sink.store(r.restarts_total, std::sync::atomic::Ordering::SeqCst);
            })),
            ..cfg.clone()
        };
        let mut text = None;
        let out = run_resumable(&cfg, &train, &test, resume, 0, |cp| text = Some(cp.to_text())).expect("run");
        (out, text.expect("exit checkpoint"), restarts.load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Archive hash and exit-checkpoint hash (build lines removed) of the
    /// inline policy before this module, on `golden_sample`.
    const GOLDEN: [(&str, u64, u64); 4] = [
        ("archive", 0x47d4_6e43_d3b5_c903, 0x6aba_aec9_b267_1f93),
        ("island", 0x3d30_da98_8f09_6dca, 0x5ae0_2fdf_55aa_8c91),
        ("afp", 0x3ffe_5bb7_ec99_989f, 0x0342_9a75_4bea_2c7e),
        ("qd", 0xeca0_4df6_6525_9da7, 0x4a88_d806_da49_7cd2),
    ];

    fn golden_configs() -> [(&'static str, Config); 4] {
        [
            ("archive", golden_config()),
            ("island", Config { restart_policy: RestartPolicy::Island, restart_isolation: 7, ..golden_config() }),
            ("afp", Config { selection: Selection::Afp, ..golden_config() }),
            ("qd", Config { qd_bins: 8, qd_policy: true, ..golden_config() }),
        ]
    }

    #[test]
    fn default_trajectories_match_the_base() {
        let mut differs = Vec::new();
        for ((name, cfg), (want_name, want_archive, want_text)) in golden_configs().into_iter().zip(GOLDEN) {
            assert_eq!(name, want_name);
            let (out, text, restarts) = checkpointed(&cfg, None);
            assert!(restarts > 0, "{name}: the run restarts an island, else the pin proves less");
            // #1166: the qd_* records are new; every line the base wrote keeps its bytes.
            let kept = text
                .lines()
                .filter(|l| !l.starts_with("build ") && !l.starts_with("qd_"))
                .fold(0xcbf2_9ce4_8422_2325u64, |h, l| fnv(fnv(h, l.as_bytes()), b"\n"));
            if (out.archive.hash(), kept) != (want_archive, want_text) {
                differs.push(format!("(\"{name}\", {:#018x}, {kept:#018x})", out.archive.hash()));
            }
        }
        assert!(differs.is_empty(), "differs from the base: {}", differs.join(", "));
    }

    #[test]
    fn an_archive_policy_run_with_restarts_and_its_resume_match_exactly() {
        let cfg = |generations| Config { generations, ..golden_config() };
        let (uninterrupted, full_text, _) = checkpointed(&cfg(40), None);
        let (_, saved_text, saved_restarts) = checkpointed(&cfg(20), None);
        assert!(saved_restarts > 0, "an island restarts before the save, else the resume proves less");
        assert!(!saved_text.contains("island_stall"));
        let saved = Checkpoint::from_text(&saved_text).expect("parse saved checkpoint");
        let (resumed, resumed_text, _) = checkpointed(&cfg(40), Some(saved));
        assert_eq!(uninterrupted.archive.hash(), resumed.archive.hash(), "the archive");
        assert_eq!(full_text, resumed_text, "archive, populations, rng state and stagnant counters of the exit checkpoint");
    }

    #[test]
    fn an_explore_tier_run_and_its_resume_match_exactly() {
        for rule in [RestartPolicy::Archive, RestartPolicy::Island] {
            let cfg = |generations| Config {
                generations,
                explore_islands: 2,
                restart_policy: rule,
                restart_isolation: 7,
                ..golden_config()
            };
            let (uninterrupted, full_text, _) = checkpointed(&cfg(40), None);
            let (again, again_text, _) = checkpointed(&cfg(40), None);
            assert_eq!(uninterrupted.archive.hash(), again.archive.hash(), "{rule:?}: same seed, same archive");
            assert_eq!(full_text, again_text, "{rule:?}: same seed, same exit checkpoint");
            let (_, saved_text, saved_restarts) = checkpointed(&cfg(20), None);
            assert!(saved_restarts > 0, "{rule:?}: an island restarts before the save");
            assert!(saved_text.contains("config explore_islands 2\n"), "{rule:?}");
            let saved = Checkpoint::from_text(&saved_text).expect("parse saved checkpoint");
            let (resumed, resumed_text, resumed_restarts) = checkpointed(&cfg(40), Some(saved));
            assert!(resumed_restarts > 0, "{rule:?}: an island restarts after the save");
            assert_eq!(uninterrupted.archive.hash(), resumed.archive.hash(), "{rule:?}: the archive");
            assert_eq!(full_text, resumed_text, "{rule:?}: the exit checkpoint");
        }
    }

    fn refused(cfg: &Config, resume: Option<Checkpoint>) -> String {
        let (train, test) = (golden_sample(240), golden_sample(80));
        match run_resumable(cfg, &train, &test, resume, 0, |_| {}) {
            Ok(_) => panic!("the run must be refused"),
            Err(e) => e,
        }
    }

    #[test]
    fn an_explore_checkpoint_resumes_only_with_its_own_tier() {
        let cfg = |generations, explore_islands| Config { generations, explore_islands, ..golden_config() };
        let (_, default_text, _) = checkpointed(&cfg(10, 0), None);
        assert!(!default_text.contains("explore_islands"), "the key is written only for a tier");
        let old = Checkpoint::from_text(&default_text).expect("parse default checkpoint");
        assert_eq!(old.config.explore_islands, 0, "a checkpoint without the key loads as 0");
        let (_, text, _) = checkpointed(&cfg(10, 2), None);
        assert!(text.contains("config explore_islands 2\n"));
        for other in [0, 1, 3] {
            let saved = Checkpoint::from_text(&text).expect("parse checkpoint");
            let err = refused(&cfg(20, other), Some(saved));
            assert!(err.contains("explore_islands"), "{other}: {err}");
        }
        let err = refused(&cfg(20, 2), Some(old));
        assert!(err.contains("explore_islands"), "{err}");
    }

    #[test]
    fn explore_islands_need_an_exploit_island() {
        for (islands, explore_islands) in [(4, 4), (4, 5), (1, 1)] {
            let err = refused(&Config { islands, explore_islands, generations: 2, ..golden_config() }, None);
            assert!(err.contains("--explore-islands"), "{islands} {explore_islands}: {err}");
        }
    }

    /// A noisy table that no run here solves, so every run uses its budget.
    fn breadth_sample(seed: u32, n: usize) -> Sample {
        let mut rng = Mt::new(seed);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let a = rng.next_f64() * 4.0 - 2.0;
            let b = rng.next_f64() * 2.0 - 1.0;
            x.push(vec![a, b]);
            y.push((1.5 * a).sin() * b + 0.5 * b * b + 0.3 * a + 0.1 * (rng.next_f64() - 0.5));
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
    }

    /// From the development log of one run: the distinct island best errors
    /// at the last generation, the islands whose best is the run's best error
    /// (8 of 8 is one trial), and that best error.
    fn breadth(cfg: &Config, train: &Sample, test: &Sample) -> (usize, usize, f64) {
        use std::sync::{Arc, Mutex};
        let seen = Arc::new(Mutex::new((Vec::<f64>::new(), f64::INFINITY)));
        let sink = seen.clone();
        let cfg = Config {
            dev_log: Some(Arc::new(move |r: &crate::devlog::GenRecord| {
                let mut s = sink.lock().unwrap();
                s.0 = r.island_best.clone();
                s.1 = s.1.min(r.best_error);
            })),
            ..cfg.clone()
        };
        run_resumable(&cfg, train, test, None, 0, |_| {}).expect("run");
        let (island_best, best) = seen.lock().unwrap().clone();
        let distinct = island_best.iter().map(|b| b.to_bits()).collect::<std::collections::HashSet<_>>().len();
        (distinct, island_best.iter().filter(|&&b| b == best).count(), best)
    }

    #[test]
    fn the_explore_tier_ends_on_more_distinct_island_bests_than_the_default() {
        let (train, test) = (breadth_sample(7, 300), breadth_sample(8, 100));
        // [arm][distinct island bests, islands holding the best]
        let mut sums = [[0usize; 2]; 2];
        for seed in 1..=3 {
            for (arm, explore_islands) in [0usize, 6].into_iter().enumerate() {
                let cfg = Config {
                    islands: 8,
                    pop: 16,
                    generations: 300,
                    max_complexity: 12,
                    fit_iters: 2,
                    patience: 0,
                    seed,
                    gpu: false,
                    explore_islands,
                    ..Default::default()
                };
                let (distinct, holders, best) = breadth(&cfg, &train, &test);
                eprintln!(
                    "breadth seed {seed} explore_islands {explore_islands}: distinct island bests {distinct}, \
                     islands holding the best {holders}, best training error {best:.6e}"
                );
                sums[arm][0] += distinct;
                sums[arm][1] += holders;
            }
        }
        eprintln!("breadth sums over seeds 1-3: default {:?}, explore_islands 6 {:?}", sums[0], sums[1]);
        assert!(sums[1][0] > sums[0][0], "distinct island bests: {sums:?}");
        assert!(sums[1][1] < sums[0][1], "islands holding the best: {sums:?}");
    }

    #[test]
    fn an_island_scope_run_and_its_resume_match_exactly() {
        for rule in [RestartPolicy::Archive, RestartPolicy::Island] {
            let cfg = |generations| Config {
                generations,
                front_scope: FrontScope::Island,
                restart_policy: rule,
                restart_isolation: 7,
                ..golden_config()
            };
            let (uninterrupted, full_text, _) = checkpointed(&cfg(40), None);
            let (again, again_text, _) = checkpointed(&cfg(40), None);
            assert_eq!(uninterrupted.archive.hash(), again.archive.hash(), "{rule:?}: same seed, same archive");
            assert_eq!(full_text, again_text, "{rule:?}: same seed, same exit checkpoint");
            let (_, saved_text, saved_restarts) = checkpointed(&cfg(20), None);
            assert!(saved_restarts > 0, "{rule:?}: an island restarts before the save");
            assert!(saved_text.contains("config front_scope island\n") && saved_text.contains("\nisland_front "), "{rule:?}");
            let saved = Checkpoint::from_text(&saved_text).expect("parse saved checkpoint");
            let (resumed, resumed_text, _) = checkpointed(&cfg(40), Some(saved));
            assert_eq!(uninterrupted.archive.hash(), resumed.archive.hash(), "{rule:?}: the archive");
            assert_eq!(full_text, resumed_text, "{rule:?}: the exit checkpoint, island fronts included");
        }
    }

    #[test]
    fn an_island_scope_checkpoint_resumes_only_under_its_own_scope() {
        let cfg = |generations, front_scope| Config { generations, front_scope, ..golden_config() };
        let (_, global_text, _) = checkpointed(&cfg(10, FrontScope::Global), None);
        assert!(!global_text.contains("front_scope") && !global_text.contains("island_front"));
        let (_, text, _) = checkpointed(&cfg(10, FrontScope::Island), None);
        let err = refused(&cfg(20, FrontScope::Global), Some(Checkpoint::from_text(&text).expect("parse")));
        assert!(err.contains("front_scope"), "{err}");
        let err = refused(&cfg(20, FrontScope::Island), Some(Checkpoint::from_text(&global_text).expect("parse")));
        assert!(err.contains("front_scope"), "{err}");
        let stripped: String = text
            .lines()
            .filter(|l| !l.starts_with("config front_scope") && !l.starts_with("config_hash "))
            .map(|l| format!("{l}\n"))
            .collect();
        match Checkpoint::from_text(&stripped) {
            Ok(_) => panic!("island_front records under the global scope must be refused"),
            Err(e) => assert!(e.contains("island_front"), "{e}"),
        }
    }

    #[test]
    fn one_island_front_scope_equals_the_global_front() {
        let cfg = |front_scope| Config { islands: 1, front_scope, ..golden_config() };
        let (global, global_text, _) = checkpointed(&cfg(FrontScope::Global), None);
        let (island, island_text, _) = checkpointed(&cfg(FrontScope::Island), None);
        assert_eq!(global.archive.hash(), island.archive.hash());
        let state = |t: &str| -> Vec<String> {
            t.lines().filter(|l| l.starts_with("ind ") || l.starts_with("rng ")).map(String::from).collect()
        };
        assert_eq!(state(&global_text), state(&island_text), "populations and generators");
    }

    #[test]
    fn the_island_scope_refuses_the_qd_policy() {
        let cfg = Config { front_scope: FrontScope::Island, qd_bins: 8, qd_policy: true, generations: 2, ..golden_config() };
        let err = refused(&cfg, None);
        assert!(err.contains("--qd-policy") && err.contains("QD archive"), "{err}");
    }

    #[test]
    fn the_island_scope_ends_on_more_distinct_island_bests_than_the_global_front() {
        let (train, test) = (breadth_sample(7, 300), breadth_sample(8, 100));
        // [scope][distinct island bests, islands holding the best]
        let mut sums = [[0usize; 2]; 2];
        for seed in 1..=3 {
            for (arm, front_scope) in [FrontScope::Global, FrontScope::Island].into_iter().enumerate() {
                let cfg = Config {
                    islands: 8,
                    pop: 16,
                    generations: 300,
                    max_complexity: 12,
                    fit_iters: 2,
                    patience: 0,
                    seed,
                    gpu: false,
                    front_scope,
                    ..Default::default()
                };
                let (distinct, holders, best) = breadth(&cfg, &train, &test);
                eprintln!(
                    "breadth seed {seed} front_scope {}: distinct island bests {distinct}, \
                     islands holding the best {holders}, best training error {best:.6e}",
                    front_scope.as_str()
                );
                sums[arm][0] += distinct;
                sums[arm][1] += holders;
            }
        }
        eprintln!("breadth sums over seeds 1-3: global {:?}, island {:?}", sums[0], sums[1]);
        assert!(sums[1][0] > sums[0][0], "distinct island bests: {sums:?}");
    }

    fn breeding_fixture() -> (Sample, Config) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..21 {
            let a = -1.0 + 2.0 * i as f64 / 20.0;
            x.push(vec![a, 0.5 * a + 1.0]);
            y.push(a * a);
        }
        let s = Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None };
        let cfg = Config { pop: 12, fit_iters: 0, qd_bins: 4, qd_policy: true, gpu: false, ..Default::default() };
        (s, cfg)
    }

    #[test]
    fn an_isolated_island_breeds_from_its_own_members_only() {
        let (s, cfg) = breeding_fixture();
        let data = EvalSample::new(&s, &cfg);
        let island: Vec<Ind> = (0..cfg.pop)
            .map(|i| {
                let tree = match i % 3 {
                    0 => input(i % 2),
                    1 => node(Op::Add, vec![input(0), constant(i as f64)]),
                    _ => node(Op::Mul, vec![input(1), constant(0.5 + i as f64)]),
                };
                Ind::evaluated(tree, &data, &cfg)
            })
            .collect();
        let front = vec![
            Sol { tree: node(Op::Mul, vec![input(0), input(0)]), error: 0.0, complexity: 3, cost: 3.0 },
            Sol { tree: node(Op::Sub, vec![input(1), constant(3.0)]), error: 0.4, complexity: 3, cost: 3.0 },
        ];
        let mut qd = QdArchive::new(cfg.qd_bins, 4).with_nvars(2);
        qd.record(
            (0..8).map(|k| {
                let t = node(Op::Mul, vec![input(k % 2), constant(k as f64 + 2.0)]);
                (t, 0.1 * k as f64, 3.0)
            }),
            &s,
        );
        // 200 parent draws: each tree and age, and the generator state after.
        let draws = |parents: Parents<'_>| {
            let mut rng = Mt::new(3);
            let picks: Vec<(u64, u32)> = (0..200)
                .map(|_| {
                    let (tree, age) = pick_parent_aged(&island, &parents, &mut rng, false, &[], &[]);
                    (expr::structural_hash(&tree), age)
                })
                .collect();
            (picks, *rng.state().0, rng.state().1)
        };
        let own = draws(Parents { front: &[], qd: None, source: ParentSource { front_p: 0.0, shared: true } });
        for control in [0u8, 1] {
            let pools = |source| Parents { front: &front, qd: Some((Some(&qd), control)), source };
            assert!(draws(pools(ISOLATED)) == own, "control {control}: an isolated island drew a --qd-policy parent");
            assert!(draws(pools(OPEN)) != own, "control {control}: an open island draws --qd-policy parents");
        }
    }

    // The schedule-equivalence property: the module against the inline code it
    // replaced, on both population adapters.

    const ERRORS: [f64; 7] = [0.1, 0.2, 0.2 + 1e-12, 0.3, 0.5, f64::INFINITY, f64::NAN];

    /// The policy state of the inline code.
    struct Inline {
        stagnant: Vec<usize>,
        restart_state: Option<RestartState>,
        restarts_total: u64,
    }

    impl Inline {
        fn new(cfg: &Config) -> Self {
            Inline {
                stagnant: vec![0; cfg.islands.max(1)],
                restart_state: (cfg.restart_policy == RestartPolicy::Island)
                    .then(|| RestartState::new(cfg.islands.max(1))),
                restarts_total: 0,
            }
        }

        fn front_p(&self, cfg: &Config, i: usize) -> f64 {
            let front_p = if cfg.front_parents { FRONT_PARENT_P } else { 0.0 };
            self.restart_state.as_ref().map_or(front_p, |r| r.front_p(i, front_p))
        }

        fn is_isolated(&self, i: usize) -> bool {
            self.restart_state.as_ref().is_some_and(|r| r.is_isolated(i))
        }

        fn assert_matches(&self, policy: &IslandPolicy, cfg: &Config, at: &str) {
            assert_eq!(self.stagnant, policy.stagnant(), "{at}: stagnant");
            for i in 0..cfg.islands.max(1) {
                assert_eq!(self.restart_state.as_ref().map(|r| r.saved(i)), policy.stall(i), "{at}: island_stall {i}");
                let source = policy.parent_source(i);
                assert_eq!(self.front_p(cfg, i).to_bits(), source.front_p.to_bits(), "{at}: front_p {i}");
                assert_eq!(!self.is_isolated(i), source.shared, "{at}: shared {i}");
            }
            assert_eq!(self.restart_state.as_ref().map_or(0, |s| s.isolated_count()), policy.isolated_count(), "{at}");
            assert_eq!(self.restarts_total, policy.restarts_total(), "{at}: restarts_total");
        }
    }

    /// The migration and restart blocks of `run_search` before this module,
    /// with each move recorded.
    #[allow(clippy::too_many_arguments)]
    fn inline_tree_barrier(
        gen: usize,
        cfg: &Config,
        islands: &mut [Vec<Ind>],
        rngs: &mut [Mt],
        inline: &mut Inline,
        archive: &[Sol],
        aggs: &AggTable,
        gen_train: &Sample,
        gen_data: &EvalSample,
        defer_gpu: bool,
        qd_archive: Option<&QdArchive>,
        cached_descriptors_valid: bool,
    ) -> (Vec<Move>, Vec<usize>) {
        let (nvars, fit_train) = (2, gen_train);
        let (stagnant, restart_state) = (&mut inline.stagnant, &mut inline.restart_state);
        let mut moves = Vec::new();
        let mut restarted = Vec::new();
        if cfg.islands > 1 && cfg.migration_interval > 0 && (gen + 1) % cfg.migration_interval == 0 {
            let bests: Vec<Option<Node>> = islands
                .iter()
                .map(|pop| pop.iter().min_by(|a, b| crate::error_order::cmp_error(a.error, b.error)).map(|i| i.tree.clone()))
                .collect();
            let best_ages: Vec<u32> = islands
                .iter()
                .map(|pop| pop.iter().min_by(|a, b| crate::error_order::cmp_error(a.error, b.error)).map_or(0, |i| i.age))
                .collect();
            for ii in 0..islands.len() {
                if restart_state.as_ref().is_some_and(|r| r.is_isolated(ii)) {
                    continue;
                }
                let src = (ii + 1) % islands.len();
                moves.push(Move { from: src, to: ii });
                if let Some(t) = &bests[src] {
                    let victim = if cfg.qd_policy {
                        match qd_archive.and_then(|q| q.least_occupied_descriptor()) {
                            Some(target) => {
                                let descriptors: Vec<qd::Descriptor> = islands[ii]
                                    .iter()
                                    .map(|ind| {
                                        if cached_descriptors_valid {
                                            if let Some(d) = ind.descriptor {
                                                return d;
                                            }
                                        }
                                        qd::descriptor(&ind.tree, gen_train)
                                    })
                                    .collect();
                                descriptors
                                    .iter()
                                    .enumerate()
                                    .min_by(|(_, a), (_, b)| {
                                        let score = |d: &qd::Descriptor| {
                                            (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2)
                                        };
                                        score(a).partial_cmp(&score(b)).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                    .map(|(i, _)| i)
                            }
                            None => None,
                        }
                    } else {
                        None
                    };
                    let slot = victim.or_else(|| {
                        if cfg.selection == Selection::Afp {
                            return afp_worst(&islands[ii]);
                        }
                        islands[ii]
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| crate::error_order::cmp_error(a.error, b.error))
                            .map(|(i, _)| i)
                    });
                    if let Some(worst) = slot.and_then(|i| islands[ii].get_mut(i)) {
                        *worst = if defer_gpu { Ind::pending(t.clone(), cfg) } else { Ind::evaluated(t.clone(), gen_data, cfg) };
                        worst.age = best_ages[src];
                    }
                }
            }
        }
        if cfg.restart_interval > 0 && (gen + 1) % cfg.restart_interval == 0 {
            let arch_best = archive.iter().map(|s| s.error).fold(f64::INFINITY, f64::min);
            let keeper = restart_state.as_ref().and_then(|_| {
                let bests: Vec<f64> = islands.iter().map(|p| population_best(p)).collect();
                RestartState::keeper(&bests, arch_best)
            });
            for ii in 0..islands.len() {
                let restart_now = match restart_state.as_ref() {
                    Some(rs) => rs.due(ii, keeper, cfg.restart_interval),
                    None => {
                        let cur_best = islands[ii].iter().map(|i| i.error).fold(f64::INFINITY, f64::min);
                        if cur_best > arch_best + 1e-9 {
                            stagnant[ii] += 1;
                        } else {
                            stagnant[ii] = 0;
                        }
                        stagnant[ii] >= 2
                    }
                };
                if restart_now {
                    let mut r = rngs[ii].clone();
                    let qd_target = if cfg.qd_policy { qd_archive.and_then(|q| q.least_occupied_descriptor()) } else { None };
                    let mut have_const = false;
                    for ind in islands[ii].iter_mut() {
                        let hc = have_const;
                        let mut t = draw_row_dependent(cfg, nvars, &mut have_const, || {
                            let mut t = random_tree_with(&mut r, nvars, &cfg.ops, cfg.init_depth, Some(aggs));
                            if let Some(target) = qd_target {
                                let mut best = t.clone();
                                let best_d = qd::descriptor(&t, gen_train);
                                let mut best_score = if hc && !expr::row_dependent(&t) {
                                    f64::INFINITY
                                } else {
                                    (best_d.usage - target.usage).powi(2) + (best_d.skew - target.skew).powi(2)
                                };
                                for _ in 0..3 {
                                    let cand = random_tree_with(&mut r, nvars, &cfg.ops, cfg.init_depth, Some(aggs));
                                    let d = qd::descriptor(&cand, gen_train);
                                    let score = (d.usage - target.usage).powi(2) + (d.skew - target.skew).powi(2);
                                    if score < best_score && !(hc && !expr::row_dependent(&cand)) {
                                        best = cand;
                                        best_score = score;
                                    }
                                }
                                t = best;
                            }
                            t
                        });
                        let mut fitted = None;
                        let mut before = None; // #1078
                        if cfg.fit_iters > 0 && !gen_train.is_multi() && !defers_fit_to_gpu(cfg, defer_gpu) {
                            before = pre_fit_tree(cfg, &t);
                            fitted = fit_constants(&mut t, cfg, fit_train, gen_train);
                        }
                        *ind = if defer_gpu && before.is_none() {
                            Ind::pending(t, cfg)
                        } else {
                            gate_fit(None, None, FitPath::Restart, before, Ind::evaluated_fitted(t, fitted, gen_data, cfg), gen_data, cfg)
                        };
                    }
                    rngs[ii] = r;
                    stagnant[ii] = 0;
                    if let Some(rs) = restart_state.as_mut() {
                        rs.restarted(ii, cfg.restart_isolation);
                    }
                    inline.restarts_total += 1;
                    restarted.push(ii);
                }
            }
        }
        (moves, restarted)
    }

    /// `Arena::migrate` before this module: the ring with an isolation skip.
    #[cfg(feature = "cuda")]
    fn inline_arena_migrate(a: &mut Arena, skip: &[bool]) {
        if a.islands < 2 {
            return;
        }
        let bests: Vec<usize> = (0..a.islands).map(|i| a.best(i)).collect();
        let worsts: Vec<usize> = (0..a.islands).map(|i| a.worst(i)).collect();
        for i in 0..a.islands {
            if skip.get(i).copied().unwrap_or(false) {
                continue;
            }
            let src = bests[(i + 1) % a.islands];
            a.copy_slot(worsts[i], src);
        }
    }

    /// The migration and restart blocks of `run_postfix_search` before this
    /// module, with each move recorded.
    #[cfg(feature = "cuda")]
    fn inline_arena_barrier(
        gen: usize,
        cfg: &Config,
        next: &mut Arena,
        variation: &Variation,
        inline: &mut Inline,
        archive: &[Sol],
    ) -> (Vec<Move>, Vec<usize>) {
        let (n_islands, pop) = (cfg.islands.max(1), cfg.pop);
        let (stagnant, restart_state) = (&mut inline.stagnant, &mut inline.restart_state);
        let mut moves = Vec::new();
        let mut restarted = Vec::new();
        if n_islands > 1 && cfg.migration_interval > 0 && (gen + 1) % cfg.migration_interval == 0 {
            let skip: Vec<bool> =
                (0..n_islands).map(|i| restart_state.as_ref().is_some_and(|r| r.is_isolated(i))).collect();
            moves = (0..n_islands).filter(|&i| !skip[i]).map(|i| Move { from: (i + 1) % n_islands, to: i }).collect();
            inline_arena_migrate(next, &skip);
        }
        if cfg.restart_interval > 0 && (gen + 1) % cfg.restart_interval == 0 {
            let arch_best = archive.iter().map(|s| s.error).fold(f64::INFINITY, f64::min);
            let keeper = restart_state.as_ref().and_then(|_| {
                let bests: Vec<f64> = (0..n_islands)
                    .map(|i| (0..pop).map(|j| next.err[next.slot_index(i, j)]).fold(f64::INFINITY, f64::min))
                    .collect();
                RestartState::keeper(&bests, arch_best)
            });
            for i in 0..n_islands {
                let restart_now = match restart_state.as_ref() {
                    Some(rs) => rs.due(i, keeper, cfg.restart_interval),
                    None => {
                        let cur_best = (0..pop).map(|j| next.err[next.slot_index(i, j)]).fold(f64::INFINITY, f64::min);
                        if cur_best > arch_best + 1e-9 {
                            stagnant[i] += 1;
                        } else {
                            stagnant[i] = 0;
                        }
                        stagnant[i] >= 2
                    }
                };
                if restart_now {
                    variation.restart_island(next, i, gen as u32);
                    stagnant[i] = 0;
                    if let Some(rs) = restart_state.as_mut() {
                        rs.restarted(i, cfg.restart_isolation);
                    }
                    inline.restarts_total += 1;
                    restarted.push(i);
                }
            }
        }
        (moves, restarted)
    }

    /// Records the moves a policy hands to a population adapter.
    struct Recorded<'p, P> {
        inner: &'p mut P,
        moves: Vec<Move>,
    }

    impl<P: IslandPopulation> IslandBests for Recorded<'_, P> {
        fn island_count(&self) -> usize {
            self.inner.island_count()
        }

        fn best_error(&self, island: usize) -> f64 {
            self.inner.best_error(island)
        }
    }

    impl<P: IslandPopulation> IslandPopulation for Recorded<'_, P> {
        fn migrate(&mut self, moves: &[Move], policy: MigrationPolicy) -> Migrants {
            self.moves.extend_from_slice(moves);
            self.inner.migrate(moves, policy)
        }

        fn restart(&mut self, island: usize) {
            self.inner.restart(island);
        }
    }

    fn random_case(rng: &mut Mt) -> Config {
        let qd_policy = rng.bool(0.3);
        Config {
            islands: 1 + rng.usize_below(5),
            pop: 2 + rng.usize_below(4),
            migration_interval: [0, 1, 2, 3, 5][rng.usize_below(5)],
            restart_interval: [0, 1, 2, 3, 4][rng.usize_below(5)],
            restart_policy: if rng.bool(0.5) { RestartPolicy::Island } else { RestartPolicy::Archive },
            restart_isolation: rng.usize_below(6),
            front_parents: rng.bool(0.8),
            selection: if rng.bool(0.3) { Selection::Afp } else { Selection::Lexicographic },
            qd_bins: if qd_policy { 4 } else { 0 },
            qd_policy,
            fit_iters: if rng.bool(0.3) { 2 } else { 0 },
            max_complexity: 12,
            gpu: false,
            ..Default::default()
        }
    }

    fn tree_view(islands: &[Vec<Ind>]) -> Vec<(u64, u64, u64, u32)> {
        islands
            .iter()
            .flatten()
            .map(|i| (expr::structural_hash(&i.tree), i.error.to_bits(), i.complexity.to_bits(), i.age))
            .collect()
    }

    fn rng_view(rngs: &[Mt]) -> Vec<(Vec<u32>, usize)> {
        rngs.iter().map(|r| (r.state().0.to_vec(), r.state().1)).collect()
    }

    #[cfg(feature = "cuda")]
    type ArenaView = (Vec<u32>, Vec<u16>, Vec<u16>, Vec<u64>, Vec<u64>);

    #[cfg(feature = "cuda")]
    fn arena_view(a: &Arena) -> ArenaView {
        (a.words.clone(), a.len.clone(), a.cost.clone(), a.hash.clone(), a.err.iter().map(|e| e.to_bits()).collect())
    }

    #[cfg(feature = "cuda")]
    fn random_arena(rng: &mut Mt, islands: usize, pop: usize) -> Arena {
        let mut arena = Arena::new(islands, pop, 64);
        for idx in 0..islands * pop {
            loop {
                let tree = expr::simplify(&random_tree_with(rng, 2, &Op::default_set(), 3, None));
                let mut words = Vec::new();
                if crate::gpu::compile(&tree, &mut words) && arena.set_slot(idx, &words) {
                    break;
                }
            }
        }
        arena
    }

    /// The archive best of one generation: an island best (so a keeper exists)
    /// or a value from the error palette.
    fn archive_of(rng: &mut Mt, bests: impl Iterator<Item = f64>) -> [Sol; 1] {
        let own = bests.fold(f64::INFINITY, f64::min);
        [sol(if rng.bool(0.5) { own } else { ERRORS[rng.usize_below(ERRORS.len() - 1)] })]
    }

    /// Barriers with a move, barriers with an isolated destination skipped,
    /// restarts under the archive rule and under the island rule.
    type Seen = [usize; 4];

    fn count(seen: &mut Seen, cfg: &Config, moves: &[Move], restarts: &[usize]) {
        seen[0] += usize::from(!moves.is_empty());
        seen[1] += usize::from(!moves.is_empty() && moves.len() < cfg.islands);
        seen[2 + usize::from(cfg.restart_policy == RestartPolicy::Island)] += restarts.len();
    }

    /// One random case on the owned-tree adapter: 25 generations of random
    /// island errors, the inline code on one copy and the module on another.
    fn tree_case(case: u32, cfg: &Config, s: &Sample, aggs: &AggTable, rng: &mut Mt, seen: &mut Seen) {
        let n = cfg.islands;
        let (defer_gpu, cached) = (rng.bool(0.3), rng.bool(0.5));
        let data = EvalSample::new(s, cfg);
        let islands: Vec<Vec<Ind>> = (0..n)
            .map(|_| {
                (0..cfg.pop)
                    .map(|_| {
                        let mut ind = Ind::evaluated(random_tree_with(rng, 2, &cfg.ops, 3, Some(aggs)), &data, cfg);
                        ind.age = rng.usize_below(9) as u32;
                        ind
                    })
                    .collect()
            })
            .collect();
        let qd = cfg.qd_policy.then(|| {
            let mut q = QdArchive::new(cfg.qd_bins, 4).with_nvars(2);
            q.record(
                islands.iter().flatten().filter(|i| i.error.is_finite()).map(|i| (i.tree.clone(), i.error, i.complexity)),
                s,
            );
            q
        });
        let rngs: Vec<Mt> = (0..n).map(|i| Mt::new(case * 31 + i as u32)).collect();
        let (mut want_islands, mut want_rngs, mut want) = (islands.clone(), rngs.clone(), Inline::new(cfg));
        let (mut got_islands, mut got_rngs, mut policy) = (islands, rngs, IslandPolicy::new(cfg));
        for gen in 0..25 {
            let at = format!("owned trees, case {case} generation {gen}, {}", describe(cfg, defer_gpu));
            for (want_pop, got_pop) in want_islands.iter_mut().zip(got_islands.iter_mut()) {
                for (w, g) in want_pop.iter_mut().zip(got_pop.iter_mut()) {
                    let e = ERRORS[rng.usize_below(ERRORS.len())];
                    w.error = e;
                    g.error = e;
                }
            }
            want.assert_matches(&policy, cfg, &at);
            let archive = archive_of(rng, want_islands.iter().map(|p| population_best(p)));
            if let Some(rs) = want.restart_state.as_mut() {
                let bests: Vec<f64> = want_islands.iter().map(|p| population_best(p)).collect();
                rs.tick(&bests);
            }
            policy.observe(&got_islands[..]);
            let (want_moves, want_restarts) = inline_tree_barrier(
                gen, cfg, &mut want_islands, &mut want_rngs, &mut want, &archive, aggs, s, &data, defer_gpu, qd.as_ref(),
                cached,
            );
            let mut adapter = TreeIslands {
                islands: &mut got_islands,
                rngs: &mut got_rngs,
                cfg,
                nvars: 2,
                aggs,
                gen_train: s,
                gen_data: &data,
                fit_train: s,
                defer_gpu,
                qd: qd.as_ref(),
                cached_descriptors_valid: cached,
                fit_counts: None,
            };
            let mut recorded = Recorded { inner: &mut adapter, moves: Vec::new() };
            let got_restarts = policy.barrier(gen, &archive, &mut recorded);
            assert_eq!(recorded.moves, want_moves, "{at}: moves");
            assert_eq!(got_restarts, want_restarts, "{at}: restarts");
            assert!(tree_view(&got_islands) == tree_view(&want_islands), "{at}: populations");
            assert!(rng_view(&got_rngs) == rng_view(&want_rngs), "{at}: island generators");
            want.assert_matches(&policy, cfg, &at);
            count(seen, cfg, &want_moves, &want_restarts);
        }
    }

    /// One random case on the flat arena adapter, as [`tree_case`].
    #[cfg(feature = "cuda")]
    fn arena_case(case: u32, cfg: &Config, rng: &mut Mt, seen: &mut Seen) {
        let n = cfg.islands;
        let variation = Variation::new(64, 2, 12, case, &Op::default_set(), 3, true).expect("variation");
        let arena = random_arena(rng, n, cfg.pop);
        let (mut want_arena, mut want) = (arena.clone(), Inline::new(cfg));
        let (mut got_arena, mut policy) = (arena, IslandPolicy::new(cfg));
        for gen in 0..25 {
            let at = format!("flat arena, case {case} generation {gen}, {}", describe(cfg, false));
            for idx in 0..n * cfg.pop {
                let e = ERRORS[rng.usize_below(ERRORS.len())];
                want_arena.err[idx] = e;
                got_arena.err[idx] = e;
            }
            want.assert_matches(&policy, cfg, &at);
            let archive = archive_of(rng, (0..n).map(|i| want_arena.best_error(i)));
            if let Some(rs) = want.restart_state.as_mut() {
                let bests: Vec<f64> = (0..n)
                    .map(|i| (0..cfg.pop).map(|j| want_arena.err[want_arena.slot_index(i, j)]).fold(f64::INFINITY, f64::min))
                    .collect();
                rs.tick(&bests);
            }
            policy.observe(&got_arena);
            let (want_moves, want_restarts) = inline_arena_barrier(gen, cfg, &mut want_arena, &variation, &mut want, &archive);
            let mut adapter = ArenaIslands { arena: &mut got_arena, variation: &variation, gen: gen as u32 };
            let mut recorded = Recorded { inner: &mut adapter, moves: Vec::new() };
            let got_restarts = policy.barrier(gen, &archive, &mut recorded);
            assert_eq!(recorded.moves, want_moves, "{at}: moves");
            assert_eq!(got_restarts, want_restarts, "{at}: restarts");
            assert!(arena_view(&got_arena) == arena_view(&want_arena), "{at}: arenas");
            want.assert_matches(&policy, cfg, &at);
            count(seen, cfg, &want_moves, &want_restarts);
        }
    }

    /// The flat arena exists only in a `cuda` build, so a CPU build checks the
    /// owned-tree adapter alone.
    #[test]
    fn the_island_policy_reproduces_the_inline_schedule_on_both_population_adapters() {
        let (s, _) = breeding_fixture();
        let aggs = AggTable::from_sample(&s);
        let mut rng = Mt::new(1081);
        let mut trees: Seen = [0; 4];
        #[cfg(feature = "cuda")]
        let mut flat: Seen = [0; 4];
        for case in 0..240 {
            let cfg = random_case(&mut rng);
            tree_case(case, &cfg, &s, &aggs, &mut rng, &mut trees);
            #[cfg(feature = "cuda")]
            arena_case(case, &cfg, &mut rng, &mut flat);
        }
        assert!(trees.iter().all(|&c| c > 0), "owned trees: [moves, skipped, archive restarts, island restarts] = {trees:?}");
        #[cfg(feature = "cuda")]
        assert!(flat.iter().all(|&c| c > 0), "flat arena: [moves, skipped, archive restarts, island restarts] = {flat:?}");
    }

    /// Host code only: the arena adapter never calls the device here.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_flat_arena_applies_the_explore_tier() {
        let (n, k, pop) = (4usize, 2usize, 3usize);
        let cfg = Config { explore_islands: k, pop, ..policy_cfg(n, RestartPolicy::Archive, 1, 2, 0) };
        let mut rng = Mt::new(1100);
        let variation = Variation::new(64, 2, 12, 1100, &Op::default_set(), 3, true).expect("variation");
        let mut arena = random_arena(&mut rng, n, pop);
        let mut policy = IslandPolicy::new(&cfg);
        let island_view = |a: &Arena, i: usize| -> Vec<(Vec<u32>, u64)> {
            (0..pop)
                .map(|j| {
                    let idx = a.slot_index(i, j);
                    (a.slot(idx).to_vec(), a.err[idx].to_bits())
                })
                .collect()
        };
        let mut explore_restarts_seen = 0;
        for gen in 0..30 {
            for idx in 0..n * pop {
                arena.err[idx] = ERRORS[rng.usize_below(ERRORS.len())];
            }
            let before = arena.clone();
            let archive = archive_of(&mut rng, (0..n).map(|i| arena.best_error(i)));
            policy.observe(&arena);
            let mut adapter = ArenaIslands { arena: &mut arena, variation: &variation, gen: gen as u32 };
            let mut recorded = Recorded { inner: &mut adapter, moves: Vec::new() };
            let restarts = policy.barrier(gen, &archive, &mut recorded);
            assert_eq!(recorded.moves, [Move { from: 1, to: 0 }, Move { from: 0, to: 1 }], "generation {gen}");
            for i in n - k..n {
                let mut want = before.clone();
                if restarts.contains(&i) {
                    variation.restart_island(&mut want, i, gen as u32);
                    explore_restarts_seen += 1;
                }
                assert!(island_view(&arena, i) == island_view(&want, i), "generation {gen}: explore island {i}");
            }
        }
        assert!(explore_restarts_seen >= 8, "{explore_restarts_seen} explore restarts");
    }

    /// Host code only: decoding a slot never calls the device.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_flat_arena_offers_each_island_to_its_own_front() {
        let (n, pop) = (3usize, 4usize);
        let mut rng = Mt::new(1070);
        let mut arena = random_arena(&mut rng, n, pop);
        for idx in 0..n * pop {
            arena.err[idx] = 0.1 + idx as f64 * 0.01;
        }
        let cfg = Config { front_scope: FrontScope::Island, pop, ..policy_cfg(n, RestartPolicy::Archive, 1, 0, 0) };
        let mut policy = IslandPolicy::new(&cfg);
        policy.offer(&arena, &|_| true);
        for i in 0..n {
            let own: std::collections::HashSet<u64> = (0..pop)
                .filter_map(|j| arena.decode(arena.slot_index(i, j)))
                .map(|t| expr::structural_hash(&t))
                .collect();
            let front = policy.island_front(i);
            assert!(!front.is_empty(), "island {i}");
            assert!(front.iter().all(|s| own.contains(&expr::structural_hash(&s.tree))), "island {i} took foreign material");
        }
    }

    fn tournament_fixture(n: usize, pop: usize, error: impl Fn(usize, usize) -> f64) -> (Sample, Config, AggTable, Vec<Vec<Ind>>) {
        let (s, base) = breeding_fixture();
        let cfg = Config {
            islands: n,
            pop,
            qd_policy: false,
            qd_bins: 0,
            migration_policy: MigrationPolicy::RingTournament,
            ..base
        };
        let aggs = AggTable::from_sample(&s);
        let data = EvalSample::new(&s, &cfg);
        let mut rng = Mt::new(1029);
        let islands = (0..n)
            .map(|i| {
                (0..pop)
                    .map(|j| {
                        let mut ind = Ind::evaluated(random_tree_with(&mut rng, 2, &cfg.ops, 3, Some(&aggs)), &data, &cfg);
                        ind.error = error(i, j);
                        ind.age = (i * pop + j) as u32;
                        ind
                    })
                    .collect()
            })
            .collect();
        (s, cfg, aggs, islands)
    }

    fn migrate_tree(
        s: &Sample,
        cfg: &Config,
        aggs: &AggTable,
        islands: &mut [Vec<Ind>],
        rngs: &mut [Mt],
        moves: &[Move],
        policy: MigrationPolicy,
    ) -> Migrants {
        let data = EvalSample::new(s, cfg);
        TreeIslands {
            islands,
            rngs,
            cfg,
            nvars: 2,
            aggs,
            gen_train: s,
            gen_data: &data,
            fit_train: s,
            defer_gpu: false,
            qd: None,
            cached_descriptors_valid: false,
            fit_counts: None,
        }
        .migrate(moves, policy)
    }

    #[test]
    fn ring_tournament_draws_the_migrant_with_the_destination_rng() {
        let (s, cfg, aggs, islands) = tournament_fixture(3, 16, |i, j| 1.0 + (i * 100 + j) as f64);
        let rngs: Vec<Mt> = (0..3).map(|i| Mt::new(70 + i as u32)).collect();
        let (mut after, mut after_rngs) = (islands.clone(), rngs.clone());
        let moves = [Move { from: 1, to: 0 }];
        let got = migrate_tree(&s, &cfg, &aggs, &mut after, &mut after_rngs, &moves, MigrationPolicy::RingTournament);
        assert_eq!(got, Migrants { sent: 1, skipped: 0 });
        let (before_rngs, after_view) = (rng_view(&rngs), rng_view(&after_rngs));
        assert!(after_view[0] != before_rngs[0], "the destination generator draws");
        assert!(after_view[1] == before_rngs[1] && after_view[2] == before_rngs[2], "the source stream is untouched");
        assert!(tree_view(&after[1..]) == tree_view(&islands[1..]), "no other island changes");
        let mut replay = rngs[0].clone();
        let winner = tournament(&islands[1], &mut replay, MIGRANT_TOURNAMENT);
        let worst = 15;
        assert_eq!(expr::structural_hash(&after[0][worst].tree), expr::structural_hash(&islands[1][winner].tree));
        assert_eq!(after[0][worst].age, islands[1][winner].age, "the migrant keeps its source age");
        assert!(tree_view(&after[..1])
            .iter()
            .enumerate()
            .all(|(k, v)| k == worst || *v == tree_view(&islands[..1])[k]));
        let (mut again, mut again_rngs) = (islands.clone(), rngs.clone());
        migrate_tree(&s, &cfg, &aggs, &mut again, &mut again_rngs, &moves, MigrationPolicy::RingTournament);
        assert!(tree_view(&again) == tree_view(&after) && rng_view(&again_rngs) == after_view, "a pure function of the seeds");
    }

    #[test]
    fn the_ring_policy_draws_no_random_number_and_sends_the_best() {
        let (s, cfg, aggs, islands) = tournament_fixture(3, 16, |i, j| 1.0 + (i * 100 + j) as f64);
        let rngs: Vec<Mt> = (0..3).map(|i| Mt::new(70 + i as u32)).collect();
        let (mut after, mut after_rngs) = (islands.clone(), rngs.clone());
        let got = migrate_tree(&s, &cfg, &aggs, &mut after, &mut after_rngs, &[Move { from: 1, to: 0 }], MigrationPolicy::Ring);
        assert_eq!(got, Migrants { sent: 1, skipped: 0 });
        assert!(rng_view(&after_rngs) == rng_view(&rngs));
        assert_eq!(expr::structural_hash(&after[0][15].tree), expr::structural_hash(&islands[1][0].tree));
    }

    #[test]
    fn a_migrant_whose_book_the_destination_holds_is_skipped() {
        // Every member of island 1 has error 5.0, so the winner holds 5.0.
        let (s, cfg, aggs, islands) = tournament_fixture(3, 16, |i, j| if i == 1 { 5.0 } else { 10.0 + (i * 100 + j) as f64 });
        let rngs: Vec<Mt> = (0..3).map(|i| Mt::new(70 + i as u32)).collect();
        let mut held = islands.clone();
        held[0][3].error = 5.0;
        let (mut after, mut after_rngs) = (held.clone(), rngs.clone());
        let got = migrate_tree(&s, &cfg, &aggs, &mut after, &mut after_rngs, &[Move { from: 1, to: 0 }], MigrationPolicy::RingTournament);
        assert_eq!(got, Migrants { sent: 0, skipped: 1 });
        assert!(tree_view(&after) == tree_view(&held), "the destination population is unchanged");
        let (mut free, mut free_rngs) = (islands.clone(), rngs.clone());
        let got = migrate_tree(&s, &cfg, &aggs, &mut free, &mut free_rngs, &[Move { from: 1, to: 0 }], MigrationPolicy::RingTournament);
        assert_eq!(got, Migrants { sent: 1, skipped: 0 }, "a destination without that error takes the migrant");
        // The same count reaches the policy, and so the development log.
        let two = Config { islands: 2, migration_interval: 1, ..cfg.clone() };
        let mut pair = vec![held[1].clone(), held[1].clone()];
        let mut pair_rngs = vec![Mt::new(1), Mt::new(2)];
        let data = EvalSample::new(&s, &two);
        let mut policy = IslandPolicy::new(&two);
        policy.observe(&pair[..]);
        policy.barrier(
            0,
            &[],
            &mut TreeIslands {
                islands: &mut pair,
                rngs: &mut pair_rngs,
                cfg: &two,
                nvars: 2,
                aggs: &aggs,
                gen_train: &s,
                gen_data: &data,
                fit_train: &s,
                defer_gpu: false,
                qd: None,
                cached_descriptors_valid: false,
                fit_counts: None,
            },
        );
        assert_eq!(policy.migrants(), Migrants { sent: 0, skipped: 2 });
        policy.observe(&pair[..]);
        assert_eq!(policy.migrants(), Migrants::default(), "the count covers one generation");
    }

    #[test]
    fn a_source_without_a_finite_member_sends_nothing() {
        let (s, cfg, aggs, islands) = tournament_fixture(2, 8, |i, j| if i == 1 { f64::INFINITY } else { 1.0 + j as f64 });
        let rngs = vec![Mt::new(1), Mt::new(2)];
        let (mut after, mut after_rngs) = (islands.clone(), rngs.clone());
        let got = migrate_tree(&s, &cfg, &aggs, &mut after, &mut after_rngs, &[Move { from: 1, to: 0 }], MigrationPolicy::RingTournament);
        assert_eq!(got, Migrants::default());
        assert!(tree_view(&after) == tree_view(&islands) && rng_view(&after_rngs) == rng_view(&rngs));
    }

    #[test]
    fn a_ring_tournament_run_and_its_resume_match_exactly() {
        let cfg = |generations| Config { generations, migration_policy: MigrationPolicy::RingTournament, ..golden_config() };
        let (uninterrupted, full_text, _) = checkpointed(&cfg(40), None);
        let (again, again_text, _) = checkpointed(&cfg(40), None);
        assert_eq!(uninterrupted.archive.hash(), again.archive.hash(), "same seed, same archive");
        assert_eq!(full_text, again_text, "same seed, same exit checkpoint");
        let (_, saved_text, _) = checkpointed(&cfg(20), None);
        assert!(saved_text.contains("config migration_policy ring-tournament\n"));
        let saved = Checkpoint::from_text(&saved_text).expect("parse saved checkpoint");
        let (resumed, resumed_text, _) = checkpointed(&cfg(40), Some(saved));
        assert_eq!(uninterrupted.archive.hash(), resumed.archive.hash(), "the archive");
        assert_eq!(full_text, resumed_text, "the exit checkpoint");
        let (ring, _, _) = checkpointed(&Config { generations: 40, ..golden_config() }, None);
        assert!(ring.archive.hash() != uninterrupted.archive.hash(), "the policy changes the trajectory");
    }

    #[test]
    fn a_ring_tournament_checkpoint_resumes_only_under_its_own_policy() {
        let cfg = |generations, migration_policy| Config { generations, migration_policy, ..golden_config() };
        let (_, default_text, _) = checkpointed(&cfg(10, MigrationPolicy::Ring), None);
        assert!(!default_text.contains("migration_policy"), "the key is written only for ring-tournament");
        let old = Checkpoint::from_text(&default_text).expect("parse default checkpoint");
        assert_eq!(old.config.migration_policy, MigrationPolicy::Ring, "a checkpoint without the key loads as ring");
        let (_, text, _) = checkpointed(&cfg(10, MigrationPolicy::RingTournament), None);
        let saved = Checkpoint::from_text(&text).expect("parse checkpoint");
        let err = refused(&cfg(20, MigrationPolicy::Ring), Some(saved));
        assert!(err.contains("migration_policy"), "{err}");
        let err = refused(&cfg(20, MigrationPolicy::RingTournament), Some(old));
        assert!(err.contains("migration_policy"), "{err}");
    }

    fn describe(cfg: &Config, defer_gpu: bool) -> String {
        format!(
            "islands {} pop {} migration {} restart {} {:?} isolation {} front {} {:?} qd {} fit {} defer {}",
            cfg.islands,
            cfg.pop,
            cfg.migration_interval,
            cfg.restart_interval,
            cfg.restart_policy,
            cfg.restart_isolation,
            cfg.front_parents,
            cfg.selection,
            cfg.qd_policy,
            cfg.fit_iters,
            defer_gpu
        )
    }
}
