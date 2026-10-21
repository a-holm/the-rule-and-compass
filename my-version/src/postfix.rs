//! Flat postfix variation for the GPU search path (#122).
//!
//! The historical loop stores every individual as an owned [`Node`] tree:
//! a child is `p1.clone()`, a crossover grafts a cloned subtree, a mutation
//! walks the tree, and `expr::simplify` rebuilds it. At 10 000 candidates per
//! generation that allocation traffic is the dominant host cost.
//!
//! This module keeps the same fixed-capacity postfix representation the device
//! already evaluates ([`crate::genome::GenomePool`]) and performs tournament,
//! crossover, mutation, structural hashing and duplicate rejection directly on
//! the flat arena. A slot never moves, a program is never rebuilt from a tree,
//! and every random decision comes from [`crate::genome::Philox`] keyed by
//! `(seed, island, generation, slot)`, so a child is a pure function of its
//! position.
//!
//! It is deliberately a *host* implementation: the device evaluates the pool,
//! while variation, hashing and the archive test run on the CPU over compact
//! arrays. Porting the splice itself to CUDA remains the documented next step;
//! what this removes is the per-child tree allocation and full-tree
//! simplification that dominated the old loop.
//!
//! Semantics are not the same random trajectory as the `Node` path: the
//! generator is counter-based and the survivor rule is positional. Both paths
//! are deterministic; the path is chosen before the first generation and is
//! recorded in the backend note.

use crate::expr::{node, Node, Op};
use crate::genome::{postfix_hash, Philox, OP_HALT};
use crate::gpu;
use crate::pareto::Sol;
use std::collections::HashSet;

/// One parsed node inside a postfix program: its word span and arity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub arity: usize,
}

/// A flat, fixed-width population of `islands * pop` postfix programs.
///
/// Every program occupies exactly [`Arena::width`] words and is `HALT`-padded,
/// so `words[slot*width .. (slot+1)*width]` is a complete kernel program.
#[derive(Clone, Debug)]
pub struct Arena {
    pub width: usize,
    pub islands: usize,
    pub pop: usize,
    pub words: Vec<u32>,
    pub len: Vec<u16>,
    /// Default-model complexity (the node count).
    pub cost: Vec<u16>,
    pub hash: Vec<u64>,
    /// Search fitness: the f32 kernel SSE divided by the row count.
    pub err: Vec<f64>,
}

impl Arena {
    pub fn new(islands: usize, pop: usize, width: usize) -> Self {
        let n = islands * pop;
        Arena {
            width,
            islands,
            pop,
            words: vec![OP_HALT; n * width],
            len: vec![0; n],
            cost: vec![0; n],
            hash: vec![0; n],
            err: vec![f64::NAN; n],
        }
    }

    /// Host bytes reserved by the arena, including metadata. Capacity is fixed
    /// at construction, so this cannot grow across generations.
    pub fn bytes(&self) -> usize {
        self.words.len() * std::mem::size_of::<u32>()
            + self.len.len() * std::mem::size_of::<u16>()
            + self.cost.len() * std::mem::size_of::<u16>()
            + self.hash.len() * std::mem::size_of::<u64>()
            + self.err.len() * std::mem::size_of::<f64>()
    }

    pub fn slot_index(&self, island: usize, slot: usize) -> usize {
        island * self.pop + slot
    }

    pub fn slot(&self, index: usize) -> &[u32] {
        let base = index * self.width;
        &self.words[base..base + self.width]
    }

    /// Write one program into slot `index`. Rejects an over-wide or invalid
    /// program and leaves the slot untouched.
    pub fn set_slot(&mut self, index: usize, program: &[u32]) -> bool {
        if program.len() < 2 || program.len() > self.width {
            return false;
        }
        let Some(spans) = spans(program) else {
            return false;
        };
        let base = index * self.width;
        self.words[base..base + self.width].fill(OP_HALT);
        self.words[base..base + program.len()].copy_from_slice(program);
        self.len[index] = program.len() as u16;
        self.cost[index] = spans.len().min(u16::MAX as usize) as u16;
        self.hash[index] = postfix_hash(program);
        true
    }

    /// Copy slot `src` to slot `dst`, including its fitness.
    pub fn copy_slot(&mut self, dst: usize, src: usize) {
        if dst == src {
            return;
        }
        let (w, s) = (self.width, src * self.width);
        let d = dst * self.width;
        let program: Vec<u32> = self.words[s..s + w].to_vec();
        self.words[d..d + w].copy_from_slice(&program);
        self.len[dst] = self.len[src];
        self.cost[dst] = self.cost[src];
        self.hash[dst] = self.hash[src];
        self.err[dst] = self.err[src];
    }

    /// Copy slot `src` of `from` into slot `dst` of this arena, including its
    /// fitness. Used for elitism, where the parent and child arenas differ.
    pub fn copy_from(&mut self, from: &Arena, dst: usize, src: usize) {
        let w = self.width;
        let s = src * from.width;
        let d = dst * self.width;
        self.words[d..d + w].copy_from_slice(&from.words[s..s + w]);
        self.len[dst] = from.len[src];
        self.cost[dst] = from.cost[src];
        self.hash[dst] = from.hash[src];
        self.err[dst] = from.err[src];
    }

    /// The best slot of an island under the canonical `(error, cost, slot)`
    /// order. NaN errors sort last, so a population always has a winner.
    pub fn best(&self, island: usize) -> usize {
        let base = island * self.pop;
        (base..base + self.pop).min_by(|&a, &b| self.key(a).cmp(&self.key(b))).unwrap_or(base)
    }

    /// The worst slot of an island under the same order.
    pub fn worst(&self, island: usize) -> usize {
        let base = island * self.pop;
        (base..base + self.pop).max_by(|&a, &b| self.key(a).cmp(&self.key(b))).unwrap_or(base)
    }

    /// The canonical total order: finite error first, then cost, then slot.
    fn key(&self, index: usize) -> (u8, u64, u16, usize) {
        let e = self.err[index];
        // IEEE-754 total order on the bit pattern: finite values sort by
        // value, +inf above every finite error, NaN above everything.
        let bits = if e.is_nan() {
            0
        } else {
            let b = e.to_bits();
            if b & (1 << 63) != 0 {
                !b
            } else {
                b | (1 << 63)
            }
        };
        (if e.is_nan() { 1 } else { 0 }, bits, self.cost[index], index)
    }

    /// Rebuild the owned trees (and their configured costs) for checkpointing
    /// and reporting. `None` if a slot is not a valid postfix program, which
    /// cannot happen for a pool this module filled.
    pub fn to_trees(&self) -> Option<Vec<Vec<Node>>> {
        let mut out = vec![Vec::with_capacity(self.pop); self.islands];
        for i in 0..self.islands {
            for j in 0..self.pop {
                out[i].push(self.decode(self.slot_index(i, j))?);
            }
        }
        Some(out)
    }

    /// Rebuild one tree from slot `index`.
    pub fn decode(&self, index: usize) -> Option<Node> {
        decode(self.slot(index))
    }

    /// Build an arena from already simplified trees and their fitness. Every
    /// tree must compile to a program that fits a slot.
    pub fn from_trees(
        islands: &[Vec<Node>],
        errors: &[Vec<f64>],
        width: usize,
    ) -> Result<Arena, String> {
        let pop = islands.first().map(|p| p.len()).unwrap_or(0);
        let mut arena = Arena::new(islands.len(), pop, width);
        for (i, pop_i) in islands.iter().enumerate() {
            if pop_i.len() != pop {
                return Err("islands have different population sizes".to_string());
            }
            for (j, tree) in pop_i.iter().enumerate() {
                let idx = arena.slot_index(i, j);
                let mut words = Vec::new();
                if !gpu::compile(tree, &mut words) || !arena.set_slot(idx, &words) {
                    return Err(format!(
                        "tree {}:{} does not fit the postfix pool ({} words)",
                        i,
                        j,
                        width
                    ));
                }
                arena.err[idx] = errors[i][j];
            }
        }
        Ok(arena)
    }

    /// Device evaluation of the whole arena in one upload and one launch.
    /// Returns the mean squared error the search uses (SSE / rows).
    pub fn evaluate(
        &mut self,
        ev: &mut gpu::GpuEvaluator,
        n_rows: usize,
    ) -> Result<(), String> {
        let (sse, ms) = ev.eval_arena(&self.words, self.islands * self.pop, self.width);
        if ms < 0.0 || sse.len() != self.islands * self.pop {
            return Err("CUDA pool evaluation failed".to_string());
        }
        let n = n_rows.max(1) as f64;
        for (e, s) in self.err.iter_mut().zip(sse) {
            *e = if s.is_finite() { s as f64 / n } else { f64::INFINITY };
        }
        Ok(())
    }

    /// Ring migration: the best program of island `i` replaces the worst of
    /// island `i+1`. Slot order and the source fitness are preserved, so the
    /// next generation sees the same program the archive saw.
    pub fn migrate(&mut self) {
        if self.islands < 2 {
            return;
        }
        let bests: Vec<usize> = (0..self.islands).map(|i| self.best(i)).collect();
        let worsts: Vec<usize> = (0..self.islands).map(|i| self.worst(i)).collect();
        for i in 0..self.islands {
            let src = bests[(i + 1) % self.islands];
            self.copy_slot(worsts[i], src);
        }
    }
}

/// Parse a postfix program into its nodes. Returns `None` when the stack does
/// not collapse to exactly one value before `HALT`.
pub fn spans(program: &[u32]) -> Option<Vec<Span>> {
    let mut out: Vec<Span> = Vec::new();
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (start word, node)
    let mut i = 0usize;
    while i < program.len() {
        let w = program[i];
        if w == OP_HALT {
            break;
        }
        if w == gpu::OP_CONST || w == gpu::OP_VAR {
            if i + 1 >= program.len() || program[i + 1] == OP_HALT {
                return None;
            }
            let node_idx = out.len();
            out.push(Span {
                start: i,
                end: i + 2,
                arity: 0,
            });
            stack.push((i, node_idx));
            i += 2;
        } else {
            let arity = gpu::opcode_arity(w)?;
            if stack.len() < arity || arity == 0 {
                return None;
            }
            let (start, _) = *stack.get(stack.len() - arity)?;
            stack.truncate(stack.len() - arity);
            let node_idx = out.len();
            out.push(Span {
                start,
                end: i + 1,
                arity,
            });
            stack.push((start, node_idx));
            i += 1;
        }
    }
    if stack.len() != 1 || i == 0 {
        return None;
    }
    Some(out)
}

/// Rebuild an owned tree from a postfix program.
pub fn decode(program: &[u32]) -> Option<Node> {
    let mut stack: Vec<Node> = Vec::new();
    let mut i = 0usize;
    while i < program.len() {
        let w = program[i];
        if w == OP_HALT {
            break;
        }
        if w == gpu::OP_CONST {
            let bits = *program.get(i + 1)?;
            stack.push(crate::expr::constant(f32::from_bits(bits) as f64));
            i += 2;
        } else if w == gpu::OP_VAR {
            let index = *program.get(i + 1)? as usize;
            stack.push(crate::expr::input(index));
            i += 2;
        } else {
            let op = gpu::opcode_op(w)?;
            let arity = gpu::opcode_arity(w)?;
            if stack.len() < arity {
                return None;
            }
            let kids = stack.split_off(stack.len() - arity);
            stack.push(node(op, kids));
            i += 1;
        }
    }
    if stack.len() == 1 {
        stack.pop()
    } else {
        None
    }
}

/// Archive-guided parent pool for the flat path (#76 parity, #122).
///
/// The owned-tree loop draws 15 % of parents from the previous generation's
/// Pareto front. The flat path cannot hold `Sol` trees in a slot, so the front
/// is compiled once per generation into fixed-width postfix programs. Only
/// programs that fit the slot and the complexity bound are kept; an empty pool
/// behaves exactly like a disabled feature and consumes no random draw.
#[derive(Clone, Debug, Default)]
pub struct FrontParents {
    programs: Vec<Vec<u32>>,
}

impl FrontParents {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Compile the archive front for the next generation. Trees the kernel
    /// cannot emit (or that do not fit `width`) are skipped, exactly as the
    /// flat path itself declines such a configuration as a whole.
    pub fn from_front(front: &[Sol], width: usize, max_complexity: usize) -> Self {
        let mut programs = Vec::with_capacity(front.len());
        for sol in front {
            let mut words = Vec::new();
            if !gpu::compile(&sol.tree, &mut words) || words.len() > width {
                continue;
            }
            let Some(spans) = spans(&words) else { continue };
            if spans.len() > max_complexity {
                continue;
            }
            programs.push(words);
        }
        Self { programs }
    }

    pub fn len(&self) -> usize {
        self.programs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.programs.is_empty()
    }

    pub fn program(&self, index: usize) -> &[u32] {
        &self.programs[index]
    }
}

/// #76 parity: with probability `p`, pick a front program; otherwise `None`.
///
/// The order of the conditions matters. A disabled feature (`p == 0`) or an
/// empty front returns without touching the RNG, which is what makes
/// `--no-front-parents` bit-identical to the pre-#76 search.
fn front_pick(front: &FrontParents, rng: &mut Philox, p: f64) -> Option<usize> {
    if p > 0.0 && !front.is_empty() && rng.unit() < p {
        Some(rng.below(front.len()))
    } else {
        None
    }
}

/// The canonical `(hash, slot)` order used by dedup and compaction.
pub fn canonical_order(arena: &Arena) -> Vec<usize> {
    let mut order: Vec<usize> = (0..arena.islands * arena.pop).collect();
    order.sort_by_key(|&i| (arena.hash[i], i));
    order
}

/// Indexes of the first slot for each distinct program, in slot order. A hash
/// collision is resolved by comparing the full programs, exactly as
/// [`crate::genome::mark_duplicates`] does.
pub fn unique_slots(arena: &Arena) -> Vec<usize> {
    let order = canonical_order(arena);
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    for &i in &order {
        if let Some(&prev) = keep.last() {
            if arena.hash[prev] == arena.hash[i] && arena.slot(prev) == arena.slot(i) {
                continue;
            }
        }
        keep.push(i);
    }
    keep.sort_unstable();
    keep
}

/// One deterministic variation engine over a fixed-width postfix arena.
#[derive(Clone, Debug)]
pub struct Variation {
    pub width: usize,
    pub nvars: usize,
    pub max_complexity: usize,
    pub seed: u32,
    pub init_depth: usize,
    pub dedup: bool,
    /// `(op, opcode, arity)` for every operator the kernel can emit.
    ops: Vec<(Op, u32, usize)>,
    /// The same operators grouped by arity, for point mutation.
    by_arity: [Vec<usize>; 4],
}

impl Variation {
    /// Build an engine, rejecting an operator set the kernel cannot emit. The
    /// caller uses that `Err` to stay on the `Node` path.
    pub fn new(
        width: usize,
        nvars: usize,
        max_complexity: usize,
        seed: u32,
        ops: &[Op],
        init_depth: usize,
        dedup: bool,
    ) -> Result<Self, String> {
        let mut table = Vec::new();
        for &op in ops {
            if matches!(op, Op::Const | Op::Input) {
                continue;
            }
            let (code, arity) = gpu::postfix_opcode(op)
                .ok_or_else(|| format!("operator '{}' has no kernel opcode", op.symbol()))?;
            if arity == 0 || arity > 3 {
                return Err(format!("operator '{}' has unsupported arity", op.symbol()));
            }
            if !table.iter().any(|&(o, _, _)| o == op) {
                table.push((op, code, arity));
            }
        }
        if table.is_empty() {
            return Err("no nullary-capable operator set".to_string());
        }
        let mut by_arity: [Vec<usize>; 4] = Default::default();
        for (i, &(_, _, arity)) in table.iter().enumerate() {
            by_arity[arity].push(i);
        }
        if by_arity[1].is_empty() && by_arity[2].is_empty() && by_arity[3].is_empty() {
            return Err("operator set contains no non-terminal operator".to_string());
        }
        Ok(Variation {
            width,
            nvars,
            max_complexity,
            seed,
            init_depth,
            dedup,
            ops: table,
            by_arity,
        })
    }

    /// The ordered operator table, for the CLI/schema report and tests.
    pub fn operators(&self) -> &[(Op, u32, usize)] {
        &self.ops
    }

    /// Generate the next generation from `parents`.
    ///
    /// `restart[i]` replaces island `i` with fresh random programs (the
    /// stagnation policy). Every child is a pure function of
    /// `(seed, island, generation, slot)`; the dedup set only decides whether
    /// the next attempt at that slot is used, and parent order is fixed, so
    /// the result never depends on thread or island scheduling.
    pub fn next_generation(&self, parents: &Arena, gen: u32, restart: &[bool]) -> Arena {
        self.next_generation_with_front(parents, gen, restart, &FrontParents::empty(), 0.0)
    }

    /// Generate the next generation with the archive-guided parent draw
    /// (#76 parity). `front_p` is the fraction of parent draws taken uniformly
    /// from `front`; the search passes 0.15 when `--front-parents` is on and
    /// 0.0 (with an empty pool and no RNG advance) when it is off.
    pub fn next_generation_with_front(
        &self,
        parents: &Arena,
        gen: u32,
        restart: &[bool],
        front: &FrontParents,
        front_p: f64,
    ) -> Arena {
        let mut attempts = 0usize;
        let mut scratch = Scratch::new(self.width);
        let mut next = Arena::new(parents.islands, parents.pop, self.width);
        let mut seen: HashSet<u64> = HashSet::with_capacity(parents.islands * parents.pop * 2);
        for i in 0..parents.islands {
            for j in 0..parents.pop {
                seen.insert(parents.hash[parents.slot_index(i, j)]);
            }
        }
        for i in 0..parents.islands {
            let restart_island = restart.get(i).copied().unwrap_or(false);
            if restart_island {
                for j in 0..parents.pop {
                    self.fill_random(&mut next, i, j, gen, 0);
                }
                continue;
            }
            // Elitism: the best parent survives unchanged (and is already in
            // the dedup set).
            let elite = parents.best(i);
            next.copy_from(parents, next.slot_index(i, 0), elite);
            for j in 1..parents.pop {
                self.fill_child(
                    &mut next,
                    parents,
                    &mut seen,
                    i,
                    j,
                    gen,
                    &mut attempts,
                    &mut scratch,
                    front,
                    front_p,
                );
            }
        }
        next
    }

    /// Replace an entire island with fresh random programs (the stagnation
    /// policy). The caller evaluates the arena afterwards.
    pub fn restart_island(&self, arena: &mut Arena, island: usize, gen: u32) {
        for j in 0..arena.pop {
            self.fill_random(arena, island, j, gen, 1);
        }
    }

    fn fill_random(&self, next: &mut Arena, i: usize, j: usize, gen: u32, salt: u64) {
        let mut rng = Philox::at(self.seed, i as u32, gen, j as u32);
        // Salt lets a fallback draw a different stream without changing the
        // documented positional contract: it advances the draw index.
        for _ in 0..salt {
            rng.next();
        }
        let mut best: Option<Vec<u32>> = None;
        for _ in 0..8 {
            let prog = self.random_program(&mut rng, self.init_depth);
            if let Some(spans) = spans(&prog) {
                if spans.len() <= self.max_complexity && prog.len() <= self.width {
                    best = Some(prog);
                    break;
                }
            }
        }
        let prog = best.unwrap_or_else(|| vec![crate::gpu::OP_VAR, 0, OP_HALT]);
        let idx = next.slot_index(i, j);
        next.set_slot(idx, &prog);
        next.err[idx] = f64::NAN; // scored by the caller's evaluation batch
    }

    fn fill_child(
        &self,
        next: &mut Arena,
        parents: &Arena,
        seen: &mut HashSet<u64>,
        island: usize,
        slot: usize,
        gen: u32,
        attempts: &mut usize,
        scratch: &mut Scratch,
        front: &FrontParents,
        front_p: f64,
    ) {
        let max_attempts = 64;
        for attempt in 0..max_attempts {
            *attempts += 1;
            let mut rng = Philox::at(self.seed, island as u32, gen, slot as u32);
            if attempt > 0 {
                rng.set_draw(attempt as u32);
            }
            // #76: the first parent may come from the archive front. With the
            // feature off, `front_pick` returns without drawing, so the stream
            // is exactly the pre-#76 one.
            scratch.program.clear();
            if let Some(fi) = front_pick(front, &mut rng, front_p) {
                let program = front.program(fi);
                scratch
                    .program
                    .extend_from_slice(&program[..logical_len(program)]);
            } else {
                let src = parents.slot(parents.tournament(island, &mut rng, 4));
                let end = logical_len(src);
                scratch.program.extend_from_slice(&src[..end]);
            }
            if rng.unit() < 0.5 {
                if let Some(fi) = front_pick(front, &mut rng, front_p) {
                    let other = front.program(fi);
                    let other = &other[..logical_len(other)];
                    self.crossover_into(
                        &mut scratch.program,
                        other,
                        &mut rng,
                        &mut scratch.spans,
                        &mut scratch.stack,
                        &mut scratch.other_spans,
                        &mut scratch.other_stack,
                    );
                } else {
                    let other = parents.tournament(island, &mut rng, 4);
                    let other = parents.slot(other);
                    let other = &other[..logical_len(other)];
                    self.crossover_into(
                        &mut scratch.program,
                        other,
                        &mut rng,
                        &mut scratch.spans,
                        &mut scratch.stack,
                        &mut scratch.other_spans,
                        &mut scratch.other_stack,
                    );
                }
            }
            self.mutate_into(&mut scratch.program, &mut scratch.spans, &mut scratch.stack, &mut scratch.tmp, &mut rng);
            if rng.unit() < 0.5 {
                self.mutate_into(&mut scratch.program, &mut scratch.spans, &mut scratch.stack, &mut scratch.tmp, &mut rng);
            }
            let program = &scratch.program;
            if program.len() > self.width
                || !scan_into(program, &mut scratch.spans, &mut scratch.stack)
            {
                continue;
            }
            if scratch.spans.len() > self.max_complexity {
                continue;
            }
            let h = postfix_hash(program);
            if self.dedup && !seen.insert(h) {
                continue;
            }
            let idx = next.slot_index(island, slot);
            if next.set_slot(idx, program) {
                return;
            }
        }
        self.fill_random(next, island, slot, gen, max_attempts as u64);
    }

    fn crossover_into(
        &self,
        program: &mut Vec<u32>,
        other: &[u32],
        rng: &mut Philox,
        spans: &mut Vec<Span>,
        stack: &mut Vec<(usize, usize)>,
        other_spans: &mut Vec<Span>,
        other_stack: &mut Vec<(usize, usize)>,
    ) {
        if !scan_into(program, spans, stack) {
            return;
        }
        if !scan_into(other, other_spans, other_stack) {
            return;
        }
        let a: &[Span] = spans;
        let b: &[Span] = other_spans;
        if a.is_empty() || b.is_empty() {
            return;
        }
        let ai = rng.below(a.len());
        let bi = rng.below(b.len());
        let (sa, sb) = (a[ai], b[bi]);
        let replacement = &other[sb.start..sb.end];
        splice(program, sa, replacement, self.width);
    }

    /// Apply one edit: point operator change (25 %), terminal resample (35 %)
    /// or subtree replacement (40 %).
    fn mutate_into(
        &self,
        program: &mut Vec<u32>,
        spans: &mut Vec<Span>,
        stack: &mut Vec<(usize, usize)>,
        tmp: &mut Vec<u32>,
        rng: &mut Philox,
    ) {
        if !scan_into(program, spans, stack) || spans.is_empty() {
            return;
        }
        let r = rng.unit();
        let order: [u8; 3] = if r < 0.25 {
            [0, 2, 1]
        } else if r < 0.6 {
            [1, 0, 2]
        } else {
            [2, 0, 1]
        };
        for kind in order {
            if self.edit(program, spans, tmp, kind, rng) {
                return;
            }
        }
    }

    fn edit(
        &self,
        program: &mut Vec<u32>,
        spans: &[Span],
        tmp: &mut Vec<u32>,
        kind: u8,
        rng: &mut Philox,
    ) -> bool {
        // Count the eligible nodes first and make exactly one draw. Drawing
        // per candidate (reservoir style) would run Philox dozens of times per
        // edit and dominated the variation cost.
        let eligible = |span: &Span| match kind {
            0 => span.arity > 0,
            1 => span.arity == 0,
            _ => true,
        };
        let count = spans.iter().filter(|s| eligible(s)).count();
        if count == 0 {
            return false;
        }
        let target = rng.below(count);
        let node_i = spans
            .iter()
            .enumerate()
            .filter(|(_, s)| eligible(s))
            .nth(target)
            .map(|(i, _)| i)
            .unwrap();
        let span = spans[node_i];
        match kind {
            0 => {
                let arity = span.arity;
                let pool = &self.by_arity[arity.min(3)];
                if pool.len() < 2 {
                    return false;
                }
                let current = program[span.end - 1];
                let count = pool
                    .iter()
                    .filter(|&&k| self.ops[k].1 != current)
                    .count();
                if count == 0 {
                    return false;
                }
                let pick = pool
                    .iter()
                    .map(|&k| self.ops[k].1)
                    .filter(|&code| code != current)
                    .nth(rng.below(count))
                    .unwrap();
                program[span.end - 1] = pick;
                true
            }
            1 => {
                let term = self.random_terminal(rng);
                program[span.start] = term[0];
                program[span.start + 1] = term[1];
                true
            }
            _ => {
                let budget = self.width.saturating_sub(1) - (program.len() - span.end);
                if budget < 2 {
                    return false;
                }
                tmp.clear();
                self.gen_subtree(rng, self.init_depth, budget, tmp);
                if span.end - span.start + tmp.len() > self.width {
                    return false;
                }
                splice(program, span, tmp, self.width)
            }
        }
    }

    fn random_terminal(&self, rng: &mut Philox) -> [u32; 2] {
        if self.nvars > 0 && rng.unit() < 0.65 {
            [crate::gpu::OP_VAR, (rng.below(self.nvars)) as u32]
        } else {
            let v = ((rng.unit() * 4.0 - 2.0) * 1000.0).round() / 1000.0;
            [crate::gpu::OP_CONST, (v as f32).to_bits()]
        }
    }

    fn random_program(&self, rng: &mut Philox, depth: usize) -> Vec<u32> {
        let budget = self.max_complexity.min((self.width - 1) / 2).max(1);
        let mut out = Vec::new();
        self.gen_subtree(rng, depth, budget, &mut out);
        out.push(OP_HALT);
        out
    }

    /// Append a valid postfix subtree of at most `budget` nodes and `depth`
    /// levels. Terminals cost two words and one node, operators one word and
    /// one node. The output buffer is reused, so a mutation allocates nothing.
    fn gen_subtree(&self, rng: &mut Philox, depth: usize, budget: usize, out: &mut Vec<u32>) {
        if depth == 0 || budget <= 1 || rng.unit() < 0.15 {
            let term = self.random_terminal(rng);
            out.extend_from_slice(&term);
            return;
        }
        let arity = if rng.unit() < 0.35 {
            1
        } else if rng.unit() < 0.75 {
            2
        } else {
            3
        };
        let pool = &self.by_arity[arity];
        if pool.is_empty() || budget < arity + 1 {
            let term = self.random_terminal(rng);
            out.extend_from_slice(&term);
            return;
        }
        let &(_, code, _) = &self.ops[pool[rng.below(pool.len())]];
        let remaining = budget - 1 - arity;
        for k in 0..arity {
            let left = arity - k;
            let share = (remaining / left).max(1);
            self.gen_subtree(rng, depth - 1, share, out);
        }
        out.push(code);
    }
}

/// Reused buffers for one child. Keeping them here removes the per-child
/// `Vec` allocations that made the first version no faster than the tree loop.
struct Scratch {
    program: Vec<u32>,
    spans: Vec<Span>,
    stack: Vec<(usize, usize)>,
    other_spans: Vec<Span>,
    other_stack: Vec<(usize, usize)>,
    tmp: Vec<u32>,
}

impl Scratch {
    fn new(width: usize) -> Self {
        Scratch {
            program: Vec::with_capacity(width),
            spans: Vec::with_capacity(width),
            stack: Vec::with_capacity(width),
            other_spans: Vec::with_capacity(width),
            other_stack: Vec::with_capacity(width),
            tmp: Vec::with_capacity(width),
        }
    }
}

/// Number of words up to and including the first `HALT`.
fn logical_len(program: &[u32]) -> usize {
    program
        .iter()
        .position(|&w| w == OP_HALT)
        .map(|i| i + 1)
        .unwrap_or(program.len())
}

/// Like [`spans`], but writes into caller-owned buffers.
fn scan_into(program: &[u32], out: &mut Vec<Span>, stack: &mut Vec<(usize, usize)>) -> bool {
    out.clear();
    stack.clear();
    let mut saw_halt = false;
    let mut i = 0usize;
    while i < program.len() {
        let w = program[i];
        if w == OP_HALT {
            saw_halt = true;
            break;
        }
        if w == crate::gpu::OP_CONST || w == crate::gpu::OP_VAR {
            if i + 1 >= program.len() {
                return false;
            }
            out.push(Span {
                start: i,
                end: i + 2,
                arity: 0,
            });
            stack.push((i, out.len() - 1));
            i += 2;
        } else {
            let Some(arity) = crate::gpu::opcode_arity(w) else {
                return false;
            };
            if arity == 0 || stack.len() < arity {
                return false;
            }
            let (start, _) = stack[stack.len() - arity];
            stack.truncate(stack.len() - arity);
            out.push(Span {
                start,
                end: i + 1,
                arity,
            });
            stack.push((start, out.len() - 1));
            i += 1;
        }
    }
    saw_halt && stack.len() == 1 && !out.is_empty()
}

/// Replace `span` in `program` with `replacement` without allocating.
fn splice(program: &mut Vec<u32>, span: Span, replacement: &[u32], width: usize) -> bool {
    let old_len = program.len();
    let new_len = old_len - (span.end - span.start) + replacement.len();
    if new_len > width || new_len < 2 {
        return false;
    }
    if new_len > old_len {
        program.resize(new_len, OP_HALT);
    }
    program.copy_within(span.end..old_len, span.start + replacement.len());
    program[span.start..span.start + replacement.len()].copy_from_slice(replacement);
    if new_len < old_len {
        program.truncate(new_len);
    }
    true
}

impl Arena {
    /// Tournament selection by `(error, cost, slot)`. `k` draws with the same
    /// counter stream, so the winner is a pure function of the position.
    pub fn tournament(&self, island: usize, rng: &mut Philox, k: usize) -> usize {
        let base = island * self.pop;
        let mut best = base + rng.below(self.pop);
        for _ in 1..k.max(1) {
            let c = base + rng.below(self.pop);
            if self.key(c) < self.key(best) {
                best = c;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input};

    fn solution(tree: Node, error: f64, cost: f64) -> Sol {
        Sol {
            complexity: crate::expr::complexity(&tree),
            cost,
            tree,
            error,
        }
    }

    fn arena_of(programs: &[Vec<u32>], islands: usize, pop: usize, width: usize) -> Arena {
        let mut a = Arena::new(islands, pop, width);
        for (i, p) in programs.iter().enumerate() {
            assert!(a.set_slot(i, p));
        }
        a
    }

    #[test]
    fn roundtrip_preserves_the_program() {
        let tree = node(
            Op::Add,
            vec![input(0), node(Op::Mul, vec![constant(2.5), input(1)])],
        );
        let mut words = Vec::new();
        assert!(gpu::compile(&tree, &mut words));
        let back = decode(&words).expect("decode");
        let mut again = Vec::new();
        assert!(gpu::compile(&back, &mut again));
        assert_eq!(words, again);
    }

    #[test]
    fn spans_reject_malformed_programs() {
        assert!(spans(&[crate::gpu::OP_ADD, OP_HALT]).is_none());
        assert!(spans(&[crate::gpu::OP_VAR, 0, OP_HALT]).is_some());
        assert!(spans(&[OP_HALT]).is_none());
    }

    #[test]
    fn variation_is_bounded_and_valid() {
        let v = Variation::new(64, 2, 30, 7, &Op::default_set(), 3, true).unwrap();
        let mut parents = Arena::new(3, 32, 64);
        let mut rng = Philox::at(7, 0, 0, 0);
        for i in 0..parents.islands * parents.pop {
            let mut p = v.random_program(&mut rng, 4);
            while spans(&p).is_none() {
                p = v.random_program(&mut rng, 4);
            }
            parents.set_slot(i, &p);
            parents.err[i] = (i as f64 * 0.5).sin().abs();
        }
        let next = v.next_generation(&parents, 1, &[false, false, false]);
        for i in 0..next.islands * next.pop {
            assert!(next.len[i] as usize <= 64);
            assert!(spans(next.slot(i)).is_some(), "slot {} is malformed", i);
            assert!(next.cost[i] as usize <= 30);
            assert!(next.decode(i).is_some());
        }
    }

    #[test]
    fn the_same_position_gives_the_same_child() {
        let v = Variation::new(64, 2, 30, 9, &Op::default_set(), 3, true).unwrap();
        let mut parents = Arena::new(4, 40, 64);
        let mut rng = Philox::at(9, 0, 0, 0);
        for i in 0..parents.islands * parents.pop {
            let mut p = v.random_program(&mut rng, 3);
            while spans(&p).is_none() {
                p = v.random_program(&mut rng, 3);
            }
            parents.set_slot(i, &p);
            parents.err[i] = ((i * 37) % 101) as f64 / 101.0;
        }
        let a = v.next_generation(&parents, 3, &[false; 4]);
        let b = v.next_generation(&parents, 3, &[false; 4]);
        assert_eq!(a.words, b.words);
        assert_eq!(a.hash, b.hash);
        for (x, y) in a.err.iter().zip(&b.err) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn islands_do_not_depend_on_how_many_are_evolved() {
        // Evolving islands 0..1 alone must produce exactly the same slots as
        // evolving them inside a four-island arena: there is no shared RNG,
        // hash set or slot cursor.
        let v = Variation::new(64, 2, 30, 4, &Op::default_set(), 3, true).unwrap();
        let mut all = Arena::new(4, 24, 64);
        let mut rng = Philox::at(4, 0, 0, 0);
        for i in 0..all.islands * all.pop {
            let mut p = v.random_program(&mut rng,  NM);
            while spans(&p).is_none() {
                p = v.random_program(&mut rng, NM);
            }
            all.set_slot(i, &p);
            all.err[i] = ((i * 13) % 97) as f64 / 97.0;
        }
        let four = v.next_generation(&all, 2, &[false; 4]);
        let mut one = Arena::new(1, 24, 64);
        for j in 0..24 {
            let s = all.slot_index(0, j);
            one.copy_from(&all, j, s);
        }
        let solo = v.next_generation(&one, 2, &[false]);
        for j in 0..24 {
            assert_eq!(four.slot(four.slot_index(0, j)), solo.slot(j));
            let (a, b) = (four.err[four.slot_index(0, j)], solo.err[j]);
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    const NM: usize = 3;

    #[test]
    fn front_draw_matches_the_owned_tree_contract() {
        let tree = node(Op::Add, vec![input(0), constant(2.0)]);
        let front = FrontParents::from_front(&[solution(tree, 0.5, 3.0)], 64, 30);
        assert_eq!(front.len(), 1);

        // Off (`p == 0`): no draw, exactly like `front_parent(..., 0.0)`.
        let mut rng = Philox::at(1, 0, 0, 0);
        let before = rng.counter();
        assert_eq!(front_pick(&front, &mut rng, 0.0), None);
        assert_eq!(rng.counter(), before);

        // Empty front: no draw either, even when the feature is enabled.
        let empty = FrontParents::empty();
        let mut rng = Philox::at(1, 0, 0, 0);
        let before = rng.counter();
        assert_eq!(front_pick(&empty, &mut rng, 0.15), None);
        assert_eq!(rng.counter(), before);

        // On (`p == 1`): one Bernoulli draw plus one index draw.
        let mut rng = Philox::at(1, 0, 0, 0);
        assert_eq!(front_pick(&front, &mut rng, 1.0), Some(0));

        // The 15 % rate is a property of the draw, not of the caller.
        let mut hits = 0;
        for i in 0..10_000u32 {
            let mut rng = Philox::at(7, 0, i / 100, i % 100);
            if front_pick(&front, &mut rng, 0.15).is_some() {
                hits += 1;
            }
        }
        assert!(
            (1_300..=1_700).contains(&hits),
            "15% draw produced {hits}/10000"
        );
    }

    #[test]
    fn front_parents_are_deterministic_and_leave_the_off_path_unchanged() {
        let v = Variation::new(64, 2, 30, 9, &Op::default_set(), 3, true).unwrap();
        let mut parents = Arena::new(2, 64, 64);
        let mut rng = Philox::at(9, 0, 0, 0);
        for i in 0..parents.islands * parents.pop {
            let mut p = v.random_program(&mut rng, 3);
            while spans(&p).is_none() {
                p = v.random_program(&mut rng, 3);
            }
            parents.set_slot(i, &p);
            parents.err[i] = ((i * 37) % 101) as f64 / 101.0;
        }
        let front = FrontParents::from_front(
            &[solution(
                node(Op::Add, vec![input(0), constant(7.0)]),
                0.25,
                3.0,
            )],
            64,
            30,
        );
        let a = v.next_generation_with_front(&parents, 5, &[false; 2], &front, 0.15);
        let b = v.next_generation_with_front(&parents, 5, &[false; 2], &front, 0.15);
        assert_eq!(a.words, b.words);
        assert_eq!(a.hash, b.hash);

        // `--no-front-parents` must be bit-identical to the pre-#76 path.
        let off = v.next_generation(&parents, 5, &[false; 2]);
        let off_explicit =
            v.next_generation_with_front(&parents, 5, &[false; 2], &FrontParents::empty(), 0.0);
        assert_eq!(off.words, off_explicit.words);
        assert_eq!(off.hash, off_explicit.hash);
        assert_ne!(a.words, off.words, "the feedback must change some children");
    }

    #[test]
    fn ten_thousand_slots_have_a_fixed_memory_bound() {
        let a = Arena::new(1, 10_000, 64);
        assert_eq!(a.words.len() * 4, 2_560_000);
        assert_eq!(
            a.bytes(),
            2_560_000 + 10_000 * (2 + 2 + 8 + 8)
        );
        // Building the next generation allocates the same fixed capacity, not
        // a per-child tree.
        let v = Variation::new(64, 2, 30, 1, &Op::default_set(), 3, true).unwrap();
        let b = v.next_generation(&a, 0, &[false]);
        assert_eq!(b.bytes(), a.bytes());
    }

    #[test]
    fn dedup_is_stable_and_keeps_the_first_slot() {
        let a = arena_of(
            &[
                vec![crate::gpu::OP_VAR, 0, OP_HALT],
                vec![crate::gpu::OP_VAR, 0, OP_HALT],
                vec![crate::gpu::OP_VAR, 1, OP_HALT],
            ],
            1,
            3,
            8,
        );
        assert_eq!(unique_slots(&a), vec![0, 2]);
    }

    #[test]
    fn migration_moves_the_best_into_the_next_island() {
        let mut a = Arena::new(2, 3, 8);
        for i in 0..6 {
            a.set_slot(i, &[crate::gpu::OP_VAR, i as u32, OP_HALT]);
            a.err[i] = i as f64;
        }
        // Best of island 0 is slot 0; worst of island 1 is slot 5.
        a.migrate();
        assert_eq!(a.slot(5), a.slot(0));
        assert_eq!(a.err[5], 0.0);
    }
}
