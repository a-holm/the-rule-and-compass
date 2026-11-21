//! #1258: the device arm of `rcsr enumerate` against the CPU book on the
//! device's own f32 predictions. Under compute-sanitizer
//! (`RCSR_UNDER_SANITIZER`) the batches hold at most 8 trees and the grids at
//! most 16 cuts. Device-dependent tests skip cleanly without CUDA.

use super::*;
use crate::data::TradingSlice;
use crate::enumerate::{book_of, device_members, device_pass, grid, members, pass, shifted_slice, BookParams, Class, MemberCtx, Outcome, Pass, Scan, Side};
use crate::expr::Op;
use crate::trading::{BlockStat, TradingRows, TradingSpec};

const OPS10: &str = "add,sub,mul,div,abs,exp,square,tanh,logistic,gauss";

fn ops(names: &str) -> Vec<Op> {
    Op::parse_list_checked(names).unwrap()
}

fn sanitized() -> bool {
    std::env::var("RCSR_UNDER_SANITIZER").is_ok()
}

fn device() -> bool {
    if gpu::probe().available() {
        true
    } else {
        crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        false
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// 3 x 8,192 + 77 rows in five groups. x1 is noise with a tail where
/// `exp(exp(x1))` overflows in f32; x2 takes three values, 0 among them, so
/// `div(_, x2)` is not finite and trees of x2 alone have fewer than 16
/// distinct values; x3 takes five neighbouring f32 values, so its midpoint cuts
/// lie halfway between neighbouring f32 values, where an f32 comparison
/// differs from the f64 one; x4 holds one value, so its trees are constant.
fn panel() -> (Vec<Vec<f64>>, TradingSlice) {
    let sizes = [6_000usize, 5_000, 7_000, 3_653, 3_000];
    let n: usize = sizes.iter().sum();
    let mut rng = Rng(1258);
    let (mut x1, mut x2, mut x3, x4) = (vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n], vec![1.5f64; n]);
    let (mut time, mut group, mut ret) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut i = 0usize;
    for (g, &len) in sizes.iter().enumerate() {
        for k in 0..len {
            x1[i] = if i % 997 == 0 { 6.0 + (i % 5) as f64 } else { rng.gauss() };
            x2[i] = [-1.0, 0.0, 2.0][(i * 7 + i / 13) % 3];
            x3[i] = f32::from_bits(1.5f32.to_bits() + ((i * 11) % 5) as u32) as f64;
            time.push(3600.0 * k as f64);
            group.push(g as u64);
            ret.push(0.004 * rng.gauss() + if x1[i] > x2[i] { 0.001 } else { -0.001 });
            i += 1;
        }
    }
    (vec![x1, x2, x3, x4], TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret })
}

/// The observed book and `count - 1` shifted copies.
fn books(slice: &TradingSlice, count: usize, threshold: f64, blocks: usize) -> Vec<FitnessBook> {
    let spec = TradingSpec { cost_bps: 5.0, min_trades: 1, threshold, ..TradingSpec::default() };
    let params = BookParams { spec: &spec, horizon_seconds: 0.0, blocks, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] };
    let mut stamps: Vec<u64> = slice.rows.time.iter().map(|t| t.to_bits()).collect();
    stamps.sort_unstable();
    stamps.dedup();
    (0..count as u64)
        .map(|k| match k {
            0 => book_of(slice, &params).unwrap(),
            _ => book_of(&shifted_slice(slice, 37 * k + 11, stamps.len() as u64).unwrap(), &params).unwrap(),
        })
        .collect()
}

fn same_cut(a: &Cut, b: &Cut) -> bool {
    a.side == b.side && a.value.to_bits() == b.value.to_bits() && a.share.map(f64::to_bits) == b.share.map(f64::to_bits)
}

fn same_outcome(a: &Outcome, b: &Outcome) -> bool {
    match (a, b) {
        (Outcome::Scored { fitness: f, cut: c, cuts: k }, Outcome::Scored { fitness: g, cut: d, cuts: m }) => f.to_bits() == g.to_bits() && same_cut(c, d) && k == m,
        _ => a == b,
    }
}

fn same_pass(a: &Pass, b: &Pass) -> Result<(), String> {
    if a.best.iter().map(|f| f.to_bits()).ne(b.best.iter().map(|f| f.to_bits())) {
        return Err(format!("per-book minima differ: {:?} vs {:?}", a.best, b.best));
    }
    if a.outcomes.len() != b.outcomes.len() {
        return Err("outcome counts differ".to_string());
    }
    for (pos, (x, y)) in a.outcomes.iter().zip(&b.outcomes).enumerate() {
        if !same_outcome(x, y) {
            return Err(format!("tree {pos}: {x:?} vs {y:?}"));
        }
    }
    if a.books != b.books {
        return Err(format!("book keys differ: {} vs {}", a.books.len(), b.books.len()));
    }
    Ok(())
}

/// Every (tree, cut, book) total of the device equals `BookFamily::cut_totals`
/// (bit for bit `FitnessBook::block_totals` of the explicit prediction) on the
/// widened f32 predictions of the device, and every cut list is `grid` of them.
#[test]
fn device_enumerate_totals_equal_the_cpu_book() {
    if !device() {
        return;
    }
    let (cols, slice) = panel();
    let n = slice.ret.len();
    assert_eq!(n, 3 * 8_192 + 77);
    let class = Class::build(4, &ops(OPS10), 3, true).unwrap();
    assert_eq!(class.scored().len(), 76);
    let nodes: Vec<crate::expr::Node> = class.scored().iter().map(|&e| class.node(e as usize)).collect();
    let names: Vec<String> = ["x1", "x2", "x3", "x4"].iter().map(|s| s.to_string()).collect();
    let cuts_list: &[usize] = if sanitized() { &[2, 16] } else { &[2, 16, 64] };
    let batch = if sanitized() { 8 } else { 64 };
    let (mut compared, mut host, mut constant, mut nonfinite) = (0usize, 0usize, 0usize, 0usize);
    for blocks in [1usize, 4, 9] {
        for threshold in [0.0, 1e-30] {
            let family = books(&slice, 3, threshold, blocks);
            let refs: Vec<&FitnessBook> = family.iter().collect();
            let fam = BookFamily::new(refs.clone()).unwrap();
            for &q in cuts_list {
                let opts = DeviceOptions { max_batch: batch, ..DeviceOptions::default() };
                let mut en = Enumerator::new(&cols, n, &refs, q, class.max_size(), nodes.len(), &opts).expect("the enumerator opens");
                assert_eq!(en.batch(), batch.min(nodes.len()));
                en.set_books(&refs).unwrap();
                for chunk in nodes.chunks(en.batch()) {
                    let grids = en.eval(chunk).expect("the batch evaluates");
                    let totals = en.score(&grids, refs.len()).expect("the batch scores");
                    for (i, g) in grids.iter().enumerate() {
                        let label = format!("{blocks} blocks, threshold {threshold}, q {q}, tree {}", crate::expr::to_string(&chunk[i], &names));
                        let wide: Vec<f64> = en.predictions_for_tests(i).unwrap().iter().map(|&v| v as f64).collect();
                        if wide.iter().any(|v| !v.is_finite()) {
                            assert_eq!(*g, TreeGrid::NonFinite, "{label}");
                            nonfinite += 1;
                            continue;
                        }
                        let want = grid(&wide, q, &mut Vec::new());
                        let TreeGrid::Cuts { cuts, host: h } = g else { panic!("{label}: a finite tree is not cut") };
                        assert_eq!(cuts.len(), want.len(), "{label}: cut count");
                        for (a, b) in cuts.iter().zip(&want) {
                            assert!(same_cut(a, b), "{label}: cut {a:?} vs {b:?}");
                        }
                        if cuts.is_empty() {
                            assert!(totals[i].is_empty());
                            constant += 1;
                            continue;
                        }
                        host += *h as usize;
                        let consts: Vec<(bool, f64)> = cuts.iter().map(|c| (c.side == Side::Above, c.constant(threshold))).collect();
                        let mut reference = Vec::new();
                        fam.cut_totals(&wide, &consts, &mut crate::trading::CutScratch::default(), &mut reference);
                        assert_eq!(totals[i].len(), reference.len(), "{label}: totals");
                        for (j, (d, r)) in totals[i].iter().zip(&reference).enumerate() {
                            assert_eq!(
                                (d.gross.to_bits(), d.flips, d.exits, d.long_rows, d.nonfinite),
                                (r.gross.to_bits(), r.flips, r.exits, r.long_rows, r.nonfinite),
                                "{label}: total {j} (book {}, cut {}, block {}): {d:?} vs {r:?}",
                                j / (cuts.len() * blocks),
                                (j / blocks) % cuts.len(),
                                j % blocks
                            );
                        }
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 0 && host > 0 && constant > 0 && nonfinite > 0, "compared {compared}, host grids {host}, constant {constant}, non-finite {nonfinite}");
    eprintln!("device enumerate parity: {compared} trees compared, {host} host grids, {constant} constant, {nonfinite} non-finite, 0 mismatches");
}

/// Outcomes, per-book minima and book keys do not depend on the tree batch or
/// on the number of book batches (the 504 trees of max size 4 over 3 variables).
#[test]
fn device_enumerate_is_batch_independent() {
    if !device() {
        return;
    }
    let (cols, slice) = panel();
    let n = slice.ret.len();
    let class = Class::build(3, &ops(OPS10), 4, true).unwrap();
    assert_eq!(class.scored().len(), 504);
    let batches: &[usize] = if sanitized() { &[1, 7, 8] } else { &[1, 7, 64] };
    for blocks in [1usize, 4] {
        let family = books(&slice, 3, 0.0, blocks);
        let refs: Vec<&FitnessBook> = family.iter().collect();
        let run = |max_batch: usize, max_book_batch: usize| {
            let opts = DeviceOptions { max_batch, max_book_batch, ..DeviceOptions::default() };
            device_pass(&class, &cols, n, 16, &refs, &opts, true).expect("the device pass runs")
        };
        let (first, r0) = run(batches[0], usize::MAX);
        assert_eq!((r0.batch, r0.book_batch, r0.book_batches), (batches[0], 3, 1));
        for &b in &batches[1..] {
            let (other, r) = run(b, usize::MAX);
            assert_eq!(r.batch, b.min(504));
            same_pass(&first, &other).unwrap_or_else(|e| panic!("{blocks} blocks, batch {b}: {e}"));
        }
        let (cycled, r) = run(batches[1], 1);
        assert_eq!((r.book_batch, r.book_batches), (1, 3));
        same_pass(&first, &cycled).unwrap_or_else(|e| panic!("{blocks} blocks, three book batches: {e}"));
        let again = run(batches[0], usize::MAX).0;
        same_pass(&first, &again).unwrap_or_else(|e| panic!("{blocks} blocks, a second run: {e}"));
    }
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// The training rows of the planted panel of `tests/enumerate.rs` (four
/// symbols, the first 300 hourly stamps, the identity transform), in canonical
/// order: the return pays where `x1 - x2 < -0.3`.
fn planted() -> (Vec<Vec<f64>>, TradingSlice) {
    let (stamps, symbols) = (300usize, 4usize);
    let mut cols = vec![Vec::new(), Vec::new(), Vec::new()];
    let (mut time, mut group, mut ret) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..symbols {
        for k in 0..stamps {
            let i = (k as f64 + 40.0) * 4.0 + j as f64;
            let (x1, x2) = (frac(i * 0.618_033_988_749_894_9) - 0.5, frac(i * 0.754_877_666_246_692_7) - 0.5);
            let x3 = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            cols[0].push(x1);
            cols[1].push(x2);
            cols[2].push(x3);
            time.push(3600.0 * k as f64);
            group.push(j as u64);
            ret.push((if x1 - x2 < -0.3 { 0.004 } else { -0.004 }) + 0.002 * noise);
        }
    }
    (cols, TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret })
}

/// The device arm's class best of size 3 is the planted tree with a cut
/// within one grid step, as in the CPU test; prints the CPU f64 fitness of
/// both arms' class best and the trees whose chosen cut differs.
#[test]
fn device_enumerate_finds_the_planted_rule() {
    if !device() {
        return;
    }
    let (cols, slice) = planted();
    let n = slice.ret.len();
    let spec = TradingSpec { cost_bps: 5.0, min_trades: 1, ..TradingSpec::default() };
    let params = BookParams { spec: &spec, horizon_seconds: 3600.0, blocks: 1, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] };
    let book = book_of(&slice, &params).unwrap();
    let class = Class::build(3, &ops(OPS10), 3, true).unwrap();
    let opts = DeviceOptions { max_batch: if sanitized() { 8 } else { MAX_BATCH }, ..DeviceOptions::default() };
    let (dev, _) = device_pass(&class, &cols, n, 16, &[&book], &opts, true).expect("the device pass runs");
    let vars: Vec<String> = ["x1", "x2", "x3"].iter().map(|s| s.to_string()).collect();
    let ctx = MemberCtx { class: &class, cols: &cols, rows: n, time: &slice.rows.time, book: &book, means: &[0.0; 3], stds: &[1.0; 3], vars: &vars };
    let (best, top) = device_members(&ctx, &dev.outcomes, 50);
    let b3 = best[2].as_ref().expect("a finite member of size 3");
    assert!(matches!((b3.tree.as_str(), b3.cut.side), ("(x1 - x2)", Side::Below) | ("(x2 - x1)", Side::Above)), "{} {:?}", b3.tree, b3.cut.side);
    let shares = crate::enumerate::shares(16);
    let k = shares.iter().position(|&s| Some(s) == b3.cut.share).expect("a grid share");
    assert!(k.abs_diff(2) <= 1, "share {:?} (grid index {k}); the planted share is grid index 2", b3.cut.share);
    let cut = b3.cut.value * if b3.cut.side == Side::Below { 1.0 } else { -1.0 };
    assert!((cut + 0.3).abs() < 0.05, "cut {cut}, planted -0.3");
    assert_eq!(top[0].fitness.to_bits(), b3.fitness.to_bits(), "the planted tree is the best of the class");
    let cpu = pass(&Scan { class: &class, cols: &cols, rows: n, cuts: 16, threads: 2 }, &[&book], true);
    let (_, cpu_top) = members(&ctx, &cpu.outcomes, 1);
    let differ = cpu
        .outcomes
        .iter()
        .zip(&dev.outcomes)
        .filter(|(a, b)| match (a, b) {
            (Outcome::Scored { cut: c, .. }, Outcome::Scored { cut: d, .. }) => c.side != d.side || c.share.map(f64::to_bits) != d.share.map(f64::to_bits),
            _ => a != b,
        })
        .count();
    eprintln!(
        "planted fixture, max size 3: CPU arm class best {} (CPU f64 fitness {}), device arm class best {} (CPU f64 fitness {}, device fitness {:?}); {} of {} trees choose another cut or outcome on the device",
        cpu_top[0].expression,
        cpu_top[0].fitness,
        top[0].expression,
        top[0].fitness,
        top[0].device_fitness,
        differ,
        class.scored().len()
    );
}

/// A refused allocation and a refused launch end the device pass with an
/// error that names the bytes and tag, or the launch, and the command maps it
/// to exit 1: no CPU fallback.
#[test]
fn device_enumerate_refuses_with_bytes() {
    if !device() {
        return;
    }
    let (cols, slice) = planted();
    let n = slice.ret.len();
    let spec = TradingSpec { cost_bps: 5.0, min_trades: 1, ..TradingSpec::default() };
    let params = BookParams { spec: &spec, horizon_seconds: 3600.0, blocks: 4, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] };
    let book = book_of(&slice, &params).unwrap();
    let class = Class::build(3, &ops("add,sub,abs"), 2, true).unwrap();
    let opts = DeviceOptions { max_batch: 4, ..DeviceOptions::default() };
    let mut tags = Vec::new();
    let mut ok = None;
    for at in 1..=64u64 {
        crate::enrich_gpu::set_alloc_fault_for_tests(at);
        let got = device_pass(&class, &cols, n, 16, &[&book], &opts, true);
        crate::enrich_gpu::set_alloc_fault_for_tests(0);
        match got {
            Ok(p) => {
                ok = Some(p);
                break;
            }
            Err(e) => {
                assert!(e.starts_with("device_memory: the device refused an allocation of "), "allocation {at}: {e}");
                let tag = e.split("(tag ").nth(1).and_then(|t| t.split(')').next()).unwrap_or_else(|| panic!("allocation {at}: no tag in {e}"));
                assert!(e.contains(" bytes (tag "), "allocation {at}: {e}");
                tags.push(tag.to_string());
            }
        }
    }
    let (pass, _) = ok.expect("the pass runs once no allocation is refused");
    for tag in ["enum_rows", "enum_books", "enum_prog", "enum_pred", "enum_tree", "enum_hist", "enum_totals"] {
        assert!(tags.iter().any(|t| t == tag), "no refusal named {tag}: {tags:?}");
    }
    let budget = DeviceOptions { budget: Some(20_000), ..opts.clone() };
    let e = device_pass(&class, &cols, n, 16, &[&book], &budget, true).err().expect("a budget below the plan refuses it");
    assert!(e.contains("bytes (tag enum_rows)"), "{e}");
    for kernel in ["rc_enum_eval_kernel", "rc_enum_probe_kernel", "rc_enum_pick_kernel", "rc_enum_hist_kernel", "rc_enum_score_kernel"] {
        force_launch_refusal_for_tests(kernel);
        let got = device_pass(&class, &cols, n, 16, &[&book], &opts, true);
        force_launch_refusal_for_tests("");
        let e = got.err().unwrap_or_else(|| panic!("a refused {kernel} still gave a pass"));
        assert!(e.contains(kernel) && e.contains("launch guard refused"), "{kernel}: {e}");
    }
    let again = device_pass(&class, &cols, n, 16, &[&book], &opts, true).expect("the pass runs after the refusals").0;
    same_pass(&pass, &again).expect("a refusal leaves no state behind");
    force_launch_refusal_for_tests("rc_enum_score");
    let mapped = crate::enumerate::run_device(&class, &cols, n, 16, &[&book]);
    force_launch_refusal_for_tests("");
    let (code, msg) = mapped.err().expect("a refused launch is an error");
    assert_eq!(code, 1, "{msg}");
    assert!(msg.contains("rc_enum_score_kernel"), "{msg}");
}

/// `groups x steps` hours of 32 standard normal variables and returns with a
/// weak signal on the first; 306 x 7,403 is the panel of
/// `enumerate_cost_per_tree` (lib, `enumerate.rs`), drawn from the same stream.
fn campaign_panel(groups: usize, steps: usize) -> (Vec<Vec<f64>>, TradingSlice) {
    let nvars = 32usize;
    let n = groups * steps;
    let mut rng = Rng(268);
    let cols: Vec<Vec<f64>> = (0..nvars).map(|_| (0..n).map(|_| rng.gauss()).collect()).collect();
    let (mut time, mut group, mut ret) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for g in 0..groups {
        for k in 0..steps {
            time.push(3600.0 * k as f64);
            group.push(g as u64);
            ret.push(0.004 * rng.gauss() + 0.0005 * cols[0][g * steps + k]);
        }
    }
    (cols, TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret })
}

/// Device throughput at the campaign training shape (2,265,318 rows, 32
/// variables, the ten operators, 4 blocks, 16 cuts). `RCSR_ENUMERATE_DEVICE`
/// lists the arms `<max size>x<books>/<stride>` (default `4x1/1,4x20/16,5x1/64`):
/// every `stride`-th scored tree of the class, priced on `books` books (the
/// observed book and its shifted copies). Prints trees per second, device ms
/// per tree and per (tree, book), the batch and the peak device bytes; run by
/// hand, it asserts no time. Under compute-sanitizer it runs on 4 x 2,000 rows.
#[test]
#[ignore]
fn enumerate_device_throughput_at_the_campaign_shape() {
    if !device() {
        return;
    }
    let (groups, steps, default) = if sanitized() { (4, 2_000, "4x1/512,4x3/2048,5x1/65536") } else { (306, 7_403, "4x1/1,4x20/16,5x1/64") };
    let arms = std::env::var("RCSR_ENUMERATE_DEVICE").unwrap_or_else(|_| default.to_string());
    let arms: Vec<(usize, usize, usize)> = arms
        .split(',')
        .map(|a| {
            let (size, rest) = a.split_once('x').expect("<size>x<books>/<stride>");
            let (books, stride) = rest.split_once('/').expect("<size>x<books>/<stride>");
            (size.parse().unwrap(), books.parse().unwrap(), stride.parse().unwrap())
        })
        .collect();
    let (cols, slice) = campaign_panel(groups, steps);
    let n = slice.ret.len();
    let spec = TradingSpec { cost_bps: 5.0, min_trades: 1, ..TradingSpec::default() };
    let params = BookParams { spec: &spec, horizon_seconds: 3600.0, blocks: 4, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] };
    let observed = book_of(&slice, &params).unwrap();
    let plan = crate::enumerate::null_plan(&slice.rows, 3600.0, 19, 1).unwrap();
    let shifted: Vec<FitnessBook> = plan.nums.iter().map(|&s| book_of(&shifted_slice(&slice, s, plan.periods as u64).unwrap(), &params).unwrap()).collect();
    let mut all: Vec<&FitnessBook> = vec![&observed];
    all.extend(shifted.iter());
    for (size, nbooks, stride) in arms {
        let class = Class::build(32, &ops(OPS10), size, true).unwrap();
        let nodes: Vec<crate::expr::Node> = class.scored().iter().step_by(stride).map(|&e| class.node(e as usize)).collect();
        let refs = &all[..nbooks];
        let opened = std::time::Instant::now();
        let opts = DeviceOptions { max_batch: if sanitized() { 8 } else { MAX_BATCH }, ..DeviceOptions::default() };
        let mut en = Enumerator::new(&cols, n, refs, 16, size, nodes.len(), &opts).expect("the enumerator opens");
        en.set_books(refs).unwrap();
        let setup = opened.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let mut scored = 0usize;
        for chunk in nodes.chunks(en.batch()) {
            let grids = en.eval(chunk).unwrap();
            scored += grids.iter().filter(|g| matches!(g, TreeGrid::Cuts { cuts, .. } if !cuts.is_empty())).count();
            en.score(&grids, nbooks).unwrap();
        }
        let wall = start.elapsed().as_secs_f64();
        let (eval_ms, score_ms) = en.kernel_ms();
        let t = nodes.len() as f64;
        eprintln!(
            "enumerate device throughput: max size {size}, every {stride}th of {} trees: {} trees ({scored} cut), {n} rows, 4 blocks, 16 cuts, {nbooks} book(s), batch {}: {wall:.1} s wall ({setup:.1} s setup), {:.1} trees/s, {:.3} ms wall per tree, device {:.3} ms per tree for evaluation and grid, {:.3} ms per (tree, book) for scoring, peak {} device bytes",
            class.scored().len(),
            nodes.len(),
            en.batch(),
            t / wall,
            1e3 * wall / t,
            eval_ms / t,
            score_ms / (t * nbooks as f64),
            en.peak_bytes()
        );
    }
}
