//! #299: device benchmark joins, relative returns, rolling beta and
//! per-timestamp cross means.
//!
//! Direct tests drive the device contexts against a plain-Rust oracle that
//! repeats the CPU reference arithmetic (exact timestamp lookup, series-local
//! windows, ascending-order sums). The store test drives the real `--gpu on`
//! path and compares every column at its declared tier.

use rcsr::bench_gpu as bg;

fn registry_tiers_match_the_device_tables() {
    let reg = rcsr::enrich::default_registry();
    for col in bg::BENCH_DEVICE_COLUMNS {
        if let Some(e) = reg.iter().find(|e| e.name == col.name) {
            assert!(e.available, "{} must be available", col.name);
            assert_eq!(e.parity_tier, col.tier.as_str(), "{} parity tier", col.name);
        }
    }
    for (name, tier) in bg::MARKET_DEVICE_COLUMNS {
        let e = reg
            .iter()
            .find(|e| e.name == *name)
            .unwrap_or_else(|| panic!("{name} has no registry entry"));
        assert!(e.available, "{name} must be available");
        assert_eq!(e.parity_tier, tier.as_str(), "{name} parity tier");
    }
}

#[test]
fn inventory_and_declared_tiers() {
    assert_eq!(bg::BENCH_DEVICE_COLUMNS.len(), 30);
    assert_eq!(bg::MARKET_DEVICE_COLUMNS.len(), 13);
    // Every #273 BTC and ETH column is a device column.
    for n in rcsr::enrich::CROSS_BTC_NAMES.iter().chain(rcsr::enrich::CROSS_ETH_NAMES) {
        assert!(bg::is_device_bench(n), "{n} must be a device column");
    }
    for n in ["cross_ret1", "cross_beta20", "rel_btc96", "beta384"] {
        assert!(bg::is_device_bench(n), "{n} must be a device column");
    }
    // Pure lookups are bitwise; anything that uses the asset's own log is 1e-12.
    for n in ["btc_r1", "btc_vol96", "eth_r672", "cross_ret1"] {
        assert_eq!(bg::device_tier(n), Some("bitwise"), "{n}");
    }
    for n in ["rel_btc_r16", "beta_btc_60", "beta_eth_384", "cross_beta20", "beta384"] {
        assert_eq!(bg::device_tier(n), Some("1e-12"), "{n}");
    }
    assert_eq!(bg::device_tier("btc_volume_share"), Some("bitwise"));
    assert_eq!(bg::device_tier("xs_rank_r1"), None);
    assert_eq!(bg::device_tier("funding_rate"), None);
    registry_tiers_match_the_device_tables();
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use rcsr::bench_gpu::{BenchSeries, Kind, MarketDevice};
    use rcsr::data;
    use rcsr::enrich::{self, GpuMode, Options};
    use rcsr::enrich_gpu::EnrichGpu;
    use rcsr::store::{self, FileStore};
    use std::collections::BTreeMap;

    const STEP: i64 = 900;

    fn same_bits(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
    }

    fn close_enough(a: f64, b: f64) -> bool {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() && b.is_nan();
        }
        a == b || (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    struct Seg {
        name: &'static str,
        len: usize,
        gap_at: Option<usize>,
    }

    struct Panel {
        times: Vec<i64>,
        close: Vec<f64>,
        starts: Vec<u32>,
        lens: Vec<u32>,
        names: Vec<&'static str>,
    }

    fn panel(segs: &[Seg], seed: u64) -> Panel {
        let mut st = seed;
        let mut p = Panel {
            times: vec![],
            close: vec![],
            starts: vec![],
            lens: vec![],
            names: vec![],
        };
        for s in segs {
            p.starts.push(p.times.len() as u32);
            p.lens.push(s.len as u32);
            p.names.push(s.name);
            let mut px = 100.0;
            for i in 0..s.len {
                let mut t = i as i64 * STEP;
                if let Some(g) = s.gap_at {
                    if i >= g {
                        t += 3 * STEP;
                    }
                }
                px *= (0.002 * (lcg(&mut st) - 0.5)).exp();
                let mut c = px;
                // A zero and a NaN close exercise the positive-price gate.
                if i == 17 {
                    c = 0.0;
                }
                if i == 41 {
                    c = f64::NAN;
                }
                p.times.push(t);
                p.close.push(c);
            }
        }
        p
    }

    /// Benchmark rows on the base grid with a missing bar every 37th step, NaN
    /// planes at a few bars, and the last base bar present.
    fn bench_series(len: usize, seed: u64, skip: usize) -> BenchSeries {
        let mut st = seed;
        let mut b = BenchSeries::default();
        for i in 0..len + 3 {
            if i % skip == skip - 1 {
                continue;
            }
            b.times.push(i as i64 * STEP);
            for p in 0..bg::PLANE_COUNT {
                let mut v = 0.01 * (lcg(&mut st) - 0.5);
                if p == bg::PLANE_VOLUME {
                    v = 100.0 + 50.0 * lcg(&mut st);
                }
                if i % 53 == 7 && i < 100 && p < 3 {
                    v = f64::NAN;
                }
                b.planes[p].push(v);
            }
        }
        b
    }

    fn at(b: &BenchSeries, plane: usize, t: i64) -> f64 {
        match b.times.binary_search(&t) {
            Ok(j) => b.planes[plane][j],
            Err(_) => f64::NAN,
        }
    }

    fn ret_k(p: &Panel, seg: usize, k: usize) -> Vec<f64> {
        let (s, n) = (p.starts[seg] as usize, p.lens[seg] as usize);
        let t = &p.times[s..s + n];
        let c = &p.close[s..s + n];
        let mut out = vec![f64::NAN; n];
        for i in k..n {
            if t[i] - t[i - k] == k as i64 * STEP && c[i] > 0.0 && c[i - k] > 0.0 {
                out[i] = (c[i] / c[i - k]).ln();
            }
        }
        out
    }

    fn contig(t: &[i64], i: usize, w: usize) -> bool {
        i + 1 >= w && t[i + 1 - w] == t[i] - (w as i64 - 1) * STEP
    }

    /// Port of `enrich::beta_window` plus the benchmark-self shortcut.
    fn oracle(col: &bg::BenchColumn, p: &Panel, b: &BenchSeries, is_self: &[bool]) -> Vec<f64> {
        let mut out = Vec::new();
        for seg in 0..p.starts.len() {
            let (s, n) = (p.starts[seg] as usize, p.lens[seg] as usize);
            let t = &p.times[s..s + n];
            let vals: Vec<f64> = match col.kind {
                Kind::Lookup(plane) => t.iter().map(|&x| at(b, plane, x)).collect(),
                Kind::Rel(k) => {
                    let own = ret_k(p, seg, k);
                    let plane = match k {
                        1 => bg::PLANE_R1,
                        16 => bg::PLANE_R16,
                        20 => bg::PLANE_R20,
                        _ => bg::PLANE_R96,
                    };
                    (0..n)
                        .map(|i| {
                            let bv = at(b, plane, t[i]);
                            if own[i].is_finite() && bv.is_finite() {
                                own[i] - bv
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                }
                Kind::Beta(w) | Kind::BetaClipCov(w) => {
                    let clip = matches!(col.kind, Kind::BetaClipCov(_));
                    let x = ret_k(p, seg, 1);
                    let mut v = vec![f64::NAN; n];
                    for i in 0..n {
                        if !contig(t, i, w) {
                            continue;
                        }
                        if !clip && is_self[seg] {
                            v[i] = 1.0;
                            continue;
                        }
                        let (mut sx, mut sy, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0);
                        let mut ok = true;
                        for j in i + 1 - w..=i {
                            let yv = at(b, bg::PLANE_R1, t[j]);
                            if !x[j].is_finite() || !yv.is_finite() {
                                ok = false;
                                break;
                            }
                            sx += x[j];
                            sy += yv;
                            syy += yv * yv;
                            sxy += x[j] * yv;
                        }
                        if !ok {
                            continue;
                        }
                        let nf = w as f64;
                        let mut cov = sxy / nf - (sx / nf) * (sy / nf);
                        if clip {
                            cov = cov.max(-0.05).min(0.05);
                        }
                        let var = syy / nf - (sy / nf) * (sy / nf);
                        if var > 0.0 {
                            v[i] = (cov / var).max(-5.0).min(5.0);
                        }
                    }
                    v
                }
            };
            out.extend(vals);
        }
        out
    }

    fn have_device() -> bool {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return false;
        }
        true
    }

    fn segs() -> Vec<Seg> {
        // Lengths straddle the 20/60/384 windows so both edges of every window
        // are exercised; the second and third segments are the benchmarks.
        let lens = [59usize, 60, 61, 383, 384, 385, 1000, 20, 19];
        let mut v: Vec<Seg> = lens
            .iter()
            .map(|&len| Seg { name: "AAA", len, gap_at: None })
            .collect();
        v[1].name = "BTCUSDT";
        v[2].name = "ETHUSDT";
        v[5].gap_at = Some(200);
        v
    }

    #[test]
    fn benchmark_columns_match_the_oracle_with_gaps_missing_bars_and_window_edges() {
        if !have_device() {
            return;
        }
        let segs = segs();
        let p = panel(&segs, 7);
        let btc = bench_series(1300, 11, 500);
        let eth = bench_series(1300, 13, 450);
        let nseg = segs.len();
        let mut flags = vec![0u8; 2 * nseg];
        flags[1] = 1; // BTC itself
        flags[nseg + 2] = 1; // ETH itself
        let all: Vec<String> = bg::BENCH_DEVICE_COLUMNS.iter().map(|c| c.name.to_string()).collect();
        let (cols, _stats) = bg::compute(
            &p.times, &p.close, &p.starts, &p.lens, STEP, Some(&btc), Some(&eth), &flags, &all,
        )
        .expect("device benchmark context");
        assert_eq!(cols.len(), 30);
        let mut worst: f64 = 0.0;
        for (name, got) in &cols {
            let col = bg::bench_column(name).unwrap();
            let bench = match col.bench {
                bg::Bench::Btc => &btc,
                bg::Bench::Eth => &eth,
            };
            let is_self: Vec<bool> = (0..nseg)
                .map(|s| flags[col.bench.index() * nseg + s] == 1)
                .collect();
            let want = oracle(col, &p, bench, &is_self);
            assert_eq!(got.len(), want.len(), "{name}");
            let mut finite = 0usize;
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if col.tier == bg::Tier::Bitwise {
                    assert!(same_bits(*g, *w), "{name} row {i}: device {g:?} oracle {w:?}");
                } else {
                    assert!(close_enough(*g, *w), "{name} row {i}: device {g:?} oracle {w:?}");
                    if g.is_finite() && w.is_finite() {
                        worst = worst.max((g - w).abs());
                    }
                }
                if g.is_finite() {
                    finite += 1;
                }
            }
            // Every column must carry both finite and NaN rows on this fixture.
            assert!(finite > 0, "{name}: no finite rows, the fixture proves nothing");
            assert!(finite < got.len(), "{name}: no NaN rows");
        }
        eprintln!("worst 1e-12-tier absolute difference: {worst:e}");
    }

    #[test]
    fn missing_benchmark_timestamp_is_nan_not_stale() {
        if !have_device() {
            return;
        }
        let segs = [Seg { name: "AAA", len: 200, gap_at: None }];
        let p = panel(&segs, 3);
        let mut btc = bench_series(200, 5, 1000);
        // Drop bar 100 only.
        let drop = btc.times.iter().position(|&t| t == 100 * STEP).unwrap();
        btc.times.remove(drop);
        for pl in btc.planes.iter_mut() {
            pl.remove(drop);
        }
        let flags = vec![0u8; 2];
        let names = vec!["btc_r1".to_string(), "rel_btc_r1".to_string()];
        let (cols, _) = bg::compute(
            &p.times, &p.close, &p.starts, &p.lens, STEP, Some(&btc), None, &flags, &names,
        )
        .unwrap();
        for (name, v) in &cols {
            assert!(v[100].is_nan(), "{name} must be NaN at the missing bar");
            assert!(v[99].is_finite() || name == "rel_btc_r1" || v[99].is_nan());
        }
        assert!(cols[0].1[101].is_finite(), "bar after the hole is a fresh lookup");
        // An ETH column without an ETH series is a refusal, not a zero column.
        let eth_only = vec!["eth_r1".to_string()];
        assert!(bg::compute(
            &p.times, &p.close, &p.starts, &p.lens, STEP, Some(&btc), None, &flags, &eth_only,
        )
        .is_none());
    }

    #[test]
    fn repeated_runs_are_hash_identical() {
        if !have_device() {
            return;
        }
        let segs = segs();
        let p = panel(&segs, 9);
        let btc = bench_series(1300, 11, 500);
        let eth = bench_series(1300, 13, 450);
        let flags = vec![0u8; 2 * segs.len()];
        let all: Vec<String> = bg::BENCH_DEVICE_COLUMNS.iter().map(|c| c.name.to_string()).collect();
        let run = || {
            bg::compute(
                &p.times, &p.close, &p.starts, &p.lens, STEP, Some(&btc), Some(&eth), &flags, &all,
            )
            .unwrap()
            .0
        };
        let a = run();
        let b = run();
        for ((na, va), (nb, vb)) in a.iter().zip(&b) {
            assert_eq!(na, nb);
            for (i, (x, y)) in va.iter().zip(vb).enumerate() {
                assert!(same_bits(*x, *y), "{na} row {i} differs between runs");
            }
        }
        // Batch invariance: one column at a time equals the batched request.
        for name in ["rel_btc_r20", "beta_eth_60", "beta384"] {
            let one = bg::compute(
                &p.times,
                &p.close,
                &p.starts,
                &p.lens,
                STEP,
                Some(&btc),
                Some(&eth),
                &flags,
                &[name.to_string()],
            )
            .unwrap()
            .0;
            let batched = a.iter().find(|(n, _)| n == name).unwrap();
            for (i, (x, y)) in one[0].1.iter().zip(&batched.1).enumerate() {
                assert!(same_bits(*x, *y), "{name} row {i}: single vs batched");
            }
        }
    }

    // ---------------------------------------------------------------------
    // Cross-section means
    // ---------------------------------------------------------------------

    struct Cs {
        times: Vec<i64>,
        sym: Vec<u32>,
        member: Vec<bool>,
        starts: Vec<u32>,
        lens: Vec<u32>,
        close: Vec<f64>,
    }

    fn cross_section(symbols: usize, rows: usize, seed: u64) -> Cs {
        let mut st = seed;
        let mut c = Cs {
            times: vec![],
            sym: vec![],
            member: vec![],
            starts: vec![],
            lens: vec![],
            close: vec![],
        };
        for s in 0..symbols {
            // Symbols list late and drop a few bars so group sizes vary around
            // the 20-member floor.
            let first = (s * 7) % 40;
            c.starts.push(c.times.len() as u32);
            let mut n = 0u32;
            let mut px = 50.0 + s as f64;
            for i in first..rows {
                if (i + s) % 401 == 0 {
                    continue;
                }
                px *= (0.004 * (lcg(&mut st) - 0.5)).exp();
                c.times.push(i as i64 * STEP);
                c.sym.push(s as u32);
                c.member.push(!(s % 5 == 0 && i < 120));
                c.close.push(px);
                n += 1;
            }
            c.lens.push(n);
        }
        c
    }

    fn order(c: &Cs) -> Vec<usize> {
        let mut o: Vec<usize> = (0..c.times.len()).collect();
        o.sort_by(|&a, &b| {
            c.times[a].cmp(&c.times[b]).then(c.sym[a].cmp(&c.sym[b])).then(a.cmp(&b))
        });
        o
    }

    fn groups(c: &Cs, o: &[usize]) -> Vec<Vec<usize>> {
        let mut g: Vec<Vec<usize>> = Vec::new();
        for &r in o {
            match g.last_mut() {
                Some(last) if c.times[last[0]] == c.times[r] => last.push(r),
                _ => g.push(vec![r]),
            }
        }
        g
    }

    fn source(c: &Cs, seed: u64, scale: f64) -> Vec<f64> {
        source_nan(c, seed, scale, 29)
    }

    fn source_nan(c: &Cs, seed: u64, scale: f64, nan_every: usize) -> Vec<f64> {
        let mut st = seed;
        (0..c.times.len())
            .map(|i| {
                let v = scale * (lcg(&mut st) - 0.45);
                if i % nan_every == 3 {
                    f64::NAN
                } else if i % 31 == 5 && nan_every < 100 {
                    0.0
                } else {
                    v
                }
            })
            .collect()
    }

    /// Port of the r16 block of `market_state_pass`.
    fn oracle_stats(c: &Cs, src: &[f64]) -> [Vec<f64>; 4] {
        let rows = src.len();
        let (mut mkt, mut br, mut sd, mut rel) =
            (vec![f64::NAN; rows], vec![f64::NAN; rows], vec![f64::NAN; rows], vec![f64::NAN; rows]);
        for group in groups(c, &order(c)) {
            let mut vals = Vec::new();
            let mut sum = 0.0;
            for &r in &group {
                if c.member[r] && src[r].is_finite() {
                    vals.push(r);
                    sum += src[r];
                }
            }
            let n = vals.len();
            if n < 20 {
                continue;
            }
            let nf = n as f64;
            let mean = sum / nf;
            let mut ss = 0.0;
            let mut positive = 0usize;
            for &r in &vals {
                let d = src[r] - mean;
                ss += d * d;
                if src[r] > 0.0 {
                    positive += 1;
                }
            }
            let s = (ss / nf).max(0.0).sqrt();
            let b = positive as f64 / nf;
            for &r in &group {
                if !c.member[r] {
                    continue;
                }
                mkt[r] = mean;
                br[r] = b;
                sd[r] = s;
                if src[r].is_finite() && n > 1 {
                    rel[r] = src[r] - (sum - src[r]) / (n - 1) as f64;
                }
            }
        }
        [mkt, br, sd, rel]
    }

    /// Port of the `avg_corr` block of `market_state_pass`.
    fn oracle_avg_corr(c: &Cs, r1: &[f64]) -> Vec<f64> {
        let rows = r1.len();
        let g = groups(c, &order(c));
        let mut by_time = BTreeMap::new();
        for group in &g {
            let mut vals = 0usize;
            let mut sum = 0.0;
            for &r in group {
                if c.member[r] && r1[r].is_finite() {
                    vals += 1;
                    sum += r1[r];
                }
            }
            if vals >= 20 {
                by_time.insert(c.times[group[0]], sum / vals as f64);
            }
        }
        let mut member_corr = vec![f64::NAN; rows];
        for s in 0..c.starts.len() {
            let (start, nt) = (c.starts[s] as usize, c.lens[s] as usize);
            if nt < 96 {
                continue;
            }
            let mut p = vec![vec![0.0f64; nt + 1]; 5];
            let mut pc = vec![0usize; nt + 1];
            for j in 0..nt {
                let gi = start + j;
                let x = r1[gi];
                let y = if c.member[gi] {
                    by_time.get(&c.times[gi]).copied().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                };
                let ok = x.is_finite() && y.is_finite();
                let add = [x, y, x * x, y * y, x * y];
                for k in 0..5 {
                    p[k][j + 1] = p[k][j] + if ok { add[k] } else { 0.0 };
                }
                pc[j + 1] = pc[j] + ok as usize;
            }
            let t = &c.times[start..start + nt];
            for i in 0..nt {
                if !contig(t, i, 96) || pc[i + 1] - pc[i + 1 - 96] != 96 {
                    continue;
                }
                let lo = i + 1 - 96;
                let w = 96.0;
                let d = |k: usize| p[k][i + 1] - p[k][lo];
                let (sx, sy, sxx, syy, sxy) = (d(0), d(1), d(2), d(3), d(4));
                let vx = sxx - sx * sx / w;
                let vy = syy - sy * sy / w;
                if vx > 0.0 && vy > 0.0 {
                    member_corr[start + i] =
                        ((sxy - sx * sy / w) / (vx * vy).sqrt()).max(-1.0).min(1.0);
                }
            }
        }
        let mut avg = vec![f64::NAN; rows];
        for group in &g {
            let mut vals = 0usize;
            let mut sum = 0.0;
            for &r in group {
                if c.member[r] && member_corr[r].is_finite() {
                    vals += 1;
                    sum += member_corr[r];
                }
            }
            if vals < 20 {
                continue;
            }
            let mean = sum / vals as f64;
            for &r in group {
                if c.member[r] {
                    avg[r] = mean;
                }
            }
        }
        avg
    }

    fn market_ctx(c: &Cs) -> EnrichGpu {
        let n = c.times.len();
        let mut gpu = EnrichGpu::new(
            &c.times,
            &c.close,
            &c.close,
            &c.close,
            &c.close,
            &vec![1.0; n],
            &vec![1.0; n],
            &c.starts,
            &c.lens,
            STEP,
        )
        .expect("enrich context");
        assert!(gpu.prepare_cross_section(&c.sym, Some(&c.member)));
        gpu
    }

    #[test]
    fn market_group_statistics_are_bit_identical_to_the_cpu_loops() {
        if !have_device() {
            return;
        }
        let c = cross_section(34, 420, 21);
        let mut gpu = market_ctx(&c);
        let want_all = bg::WANT_MEAN | bg::WANT_BREADTH | bg::WANT_DISPERSION | bg::WANT_REL;
        for (seed, scale) in [(1u64, 0.02), (2, 0.5)] {
            let src = source(&c, seed, scale);
            let st = gpu.group_stats(&src, want_all).expect("group stats");
            let [mkt, br, sd, rel] = oracle_stats(&c, &src);
            let mut finite = 0usize;
            for i in 0..src.len() {
                assert!(same_bits(st.mean[i], mkt[i]), "mean row {i}: {:?} vs {:?}", st.mean[i], mkt[i]);
                assert!(same_bits(st.breadth[i], br[i]), "breadth row {i}");
                assert!(same_bits(st.dispersion[i], sd[i]), "dispersion row {i}");
                assert!(same_bits(st.rel[i], rel[i]), "rel row {i}");
                finite += st.mean[i].is_finite() as usize;
            }
            assert!(finite > 0 && finite < src.len(), "fixture must mix valid and thin timestamps");
            // A partial request returns only what was asked for.
            let only = gpu.group_stats(&src, bg::WANT_MEAN).unwrap();
            assert_eq!(only.mean.len(), src.len());
            assert!(only.breadth.is_empty() && only.rel.is_empty());
        }
    }

    #[test]
    fn average_correlation_and_volume_share_match_the_cpu_loops() {
        if !have_device() {
            return;
        }
        let c = cross_section(30, 520, 31);
        let mut gpu = market_ctx(&c);
        let r1 = source_nan(&c, 5, 0.01, 997);
        let got = gpu.avg_corr(&r1).expect("avg_corr");
        let want = oracle_avg_corr(&c, &r1);
        let mut finite = 0usize;
        for i in 0..r1.len() {
            assert!(same_bits(got[i], want[i]), "avg_corr row {i}: {:?} vs {:?}", got[i], want[i]);
            finite += got[i].is_finite() as usize;
        }
        assert!(finite > 0, "avg_corr fixture is empty");

        // btc_volume_share against a benchmark that misses some timestamps.
        let vol: Vec<f64> = source(&c, 8, 100.0).iter().map(|v| v.abs()).collect();
        let mut bench = bench_series(520, 77, 17);
        bench.planes[bg::PLANE_VOLUME].iter_mut().for_each(|v| *v += 5.0);
        let share = gpu.volume_share(&vol, &bench).expect("volume share");
        let mut expect = vec![f64::NAN; vol.len()];
        for group in groups(&c, &order(&c)) {
            let (mut total, mut count) = (0.0, 0usize);
            for &r in &group {
                if c.member[r] && vol[r].is_finite() {
                    total += vol[r];
                    count += 1;
                }
            }
            if count < 20 || !(total > 0.0) {
                continue;
            }
            let b = at(&bench, bg::PLANE_VOLUME, c.times[group[0]]);
            if !b.is_finite() {
                continue;
            }
            for &r in &group {
                if c.member[r] {
                    expect[r] = b / total;
                }
            }
        }
        let mut finite = 0usize;
        for i in 0..vol.len() {
            assert!(same_bits(share[i], expect[i]), "share row {i}");
            finite += share[i].is_finite() as usize;
        }
        assert!(finite > 0);
    }

    // ---------------------------------------------------------------------
    // Real enrich path
    // ---------------------------------------------------------------------

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rcsr-bench-gpu-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn column(path: &str, name: &str) -> Vec<f64> {
        let s = FileStore::open(path).unwrap();
        let c = s.header().column_index(name).unwrap();
        let mut out = Vec::new();
        for g in 0..s.n_groups() {
            out.extend(s.read_f64_group(g, c).unwrap());
        }
        out
    }

    /// 24 symbols including BTCUSDT and ETHUSDT. BTC misses a bar every 211
    /// steps, ETH every 173; one alt has a time gap; all share one grid so every
    /// timestamp has more than the 20-member floor.
    fn write_panel(path: &str, per_symbol: usize) {
        let mut text = String::from("ts,symbol,open,high,low,close,volume\n");
        let mut st = 99u64;
        let mut names: Vec<String> = vec!["BTCUSDT".into(), "ETHUSDT".into()];
        names.extend((0..22).map(|i| format!("ALT{:02}", i)));
        names.sort();
        for (si, name) in names.iter().enumerate() {
            let mut px = 20.0 + si as f64;
            let gap_at = if name == "ALT05" { Some(per_symbol / 2) } else { None };
            for i in 0..per_symbol {
                if name == "BTCUSDT" && i % 1500 == 1499 {
                    continue;
                }
                if name == "ETHUSDT" && i % 1200 == 1199 {
                    continue;
                }
                let mut t = i as i64 * STEP;
                if let Some(g) = gap_at {
                    if i >= g {
                        t += 4 * STEP;
                    }
                }
                px *= (0.003 * (lcg(&mut st) - 0.5)).exp();
                let o = px;
                let c = px * (1.0 + 0.0005 * (lcg(&mut st) - 0.5));
                let h = o.max(c) * 1.0008;
                let l = o.min(c) * 0.9992;
                let v = 10.0 + 90.0 * lcg(&mut st);
                text.push_str(&format!("{},{},{},{},{},{},{}\n", t, name, o, h, l, c, v));
            }
        }
        let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
        store::write_store_with_time(path, &table, 65_536, Some(0)).unwrap();
    }

    fn opts(data: &str, out: &str, features: &str, gpu: GpuMode, chunk: usize) -> Options {
        Options {
            data: data.to_string(),
            out: Some(out.to_string()),
            features: vec![features.to_string()],
            allow_lockbox: true,
            gpu,
            chunk_rows: chunk,
            ..Default::default()
        }
    }

    const STORE_FEATURES: &str = "btc_r1,btc_r16,btc_r20,btc_r96,btc_r672,btc_vol20,btc_vol96,\
rel_btc_r1,rel_btc_r16,rel_btc_r20,rel_btc_r96,beta_btc_60,cross_ret1,cross_beta20,\
eth_r1,eth_r96,eth_vol20,rel_eth_r1,rel_eth_r96,beta_eth_60,\
mkt_r16,mkt_ew_ret16,breadth,mkt_breadth,dispersion,mkt_dispersion,rel_mkt16,mkt_r96,\
rel_mkt96,mkt_vol96,mkt_ew_ret1,avg_corr,btc_volume_share";

    #[test]
    fn store_path_matches_cpu_at_the_declared_tiers_and_is_repeatable() {
        if !have_device() {
            return;
        }
        let dir = tmp_dir("store");
        let src = dir.join("in.rcsb").to_string_lossy().into_owned();
        // 24 symbols x 44,200 rows crosses the 1M-row dispatch floor.
        write_panel(&src, 44_200);
        let run = |tag: &str, mode: GpuMode, chunk: usize| {
            let out = dir.join(format!("{tag}.rcsb")).to_string_lossy().into_owned();
            let r = enrich::build(&opts(&src, &out, STORE_FEATURES, mode, chunk)).unwrap();
            (r, out)
        };
        let (cpu, cpu_out) = run("cpu", GpuMode::Off, 65_536);
        let (gpu, gpu_out) = run("gpu", GpuMode::On, 65_536);
        let (gpu2, _) = run("gpu2", GpuMode::On, 4_096);
        let (gpu3, _) = run("gpu3", GpuMode::On, 65_536);

        let names: Vec<String> = STORE_FEATURES.split(',').map(String::from).collect();
        assert_eq!(cpu.gpu_columns, 0);
        assert_eq!(gpu.gpu_fallback, 0, "no column may fall back");
        for n in &names {
            assert!(
                bg::is_device_bench(n) || bg::is_device_market(n),
                "{n} is not a device column"
            );
        }
        // Every benchmark column and every market column ran on the device.
        let device_bench = names.iter().filter(|n| bg::is_device_bench(n)).count();
        assert!(
            gpu.gpu_columns >= device_bench + 12,
            "gpu_columns {} < {}",
            gpu.gpu_columns,
            device_bench + 12
        );

        let mut checked = 0usize;
        for n in &names {
            let a = column(&cpu_out, n);
            let b = column(&gpu_out, n);
            assert_eq!(a.len(), b.len(), "{n} length");
            let tier = bg::device_tier(n).unwrap();
            let mut finite = 0usize;
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                if tier == "bitwise" {
                    assert!(same_bits(*x, *y), "{n} row {i}: CPU {x:?} GPU {y:?}");
                } else {
                    assert!(close_enough(*x, *y), "{n} row {i}: CPU {x:?} GPU {y:?}");
                }
                finite += x.is_finite() as usize;
            }
            assert!(finite > 0, "{n}: CPU column is all NaN, fixture proves nothing");
            checked += 1;
        }
        assert_eq!(checked, names.len());

        // Bitwise-tier hashes are identical between the arms.
        for f in &cpu.features {
            if bg::device_tier(&f.name) == Some("bitwise") {
                let g = gpu.features.iter().find(|x| x.name == f.name).unwrap();
                assert_eq!(f.hash, g.hash, "{} hash CPU vs GPU", f.name);
            }
        }
        // Repeated runs and a different chunk size give identical hashes for
        // every column.
        for (a, b) in gpu.features.iter().zip(&gpu2.features) {
            assert_eq!(a.hash, b.hash, "{} hash across chunk sizes", a.name);
        }
        for (a, b) in gpu.features.iter().zip(&gpu3.features) {
            assert_eq!(a.hash, b.hash, "{} hash across repeated runs", a.name);
        }
        eprintln!(
            "store path: CPU feature {:.3}s xs {:.3}s, GPU feature {:.3}s xs {:.3}s, kernel {:.3}s",
            cpu.feature_seconds, cpu.xs_seconds, gpu.feature_seconds, gpu.xs_seconds,
            gpu.gpu_kernel_seconds
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
