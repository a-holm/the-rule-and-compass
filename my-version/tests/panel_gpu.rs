//! #301: fused 15 m → 1 h panel preparation parity and determinism.
//!
//! These tests drive the CPU reference (`panel_prep::prepare`) and the device
//! pass (`panel_prep_gpu` / `src/cuda/prep.cu`) over the same decoded blocks.
//! They cover complete and partial buckets, zero volume, `H == L`, identical
//! and conflicting duplicates, listing/delisting/migration/membership chain
//! resets, prefix replay and block-size invariance.
//!
//! The 4.78 M-row decoded block is hashed CPU vs device, cold and warm. The
//! device tests skip on a CUDA build with no device.

#![cfg(feature = "cuda")]

use rcsr::panel_prep::{prepare as cpu_prepare, PrepBar, PrepOutput, PrepSegment};
use rcsr::panel_prep_gpu::{self, PrepGpu};

fn q(t: i64, p: f64, volume: f64) -> PrepBar {
    PrepBar::new(t, p, p + 1.0, p - 1.0, p + 0.25).with_flow(volume, volume * 2.0, volume / 10.0)
}

fn segs(lens: &[u32]) -> Vec<PrepSegment> {
    let mut start = 0u32;
    lens.iter()
        .map(|len| {
            let s = PrepSegment { start, len: *len };
            start += *len;
            s
        })
        .collect()
}

fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn assert_outputs_equal(label: &str, a: &PrepOutput, b: &PrepOutput) {
    assert_eq!(a.buckets.len(), b.buckets.len(), "{label}: bucket count");
    assert_eq!(a.conflicts.len(), b.conflicts.len(), "{label}: conflict count");
    assert_eq!(a.input_rows, b.input_rows, "{label}: input rows");
    assert_eq!(a.unique_rows, b.unique_rows, "{label}: unique rows");
    for (i, (x, y)) in a.buckets.iter().zip(b.buckets.iter()).enumerate() {
        assert_eq!(x.time, y.time, "{label}: bucket {i} time");
        assert_eq!(x.segment, y.segment, "{label}: bucket {i} segment");
        assert_eq!(x.chain, y.chain, "{label}: bucket {i} chain");
        for (name, xv, yv) in [
            ("open", x.open, y.open),
            ("high", x.high, y.high),
            ("low", x.low, y.low),
            ("close", x.close, y.close),
            ("volume", x.volume, y.volume),
            ("quote_volume", x.quote_volume, y.quote_volume),
            ("trades", x.trades, y.trades),
            ("taker_buy_base", x.taker_buy_base, y.taker_buy_base),
            ("taker_buy_quote", x.taker_buy_quote, y.taker_buy_quote),
        ] {
            assert!(same_bits(xv, yv), "{label}: bucket {i} {name}: {xv:?} vs {yv:?}");
        }
        assert_eq!(x.cov15m, y.cov15m, "{label}: bucket {i} cov15m");
        assert_eq!(x.dup_rows, y.dup_rows, "{label}: bucket {i} dup_rows");
        assert_eq!(x.gap, y.gap, "{label}: bucket {i} gap");
        assert_eq!(x.complete, y.complete, "{label}: bucket {i} complete");
        assert_eq!(x.traded, y.traded, "{label}: bucket {i} traded");
        assert_eq!(x.partial, y.partial, "{label}: bucket {i} partial");
        assert_eq!(
            x.duplicate_conflict, y.duplicate_conflict,
            "{label}: bucket {i} conflict flag"
        );
        assert_eq!(x.outage, y.outage, "{label}: bucket {i} outage");
    }
    for (i, (x, y)) in a.conflicts.iter().zip(b.conflicts.iter()).enumerate() {
        assert_eq!(x, y, "{label}: conflict {i}");
    }
}

fn would_use_gpu() -> bool {
    rcsr::gpu::probe().available()
}

/// A mixed fixture: unsorted within segments, identical and conflicting
/// duplicates, a missing hour, a partial hour, zero volume and `H == L`.
fn mixed_fixture() -> (Vec<PrepBar>, Vec<PrepSegment>) {
    let mut aaa = Vec::new();
    for hour in 0..30i64 {
        if hour == 7 {
            continue; // gap: buckets 08.. carry missing counts
        }
        let cov = if hour == 21 { 3 } else { 4 };
        for k in 0..cov {
            let t = hour * 3600 + k * 900;
            let p = 100.0 + hour as f64 + k as f64 * 0.1;
            let mut b = q(t, p, if hour == 4 { 0.0 } else { 10.0 + k as f64 });
            if hour == 5 {
                let flat = 100.0 + hour as f64;
                b.open = flat;
                b.high = flat;
                b.low = flat;
                b.close = flat;
            }
            if hour == 9 && k == 2 {
                // Identical copies in non-sorted input order.
                aaa.push(b);
                aaa.push(b);
                continue;
            }
            if hour == 12 && k == 1 {
                // Conflicting copies: one payload differs.
                aaa.push(b);
                let mut c = b;
                c.close = b.close + 0.5;
                aaa.push(c);
                continue;
            }
            aaa.push(b);
        }
    }
    // Deliberately shuffle a little within the symbol: the stable sort must
    // still recover time order and preserve duplicate input order.
    aaa.swap(0, 3);
    aaa.swap(10, 11);

    let mut bbb = Vec::new();
    for hour in 2..18i64 {
        for k in 0..4 {
            let t = hour * 3600 + k * 900;
            let p = 200.0 + hour as f64 + k as f64 * 0.1;
            bbb.push(q(t, p, 5.0));
        }
    }
    // Chain change inside a symbol: migration-like reset.
    for b in bbb.iter_mut().filter(|b| b.time >= 8 * 3600) {
        b.chain = 7;
    }
    let mut bars = aaa;
    let n_a = bars.len() as u32;
    bars.extend(bbb);
    let n_b = bars.len() as u32 - n_a;
    (bars, segs(&[n_a, n_b]))
}

#[test]
fn cpu_reference_matches_the_documented_policy() {
    let (bars, segments) = mixed_fixture();
    let out = cpu_prepare(&bars, &segments, 900, 3600, true).unwrap();
    assert!(!out.buckets.is_empty());
    assert_eq!(out.input_rows, bars.len());
    // Hour 7 is absent, so every later AAA bucket carries a positive gap until
    // the next output; hour 21 is partial under keep_partial.
    let partial = out
        .buckets
        .iter()
        .find(|b| b.time == 21 * 3600)
        .expect("hour 21");
    assert!(partial.partial && !partial.complete);
    assert_eq!(partial.cov15m, 3);
    let zero = out
        .buckets
        .iter()
        .find(|b| b.time == 4 * 3600)
        .expect("hour 4");
    assert_eq!(zero.volume, 0.0);
    assert!(!zero.traded, "zero-volume bucket is kept but not traded");
    let flat = out
        .buckets
        .iter()
        .find(|b| b.time == 5 * 3600)
        .expect("hour 5");
    assert_eq!(flat.high, flat.low);
    // Identical copies must not multiply the flow: 10 + 11 + 12 + 13.
    let identical = out
        .buckets
        .iter()
        .find(|b| b.time == 9 * 3600)
        .expect("hour 9");
    assert_eq!(identical.dup_rows, 5);
    assert_eq!(identical.volume, 46.0);
    assert!(!identical.duplicate_conflict);
    let conflict = out
        .buckets
        .iter()
        .find(|b| b.time == 12 * 3600)
        .expect("hour 12");
    assert!(conflict.duplicate_conflict);
    assert_eq!(conflict.dup_rows, 5);
    assert_eq!(out.conflicts.len(), 1);
    assert_eq!(out.conflicts[0].time, 12 * 3600 + 900);
    assert_eq!(out.conflicts[0].segment, 0);
    assert_eq!(out.conflicts[0].dup_rows, 2);
}

#[test]
fn device_matches_the_cpu_reference_on_a_mixed_block() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let (bars, segments) = mixed_fixture();
    let cpu = cpu_prepare(&bars, &segments, 900, 3600, true).unwrap();
    let (gpu, stats) = panel_prep_gpu::prepare(&bars, &segments, 900, 3600, true)
        .expect("device panel prep");
    assert_outputs_equal("mixed", &cpu, &gpu);
    assert!(stats.kernel_s >= 0.0);
    assert!(stats.h2d_bytes > 0);
    assert!(stats.d2h_bytes > 0);
}

#[test]
fn identical_copies_collapse_and_conflicts_are_prefixed_in_time_order() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // Four copies of a whole 2,976-row month (JUV-like), plus a conflicting
    // AUD-like month after it. The first month must collapse, not quadruple;
    // the second must produce one ordered conflict entry.
    let mut bars = Vec::new();
    let mut month = Vec::new();
    for i in 0..2_976i64 {
        let t = i * 900;
        let p = 10.0 + (i % 97) as f64 * 0.01;
        month.push(q(t, p, 4.0));
    }
    for _ in 0..4 {
        bars.extend(month.iter().copied());
    }
    let first_month = bars.len() as u32;
    let mut conflicts = Vec::new();
    for i in 0..2_976i64 {
        let t = i * 900;
        let p = 20.0 + (i % 89) as f64 * 0.01;
        let b = q(t, p, 3.0);
        conflicts.push(b);
        if i == 17 || i == 900 || i == 2_000 {
            let mut c = b;
            c.close += 0.25;
            conflicts.push(c);
        }
    }
    let second_month = conflicts.len() as u32;
    bars.extend(conflicts);
    let segments = segs(&[first_month, second_month]);
    let cpu = cpu_prepare(&bars, &segments, 900, 3_600, true).unwrap();
    let (gpu, _) = panel_prep_gpu::prepare(&bars, &segments, 900, 3_600, true).unwrap();
    assert_outputs_equal("juv/aud", &cpu, &gpu);

    assert!(cpu.conflicts.iter().all(|c| c.segment == 1));
    assert_eq!(cpu.conflicts.len(), 3);
    assert_eq!(
        cpu.conflicts.iter().map(|c| c.time).collect::<Vec<_>>(),
        vec![17 * 900, 900 * 900, 2_000 * 900]
    );
    // Four unique bars, each seen four times: dup_rows counts the 16 source
    // rows, while the volume is one copy's worth (4 x 4 = 16 per hour).
    let hour = cpu.buckets.iter().find(|b| b.time == 0).unwrap();
    assert_eq!(hour.dup_rows, 16);
    assert_eq!(hour.volume, 16.0);
}

#[test]
fn gap_chains_reset_at_listing_delisting_migration_and_membership() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // Symbol A: listing at hour 0, then hour 3 after a hole.
    let mut a = Vec::new();
    for hour in [0i64, 3] {
        for k in 0..4 {
            a.push(q(hour * 3_600 + k * 900, 100.0 + k as f64, 1.0));
        }
    }
    // Symbol B: delisting of A / listing of B at hour 10, then a migration at
    // hour 12 and a member=0 reset at hour 15. Each starts a fresh chain.
    let mut b = Vec::new();
    for hour in [10i64, 12, 15] {
        for k in 0..4 {
            let mut bar = q(hour * 3_600 + k * 900, 200.0 + k as f64, 1.0);
            if hour >= 12 {
                bar.chain = 1;
            }
            if hour >= 15 {
                bar.chain = 2;
            }
            b.push(bar);
        }
    }
    let n_a = a.len() as u32;
    let n_b = b.len() as u32;
    let mut bars = a;
    bars.extend(b);
    let segments = segs(&[n_a, n_b]);
    let cpu = cpu_prepare(&bars, &segments, 900, 3_600, true).unwrap();
    let (gpu, _) = panel_prep_gpu::prepare(&bars, &segments, 900, 3_600, true).unwrap();
    assert_outputs_equal("chains", &cpu, &gpu);

    let at = |t: i64| cpu.buckets.iter().find(|b| b.time == t).unwrap().gap;
    assert_eq!(at(0), 0, "listing");
    assert_eq!(at(3 * 3_600), 2, "same chain keeps the hole");
    assert_eq!(at(10 * 3_600), 0, "delisting/listing resets");
    assert_eq!(at(12 * 3_600), 0, "migration resets");
    assert_eq!(at(15 * 3_600), 0, "member=0 resets");
    assert!(cpu.buckets.iter().find(|b| b.time == 3 * 3_600).unwrap().outage);
}

#[test]
fn prefix_replay_at_many_cuts_keeps_earlier_buckets_bit_identical() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut bars = Vec::new();
    for hour in 0..240i64 {
        if hour == 37 {
            continue;
        }
        for k in 0..4 {
            let t = hour * 3_600 + k * 900;
            bars.push(q(t, 50.0 + hour as f64 + k as f64 * 0.01, 2.0));
        }
    }
    let segments = segs(&[bars.len() as u32]);
    let full = panel_prep_gpu::prepare(&bars, &segments, 900, 3_600, true)
        .unwrap()
        .0;
    for cut_hour in (12..240).step_by(12) {
        let cut_time = cut_hour * 3_600;
        let prefix: Vec<PrepBar> = bars.iter().copied().filter(|b| b.time < cut_time).collect();
        let prefix_segments = segs(&[prefix.len() as u32]);
        let got = panel_prep_gpu::prepare(&prefix, &prefix_segments, 900, 3_600, true)
            .unwrap()
            .0;
        let want: Vec<_> = full
            .buckets
            .iter()
            .copied()
            .filter(|b| b.time < cut_time)
            .collect();
        assert_eq!(want.len(), got.buckets.len(), "cut {cut_hour}");
        for (a, b) in want.iter().zip(got.buckets.iter()) {
            assert_eq!(a, b, "bucket at {} changed under cut {cut_hour}", a.time);
        }
    }
}

#[test]
fn block_size_variants_produce_identical_bytes() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let (bars, segments) = mixed_fixture();
    let mut reference: Option<PrepOutput> = None;
    for threads in [32usize, 64, 256, 1024] {
        let mut gpu = PrepGpu::new(&bars, &segments).expect("context");
        assert!(gpu.run_with_threads(900, 3_600, true, threads));
        let out = gpu.output().expect("output");
        match &reference {
            Some(r) => assert_outputs_equal(&format!("threads={threads}"), r, &out),
            None => reference = Some(out),
        }
    }
}

fn canonical_bytes(o: &PrepOutput) -> Vec<u8> {
    let mut v = Vec::with_capacity(o.buckets.len() * 120);
    for b in &o.buckets {
        v.extend_from_slice(&b.time.to_le_bytes());
        v.extend_from_slice(&b.segment.to_le_bytes());
        v.extend_from_slice(&b.chain.to_le_bytes());
        for x in [
            b.open,
            b.high,
            b.low,
            b.close,
            b.volume,
            b.quote_volume,
            b.trades,
            b.taker_buy_base,
            b.taker_buy_quote,
        ] {
            v.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        v.extend_from_slice(&b.cov15m.to_le_bytes());
        v.extend_from_slice(&b.dup_rows.to_le_bytes());
        v.extend_from_slice(&b.gap.to_le_bytes());
        let flags = b.complete as u8
            | (b.traded as u8) << 1
            | (b.partial as u8) << 2
            | (b.duplicate_conflict as u8) << 3
            | (b.outage as u8) << 4;
        v.push(flags);
    }
    for c in &o.conflicts {
        v.extend_from_slice(&c.time.to_le_bytes());
        v.extend_from_slice(&c.segment.to_le_bytes());
        v.extend_from_slice(&c.chain.to_le_bytes());
        v.extend_from_slice(&c.dup_rows.to_le_bytes());
        v.extend_from_slice(&c.first_row.to_le_bytes());
        v.extend_from_slice(&c.len.to_le_bytes());
    }
    v
}

/// 4.78 M decoded rows: 200 symbols with holes, partial hours, repeated months
/// and conflicting copies, a chain change and zero-volume hours.
fn large_fixture() -> (Vec<PrepBar>, Vec<PrepSegment>) {
    let mut bars = Vec::with_capacity(4_900_000);
    let mut lens = Vec::new();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let symbols = 200usize;
    let rows_per = 4_780_000 / symbols;
    for s in 0..symbols {
        let before = bars.len();
        let mut i = 0usize;
        let mut hour = 0i64;
        while bars.len() - before < rows_per {
            let r = next();
            hour += 1 + (r % 97 == 0) as i64 * (1 + (r >> 8) % 5) as i64;
            let cov = if r % 53 == 0 { 1 + (r >> 16) % 3 } else { 4 };
            for k in 0..cov as i64 {
                let vol = if r % 41 == 0 { 0.0 } else { 1.0 + ((r >> 20) % 1000) as f64 / 7.0 };
                let p = 10.0 + s as f64 + (i % 1013) as f64 * 0.37;
                let mut b = q(hour * 3_600 + k * 900, p, vol);
                b.chain = (hour / 4_000) + (s as i64 % 3 == 0) as i64 * 100;
                bars.push(b);
                if r % 61 == 0 && k == 1 {
                    bars.push(b);
                }
                if r % 89 == 0 && k == 2 {
                    let mut c = b;
                    c.close += 0.5;
                    c.volume += 0.125;
                    bars.push(c);
                }
                i += 1;
            }
        }
        lens.push((bars.len() - before) as u32);
    }
    let segments = segs(&lens);
    (bars, segments)
}

#[test]
fn large_block_hash_is_equal_cpu_gpu_cold_warm_and_across_block_sizes() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let (bars, segments) = large_fixture();
    assert!(bars.len() >= 4_780_000, "rows {}", bars.len());
    for keep_partial in [false, true] {
        let t0 = std::time::Instant::now();
        let cpu = cpu_prepare(&bars, &segments, 900, 3_600, keep_partial).unwrap();
        let cpu_s = t0.elapsed().as_secs_f64();
        let want = rcsr::manifest::sha256_hex(&canonical_bytes(&cpu));

        // Cold: fresh context. Warm: second run on the same resident block.
        let mut gpu = PrepGpu::new(&bars, &segments).expect("context");
        let s_up = gpu.stats();
        assert!(gpu.run(900, 3_600, keep_partial));
        let s_run = gpu.stats();
        let cold = gpu.output().expect("cold output");
        let s_out = gpu.stats();
        assert!(gpu.run(900, 3_600, keep_partial));
        let warm = gpu.output().expect("warm output");
        let cold_hash = rcsr::manifest::sha256_hex(&canonical_bytes(&cold));
        let warm_hash = rcsr::manifest::sha256_hex(&canonical_bytes(&warm));
        assert_eq!(cold_hash, want, "cold gpu vs cpu (keep_partial={keep_partial})");
        assert_eq!(warm_hash, want, "warm gpu vs cpu (keep_partial={keep_partial})");

        for threads in [64usize, 1024] {
            let mut g = PrepGpu::new(&bars, &segments).expect("context");
            assert!(g.run_with_threads(900, 3_600, keep_partial, threads));
            let out = g.output().expect("output");
            assert_eq!(
                rcsr::manifest::sha256_hex(&canonical_bytes(&out)),
                want,
                "threads={threads}"
            );
        }
        let (k_s, h_s, d_s) = (
            s_run.kernel_s - s_up.kernel_s,
            s_up.h2d_s,
            s_out.d2h_s - s_run.d2h_s,
        );
        eprintln!(
            "panel_prep rows={} buckets={} conflicts={} keep_partial={} cpu_s={:.3} gpu_kernel_s={:.4} gpu_h2d_s={:.4} gpu_d2h_s={:.4} sha256={}",
            bars.len(),
            cpu.buckets.len(),
            cpu.conflicts.len(),
            keep_partial,
            cpu_s,
            k_s,
            h_s,
            d_s,
            want
        );
    }
}

#[test]
fn nan_is_a_gap_not_zero_and_negative_zero_is_a_conflict() {
    if !would_use_gpu() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut bars: Vec<PrepBar> = (0..4).map(|k| q(k * 900, 100.0, 1.0)).collect();
    bars[1].volume = f64::NAN;
    bars.extend((0..4).map(|k| q(3_600 + k * 900, 100.0, 1.0)));
    // Bit-different zero copies are a conflict, not a collapse.
    let mut z = q(7_200, 100.0, 0.0);
    z.taker_buy_base = 0.0;
    let mut nz = z;
    nz.taker_buy_base = -0.0;
    bars.extend([z, nz]);
    bars.extend((1..4).map(|k| q(7_200 + k * 900, 100.0, 0.0)));
    let segments = segs(&[bars.len() as u32]);
    let cpu = cpu_prepare(&bars, &segments, 900, 3_600, false).unwrap();
    let (gpu, _) = panel_prep_gpu::prepare(&bars, &segments, 900, 3_600, false).unwrap();
    assert_outputs_equal("nan", &cpu, &gpu);
    assert!(cpu.buckets[0].volume.is_nan(), "NaN propagates, never becomes 0");
    assert!(!cpu.buckets[1].volume.is_nan());
    assert_eq!(cpu.conflicts.len(), 1);
    assert!(!cpu.buckets[2].traded, "zero-volume bucket kept with traded=false");
    assert_eq!(cpu.buckets.len(), 3);
}
