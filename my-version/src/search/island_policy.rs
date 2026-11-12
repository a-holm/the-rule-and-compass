#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::expr::{constant, input, node};

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
            islands: 3,
            pop: 24,
            max_complexity: 10,
            fit_iters: 2,
            migration_interval: 3,
            restart_interval: 4,
            patience: 0,
            seed: 11,
            gpu: false,
            ..Default::default()
        }
    }

    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }

    /// Archive hash, exit-checkpoint hash without the build lines, and restarts.
    fn trajectory(cfg: &Config) -> (u64, u64, u64) {
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
        let out = run_resumable(&cfg, &train, &test, None, 0, |cp| text = Some(cp.to_text())).expect("run");
        let text = text.expect("exit checkpoint");
        let kept = text.lines().filter(|l| !l.starts_with("build ")).fold(0xcbf2_9ce4_8422_2325u64, |h, l| fnv(fnv(h, l.as_bytes()), b"\n"));
        (out.archive.hash(), kept, restarts.load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Archive and exit-checkpoint hashes measured at the base.
    const GOLDEN: [(&str, u64, u64); 4] = [("archive", 0, 0), ("island", 0, 0), ("afp", 0, 0), ("qd", 0, 0)];

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
        let mut seen = Vec::new();
        for ((name, cfg), (want_name, want_archive, want_text)) in golden_configs().into_iter().zip(GOLDEN) {
            assert_eq!(name, want_name);
            let (archive, text, restarts) = trajectory(&cfg);
            assert!(restarts > 0, "{name}: the run restarts an island, else the pin proves nothing");
            seen.push(format!("(\"{name}\", {archive:#018x}, {text:#018x}) restarts {restarts}"));
            if (archive, text) != (want_archive, want_text) {
                seen.push(format!("{name}: differs from the base"));
            }
        }
        assert!(!seen.iter().any(|s| s.ends_with("differs from the base")), "{}", seen.join("\n"));
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
        let aggs = AggTable::from_sample(&s);
        let data = EvalSample::new(&s, &cfg);
        let base: Vec<Ind> = (0..cfg.pop)
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
        let run = |front: &[Sol], qd: Option<&QdArchive>, qd_policy: bool, control: u8| {
            let mut isl = base.clone();
            let mut rng = Mt::new(3);
            evolve_island(
                &mut isl, &mut rng, &cfg, &data, &s, 2, &aggs, false, false, front, 0.0, qd, qd_policy, control, None,
                None, 1, &mut MixWindow::new(), None,
            );
            let pop: Vec<(u64, u64)> = isl.iter().map(|i| (expr::structural_hash(&i.tree), i.error.to_bits())).collect();
            (pop, *rng.state().0, rng.state().1)
        };
        let own = run(&[], None, false, 0);
        for control in [0u8, 1] {
            let isolated = run(&front, Some(&qd), true, control);
            assert!(isolated == own, "control {control}: an isolated island drew a --qd-policy parent");
        }
    }
}
