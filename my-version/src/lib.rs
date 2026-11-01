//! `rule-and-compass` — a symbolic-regression engine.
//!
//! See ../SPEC.md for the behaviour spec and ../docs/ for an overview.

pub mod atomic;
pub mod attach; // #353: opt-in ptrace attach for hang diagnosis
pub mod backend;
pub mod bench;
pub mod bench_stats;
pub mod bench_suites;
pub mod checkpoint;
pub mod column;
pub mod complexity;
pub mod data;
pub mod diagnostics;
pub mod dispatch; // #290: per-stage CPU/GPU cost model and route ledger
pub mod enrich; // #200: append-only OHLCV feature enrichment
#[cfg(feature = "cuda")]
pub mod enrich_gpu; // #200: resident CUDA enrichment kernels
#[cfg(feature = "cuda")]
pub mod dispatch_gpu; // #290: device calibration probe
#[cfg(feature = "cuda")]
pub mod projected_ring; // #294: bounded pinned ring and direct projected upload
#[cfg(feature = "cuda")]
pub mod mtf_gpu; // #296: device MTF resample and causal join
#[cfg(feature = "cuda")]
pub mod panel_prep_gpu; // #301: fused device duplicate/resample/gap pass
pub mod panel_pit; // #302: exact CPU reference for the PIT membership/MTF as-of joins
#[cfg(feature = "cuda")]
pub mod panel_pit_gpu; // #302: device PIT as-of merge joins
pub mod panel_prep; // #301: exact CPU reference for the panel prep policy
pub mod panel_v2; // #259: default hourly point-in-time panel (snapshot, base, build, validate)
pub mod panel_append; // #260: schema sidecar and time-major family append
pub mod panel_rank; // #493: per-timestamp rank target of a time-major panel
pub mod registry; // #260: versioned feature registry, schema sidecar, views
pub mod derivatives; // reserved, gated funding/OI/basis family
pub mod devlog; // #211: optional development log
pub mod devreport; // #211: `rcsr dev-report`
pub mod eval_soa;
pub mod expr;
pub mod features;
pub mod labels; // #262: long-only horizon, action and triple-barrier labels
pub mod bench_gpu; // #299: device benchmark joins, beta and cross means
pub mod features_gpu; // #297: fused flow / price-action device producers
pub mod fit;
pub mod genome;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod gpu_stats;
pub mod inference; // #264: block bootstrap, DSR, PBO, SPA, BH-FDR
pub mod gui;
pub mod implicit;
pub mod leakage; // #129: differential leakage detectors
pub mod lockbox; // #145: the 2026H2 holdout
pub mod manifest;
pub mod memory;
pub mod metrics;
pub mod multi;
pub mod noise; // #257: Driscoll-Kraay paired-loss noise
pub mod gates; // #257: noise-scaled supported front and verdict
pub mod gate_run; // #257: gates glue for the CLI report
pub mod target_structure; // #257: affine, binned and OLS rungs on the declared structure
pub mod ohlcv;
pub mod pareto;
#[cfg(feature = "cuda")]
pub mod postfix;
pub mod panel; // #145: point-in-time membership and panel
pub mod parse;
pub mod provenance; // #256: build-time commit, binary hash, device identity
pub mod pipeline; // #119: bounded producer/scorer overlap
pub mod prep; // #123: raw archives -> column store
pub mod progress_stats;
pub mod qd; // #120: quality-diversity descriptor grid
pub mod resample; // #92: index-based resampling with a BLB cap
pub mod risk; // #307: exact risk monoids (Sharpe, Sortino, log-growth, drawdown, beta/alpha)
pub mod rng;
pub mod mix; // #471: adaptive edit weights
pub mod scale; // #470: linear scaling wrapper
pub mod run_json; // #482: reader for the parts of run.json later commands use
pub mod schema;
pub mod search;
pub mod selection; // #111: select on train/validation, never the test lockbox
pub mod stability;
pub mod store;
pub mod baselines; // #310: long-only baseline ladder (cash, buy-and-hold, constant, matched-exposure random long)
pub mod ablation; // #266: add-one-family ablation arms, manifests, null columns and verdict
pub mod ladder; // #266: long-only baseline ladder on one panel and one set of folds
pub mod target; // #254: declared target spec
pub mod run_book; // #487: the book a search run selected, read back by the evaluators
pub mod trading; // #263: exact long/flat P&L, Sharpe and log-growth objective
pub mod trading_gpu; // #304/#305: fused long-only P&L, exact threshold and shared mask
pub mod units;
pub mod walkforward;
pub mod wf_protocol; // #264: economics, block inference and nulls for the walk-forward report
pub mod wf_batch; // #309: batch walk-forward P&L over (candidate, window)
pub mod protocol; // #264: windows table, candidate ledger, sealed list, one-read lockbox
pub mod window_stats;
pub mod zip; // #123: minimal stored/deflate ZIP reader
