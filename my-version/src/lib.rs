//! `rule-and-compass` — a symbolic-regression engine.
//!
//! See ../SPEC.md for the behaviour spec and ../docs/ for an overview.

pub mod atomic;
pub mod backend;
pub mod bench;
pub mod checkpoint;
pub mod column;
pub mod complexity;
pub mod data;
pub mod diagnostics;
pub mod enrich; // #200: append-only OHLCV feature enrichment
#[cfg(feature = "cuda")]
pub mod enrich_gpu; // #200: resident CUDA enrichment kernels
#[cfg(feature = "cuda")]
pub mod projected_ring; // #294: bounded pinned ring and direct projected upload
#[cfg(feature = "cuda")]
pub mod mtf_gpu; // #296: device MTF resample and causal join
#[cfg(feature = "cuda")]
pub mod panel_prep_gpu; // #301: fused device duplicate/resample/gap pass
pub mod panel_prep; // #301: exact CPU reference for the panel prep policy
pub mod devlog; // #211: optional development log
pub mod devreport; // #211: `rcsr dev-report`
pub mod eval_soa;
pub mod expr;
pub mod features;
pub mod features_gpu; // #297: fused flow / price-action device producers
pub mod fit;
pub mod genome;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod gpu_stats;
pub mod gui;
pub mod implicit;
pub mod leakage; // #129: differential leakage detectors
pub mod lockbox; // #145: the 2026H2 holdout
pub mod manifest;
pub mod memory;
pub mod metrics;
pub mod multi;
pub mod ohlcv;
pub mod pareto;
#[cfg(feature = "cuda")]
pub mod postfix;
pub mod panel; // #145: point-in-time membership and panel
pub mod parse;
pub mod pipeline; // #119: bounded producer/scorer overlap
pub mod prep; // #123: raw archives -> column store
pub mod progress_stats;
pub mod qd; // #120: quality-diversity descriptor grid
pub mod resample; // #92: index-based resampling with a BLB cap
pub mod rng;
pub mod schema;
pub mod search;
pub mod selection; // #111: select on train/validation, never the test lockbox
pub mod stability;
pub mod store;
pub mod units;
pub mod walkforward;
pub mod zip; // #123: minimal stored/deflate ZIP reader
