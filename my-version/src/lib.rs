//! `rule-and-compass` — a symbolic-regression engine.
//!
//! See ../SPEC.md for the behaviour spec and ../docs/ for an overview.

pub mod bench;
pub mod checkpoint;
pub mod column;
pub mod complexity;
pub mod data;
pub mod diagnostics;
pub mod eval_soa;
pub mod expr;
pub mod fit;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod gui;
pub mod implicit;
pub mod metrics;
pub mod multi;
pub mod pareto;
pub mod ohlcv;
pub mod parse;
pub mod progress_stats;
pub mod resample; // #92: index-based resampling with a BLB cap
pub mod rng;
pub mod schema;
pub mod search;
pub mod stability;
pub mod store;
pub mod units;
