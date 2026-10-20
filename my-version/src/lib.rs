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
pub mod eval_soa;
pub mod expr;
pub mod fit;
pub mod genome;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod gui;
pub mod implicit;
pub mod leakage; // #129: differential leakage detectors
pub mod manifest;
pub mod metrics;
pub mod multi;
pub mod pareto;
pub mod parse;
pub mod progress_stats;
pub mod resample; // #92: index-based resampling with a BLB cap
pub mod rng;
pub mod schema;
pub mod search;
pub mod selection; // #111: select on train/validation, never the test lockbox
pub mod stability;
pub mod store;
pub mod units;
pub mod walkforward;
