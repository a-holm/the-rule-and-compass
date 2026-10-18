//! `rule-and-compass` — a symbolic-regression engine.
//!
//! See ../SPEC.md for the behaviour spec and ../docs/ for an overview.

pub mod bench;
pub mod checkpoint;
pub mod complexity;
pub mod data;
pub mod diagnostics;
pub mod expr;
pub mod fit;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod metrics;
pub mod pareto;
pub mod parse;
pub mod rng;
pub mod search;
pub mod stability;
