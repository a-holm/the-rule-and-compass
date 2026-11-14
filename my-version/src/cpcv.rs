//! #1174 (WIP): CPCV splits and backtest-path assembly. Signatures only.

use crate::intervals::{EmbargoSource, IntervalSplit};

type Fail = (i32, String);

pub fn binomial(_n: usize, _k: usize) -> u64 {
    0
}

pub fn combinations(_n: usize, _k: usize) -> Vec<Vec<usize>> {
    Vec::new()
}

pub fn path_count(_n: usize, _k: usize) -> usize {
    0
}

pub fn assignment(_test_sets: &[Vec<usize>], _n: usize) -> Vec<Vec<usize>> {
    Vec::new()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub pieces: Vec<(i64, i64)>,
}

impl Group {
    pub fn start(&self) -> i64 {
        0
    }

    pub fn end(&self) -> i64 {
        0
    }

    pub fn seconds(&self) -> i64 {
        0
    }

    pub fn contains(&self, _t: f64) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Locked {
    pub start: i64,
    pub end: i64,
    pub embargo: i64,
}

pub fn hole(_l: &Locked, _purge: i64) -> (i64, i64) {
    (0, 0)
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub start: i64,
    pub end: i64,
    pub groups: usize,
    pub test_groups: usize,
    pub purge: i64,
    pub embargo: i64,
    pub embargo_source: EmbargoSource,
    pub embargo_column: Option<String>,
    pub validation_share: f64,
    pub locked: Vec<Locked>,
}

#[derive(Clone, Debug)]
pub struct Generated {
    pub groups: Vec<Group>,
    pub test_sets: Vec<Vec<usize>>,
    pub splits: Vec<IntervalSplit>,
    pub holes: Vec<(i64, i64)>,
}

pub fn generate(_p: &Plan) -> Result<Generated, Fail> {
    Err((2, "not implemented".to_string()))
}

pub fn table_csv(_splits: &[IntervalSplit]) -> String {
    String::new()
}

#[derive(Clone, Debug)]
pub struct Scheme {
    pub groups: Vec<Group>,
    pub test_groups: usize,
    pub test_sets: Vec<Vec<usize>>,
    pub ids: Vec<String>,
    pub tests: Vec<Vec<(i64, i64)>>,
}

impl Scheme {
    pub fn sha256(&self) -> String {
        String::new()
    }
}

pub fn read_scheme(_text: &str) -> Result<Scheme, String> {
    Err("not implemented".to_string())
}

pub fn protocol_usage() -> &'static str {
    ""
}

pub fn protocol_cli(_args: &[String]) -> ! {
    eprintln!("error: rcsr protocol cpcv is not implemented");
    std::process::exit(2)
}

pub fn assemble_cli(_args: &[String]) -> Result<(), Fail> {
    Err((2, "--cpcv-assemble is not implemented".to_string()))
}
