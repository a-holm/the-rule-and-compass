//! The `slope` metric differences only rows that are adjacent in the original
//! data order and in the same group (issue #397), whatever the split does to
//! the row order.

use rcsr::data::{self, SplitSpec};
use rcsr::metrics::{self, Metric, SLOPE_MIN_PAIRS};
use rcsr::rng::Mt;
use std::io::Write;
use std::process::{Command, Output, Stdio};

/// Rows `id,g,x,y` with `x = id` so that a sample row maps back to its table
/// row, and `g = id / group_len` as the group column.
fn table(n: usize, group_len: usize) -> data::Table {
    let mut s = String::from("id,g,x,y\n");
    for i in 0..n {
        s.push_str(&format!("{i},{},{i},{}\n", i / group_len, (i as f64).sin()));
    }
    data::parse_table(&s).unwrap()
}

/// Table row id of every row of a prepared sample.
fn ids(p: &data::Prepared, s: &data::Sample) -> Vec<usize> {
    s.x.iter()
        .map(|r| (r[0] * p.stds[0] + p.means[0]).round() as usize)
        .collect()
}

fn prepared(t: &data::Table, spec: &SplitSpec) -> data::Prepared {
    let vars = vec!["x".to_string()];
    data::prepare_full(t, &vars, "y", spec, &mut Mt::new(7)).unwrap()
}

#[test]
fn random_split_pairs_are_adjacent_in_the_original_order() {
    let t = table(400, 400);
    let mut p = prepared(&t, &SplitSpec::default());
    let rows = ids(&p, &p.train);
    assert!(
        rows.windows(2).any(|w| w[1] != w[0] + 1),
        "the random split must shuffle the training rows"
    );
    let n = p.require_slope_pairs().unwrap();
    let pairs = p.train.pairs.clone().unwrap();
    assert_eq!(n, pairs.len());
    assert!(n >= SLOPE_MIN_PAIRS);
    for [a, b] in pairs {
        assert_eq!(rows[b as usize], rows[a as usize] + 1, "non-adjacent pair");
    }
    let test_rows = ids(&p, &p.test);
    for [a, b] in p.test.pairs.clone().unwrap() {
        assert_eq!(test_rows[b as usize], test_rows[a as usize] + 1);
    }
}

#[test]
fn a_group_boundary_is_never_differenced() {
    let t = table(400, 50);
    let g = t.index_of("g").unwrap();
    let spec = SplitSpec { group_col: Some(g), ..SplitSpec::default() };
    let mut p = prepared(&t, &spec);
    p.require_slope_pairs().unwrap();
    let rows = ids(&p, &p.train);
    let pairs = p.train.pairs.clone().unwrap();
    assert!(!pairs.is_empty());
    for [a, b] in pairs {
        let (ra, rb) = (rows[a as usize], rows[b as usize]);
        assert_eq!(rb, ra + 1);
        assert_eq!(ra / 50, rb / 50, "pair {ra}->{rb} crosses a group boundary");
    }
}

#[test]
fn interleaved_groups_pair_within_the_group() {
    let prev = data::row_predecessors(6, None, Some(&[0, 1, 0, 1, 0, 1]));
    assert_eq!(prev, vec![usize::MAX, usize::MAX, 0, 1, 2, 3]);
    assert_eq!(data::adjacent_pairs(&[4, 2, 5, 3], &prev), vec![[1, 0], [3, 2]]);
}

#[test]
fn too_few_adjacent_pairs_is_refused_and_names_temporal() {
    let t = table(40, 40);
    let mut p = prepared(&t, &SplitSpec::default());
    let err = p.require_slope_pairs().unwrap_err();
    assert!(err.contains("--split temporal"), "{err}");
    assert!(err.contains(&SLOPE_MIN_PAIRS.to_string()), "{err}");
}

#[test]
fn score_uses_only_the_listed_pairs() {
    let y = [0.0, 10.0, 1.0, 20.0, 2.0];
    let p = [0.0, 99.0, 1.0, -7.0, 2.0];
    // pairs (0,2) and (2,4): exact slopes; the shuffled neighbours are ignored
    assert_eq!(metrics::score_slope_pairs(&p, &y, &[[0, 2], [2, 4]]), 0.0);
    assert!(metrics::score(&p, &y, Metric::Slope) > 1.0);
    assert!(metrics::score_slope_pairs(&p, &y, &[]).is_infinite());
}

fn run(args: &[&str], csv: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rcsr");
    child.stdin.as_mut().unwrap().write_all(csv.as_bytes()).unwrap();
    child.wait_with_output().expect("wait for rcsr")
}

fn csv(n: usize) -> String {
    let mut s = String::from("t,x,y\n");
    for i in 0..n {
        s.push_str(&format!("{i},{i},{}\n", 0.5 * i as f64 + (i as f64).sin()));
    }
    s
}

fn slope_args<'a>(split: &'a str) -> Vec<&'a str> {
    vec![
        "--data", "-", "--target", "y", "--vars", "x", "--time", "t", "--metric", "slope",
        "--split", split, "--generations", "2", "--pop", "16", "--cpu",
    ]
}

#[test]
fn cli_random_split_with_too_few_pairs_is_a_usage_error() {
    let out = run(&slope_args("random"), &csv(40));
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--split temporal"), "{err}");
}

#[test]
fn cli_temporal_split_keeps_every_consecutive_pair() {
    let out = run(&slope_args("temporal"), &csv(400));
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), err);
    assert!(text.contains("slope: differencing 299 adjacent-row pair(s) among 300"), "{text}");
}

#[test]
fn cli_random_split_with_enough_rows_runs_on_adjacent_pairs() {
    let out = run(&slope_args("random"), &csv(400));
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), err);
    assert!(text.contains("slope: differencing"), "{text}");
    assert!(!text.contains("among 300 training row(s)") || !text.contains("299 adjacent"), "{text}");
}

