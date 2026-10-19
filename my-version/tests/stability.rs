//! Stability-selection tests (issue #20).
//!
//! The claim being tested is the one that makes the feature worth having: on
//! clean data the true term is selected in essentially every resample, while a
//! variable that is pure noise is not. If that fails, a stability score would be
//! meaningless.

use rcsr::rng::Mt;
use rcsr::search::Config;
use rcsr::{data, stability};

/// `y = 2*x1 + 1`, with `x2` pure noise.
fn table(n: usize) -> data::Table {
    let mut text = String::from("x1,x2,y\n");
    let mut s: u64 = 0x1234_5678_9abc_def0;
    let mut noise = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 36) as f64 / (1u64 << 28) as f64) - 1.0
    };
    for i in 0..n {
        let x1 = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        text.push_str(&format!("{:.6},{:.6},{:.6}\n", x1, noise(), 2.0 * x1 + 1.0));
    }
    data::parse_table(&text).expect("table")
}

fn small_config(seed: u32) -> Config {
    Config {
        generations: 30,
        pop: 80,
        islands: 1,
        max_complexity: 10,
        seed,
        fit_iters: 8,
        ..Default::default()
    }
}

#[test]
fn a_true_term_is_stable_and_a_spurious_one_is_not() {
    let vars = vec!["x1".to_string(), "x2".to_string()];
    let t = table(60);
    let (train, test, _, _) = data::prepare(&t, &vars, "y", 0.25, &mut Mt::new(0)).unwrap();

    let st = stability::run(&small_config(5), &train, &test, 6, &vars);
    assert!(
        st.found >= 5,
        "the search should return a solution in almost every resample: {}/{}",
        st.found,
        st.runs
    );

    let freq = |v: usize| -> usize {
        st.variables
            .iter()
            .find(|(i, _)| *i == v)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    };
    assert_eq!(
        freq(0),
        st.found,
        "the true variable must be selected in every run"
    );
    assert!(
        freq(1) * 2 < st.found,
        "the noise variable must not be stable: {} of {}",
        freq(1),
        st.found
    );

    let modal = st.forms.first().expect("at least one form");
    assert!(
        modal.frequency >= 0.5,
        "the modal form should be stable, got {:.2} for {}",
        modal.frequency,
        modal.shape
    );
    assert!(modal.example.contains("x1"), "example: {}", modal.example);
}

#[test]
fn the_summary_is_reproducible_from_the_seed() {
    let vars = vec!["x1".to_string(), "x2".to_string()];
    let t = table(40);
    let (train, test, _, _) = data::prepare(&t, &vars, "y", 0.25, &mut Mt::new(0)).unwrap();

    let a = stability::run(&small_config(11), &train, &test, 3, &vars);
    let b = stability::run(&small_config(11), &train, &test, 3, &vars);
    assert_eq!(a.runs, b.runs);
    assert_eq!(a.found, b.found);
    assert_eq!(a.variables, b.variables);
    let shapes_a: Vec<&str> = a.forms.iter().map(|f| f.shape.as_str()).collect();
    let shapes_b: Vec<&str> = b.forms.iter().map(|f| f.shape.as_str()).collect();
    assert_eq!(shapes_a, shapes_b, "the same seed must give the same forms");
}

/// The pre-#92 resampler, kept verbatim as the bit-exactness oracle.
fn row_clone_resample(s: &data::Sample, rng: &mut Mt) -> data::Sample {
    let n = s.len();
    let mut out = data::Sample::default();
    for _ in 0..n {
        let i = rng.usize_below(n);
        out.x.push(s.x[i].clone());
        out.y.push(s.y[i]);
        if !s.w.is_empty() {
            out.w.push(s.w[i]);
        }
    }
    out
}

#[test]
fn index_resampling_is_bit_identical_to_row_cloning_below_the_cap() {
    // #92: 100 rows is well below the BLB threshold; the draw must match the
    // old per-row clone exactly, including order, duplicates and weights.
    let n = 100;
    let sample = data::Sample::weighted(
        (0..n)
            .map(|i| {
                vec![
                    i as f64 * 0.25,
                    (i as f64 * 0.7).cos(),
                    (i % 11) as f64 / 3.0,
                ]
            })
            .collect(),
        (0..n).map(|i| 3.0 - 0.5 * i as f64).collect(),
        (0..n).map(|i| 1.0 + (i % 5) as f64).collect(),
    );

    let plan = rcsr::resample::plan(sample.len());
    assert_eq!(plan.rows, n);
    assert!(!plan.capped);

    let new = stability::resample(&sample, &mut Mt::new(2024));
    let old = row_clone_resample(&sample, &mut Mt::new(2024));

    assert_eq!(new.len(), old.len());
    assert_eq!(new.x, old.x, "row order or values changed");
    assert_eq!(new.y, old.y);
    assert_eq!(new.w, old.w);
}

#[test]
fn frequency_of_an_unseen_form_is_zero() {
    let vars = vec!["x1".to_string()];
    let t = table(30);
    let (train, test, _, _) = data::prepare(&t, &vars, "y", 0.25, &mut Mt::new(0)).unwrap();
    let st = stability::run(&small_config(3), &train, &test, 2, &vars);
    assert_eq!(st.frequency("nonsense(x99)"), 0.0);
}
