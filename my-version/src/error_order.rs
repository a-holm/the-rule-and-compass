//! One order for error values: finite ascending, then `+inf`, then NaN.
//!
//! `partial_cmp(..).unwrap_or(Equal)` is not transitive once a NaN is present,
//! and `total_cmp` puts a negative NaN before every finite value. Every sort
//! that ranks errors uses [`cmp_error`] so the result does not depend on the
//! input order. Equal classes return `Equal`, so the caller's next key decides.

use std::cmp::Ordering;

fn class(v: f64) -> u8 {
    if v.is_nan() {
        2
    } else if v == f64::INFINITY {
        1
    } else {
        0
    }
}

pub fn cmp_error(a: f64, b: f64) -> Ordering {
    match class(a).cmp(&class(b)) {
        Ordering::Equal if class(a) == 0 => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn permutations(items: &mut Vec<f64>, k: usize, out: &mut Vec<Vec<f64>>) {
        if k == items.len() {
            out.push(items.clone());
            return;
        }
        for i in k..items.len() {
            items.swap(k, i);
            permutations(items, k + 1, out);
            items.swap(k, i);
        }
    }

    fn label(v: f64) -> &'static str {
        if v.is_nan() {
            "nan"
        } else if v == f64::INFINITY {
            "inf"
        } else if v == f64::NEG_INFINITY {
            "-inf"
        } else if v == 1.0 {
            "1"
        } else {
            "2"
        }
    }

    /// #622: every input order of a list with NaN, +inf and -inf sorts alike.
    #[test]
    fn every_permutation_sorts_to_the_same_classes() {
        let neg_nan = -f64::NAN;
        let mut items = vec![f64::NAN, 2.0, f64::INFINITY, 1.0, neg_nan, f64::NEG_INFINITY];
        let mut all = Vec::new();
        permutations(&mut items, 0, &mut all);
        assert_eq!(all.len(), 720);
        for mut p in all {
            p.sort_by(|a, b| cmp_error(*a, *b));
            let got: Vec<&str> = p.iter().map(|&v| label(v)).collect();
            assert_eq!(got, ["-inf", "1", "2", "inf", "nan", "nan"]);
        }
    }

    /// #622: the finite class keeps `partial_cmp` semantics, so `-0.0 == 0.0`.
    #[test]
    fn signed_zeros_stay_equal() {
        assert_eq!(cmp_error(-0.0, 0.0), Ordering::Equal);
        assert_eq!(cmp_error(1.0, f64::INFINITY), Ordering::Less);
        assert_eq!(cmp_error(f64::NAN, f64::INFINITY), Ordering::Greater);
        assert_eq!(cmp_error(f64::NAN, -f64::NAN), Ordering::Equal);
    }
}
