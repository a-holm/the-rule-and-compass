//! Model selection on a validation slice (issue #111).
//!
//! The test block is a **lockbox**: it may be scored once, after a candidate is
//! chosen, and must never influence which candidate wins. Selection therefore
//! ranks on `train`, `validation` or `aicc` only. `Test` exists here for one
//! purpose — to demonstrate, in a test, the optimism of ranking by the block
//! you are about to report.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectBy {
    Train,
    Validation,
    Aicc,
}

impl SelectBy {
    pub fn parse(s: &str) -> Result<SelectBy, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "train" => Ok(SelectBy::Train),
            "validation" | "val" => Ok(SelectBy::Validation),
            "aicc" | "aic" => Ok(SelectBy::Aicc),
            other => Err(format!(
                "unknown --select-by '{}' (expected train, validation or aicc)",
                other
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SelectBy::Train => "train",
            SelectBy::Validation => "validation",
            SelectBy::Aicc => "aicc",
        }
    }
}

/// The scores a candidate exposes. `test` is recorded for reporting and for
/// the leak demonstration; [`select`] never reads it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Candidate {
    pub train: f64,
    pub validation: Option<f64>,
    pub test: Option<f64>,
    pub aicc: f64,
}

fn finite_or_worst(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        f64::INFINITY
    }
}

impl Candidate {
    fn score(self, by: SelectBy) -> f64 {
        match by {
            SelectBy::Train => finite_or_worst(self.train),
            SelectBy::Validation => finite_or_worst(self.validation.unwrap_or(f64::INFINITY)),
            SelectBy::Aicc => finite_or_worst(self.aicc),
        }
    }
}

/// Index of the winning candidate under `by`. `None` only for an empty slice.
///
/// `by == SelectBy::Validation` requires every candidate to carry a validation
/// score; a missing score is treated as infinitely bad, and the caller should
/// have rejected that combination earlier (the CLI does).
pub fn select(candidates: &[Candidate], by: SelectBy) -> Option<usize> {
    candidates
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.score(by)
                .partial_cmp(&b.score(by))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

/// Indices ordered best-first under `by`, stable on ties.
pub fn order(candidates: &[Candidate], by: SelectBy) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..candidates.len()).collect();
    idx.sort_by(|&a, &b| {
        candidates[a]
            .score(by)
            .partial_cmp(&candidates[b].score(by))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_and_aicc_can_disagree() {
        let candidates = [
            Candidate {
                train: 0.10,
                validation: Some(0.50),
                test: Some(0.10),
                aicc: 1.0,
            },
            Candidate {
                train: 0.12,
                validation: Some(0.30),
                test: Some(0.20),
                aicc: 0.9,
            },
        ];
        assert_eq!(select(&candidates, SelectBy::Validation), Some(1));
        assert_eq!(select(&candidates, SelectBy::Aicc), Some(1));
        assert_eq!(select(&candidates, SelectBy::Train), Some(0));
    }

    #[test]
    fn a_missing_validation_score_is_never_selected() {
        let candidates = [
            Candidate {
                train: 0.01,
                validation: None,
                test: Some(0.01),
                aicc: 1.0,
            },
            Candidate {
                train: 0.9,
                validation: Some(0.8),
                test: Some(0.8),
                aicc: 9.0,
            },
        ];
        assert_eq!(select(&candidates, SelectBy::Validation), Some(1));
    }
}
