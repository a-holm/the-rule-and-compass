//! Stability selection: does the same form come back on resampled data?
//!
//! A relationship that is really in the data is rediscovered when the rows are
//! resampled; a fit to noise is a lottery winner. `--stability B` bootstraps the
//! training rows `B` times, re-runs the identical search on each resample, and
//! counts how often each *structural form* — the expression with every fitted
//! constant collapsed to `c` — and each input variable is selected. A form that
//! appears in most runs is a form the data supports; one that appears once is a
//! coincidence.
//!
//! Runs are reproducible: the resampling RNG and each sub-search's seed are
//! derived from the caller's `--seed`, so the whole summary is deterministic.

use crate::data::Sample;
use crate::expr::{self, Node, Op};
use crate::rng::Mt;
use crate::search::{self, Config};

/// The structural form of a tree: the expression with every fitted constant
/// collapsed to `c` and every input to `x0`, `x1`, ….
///
/// Two spellings of the same *polynomial* structure share one form regardless
/// of how the search parenthesised it: `c + c*x0`, `c - c*x0` and `c*(c + x0)`
/// all reduce to `poly[c, x0]`, because the constants — and therefore the signs
/// — are free parameters the fitter chooses. Without this, identical laws found
/// in different algebraic shapes would split the count and look unstable.
///
/// For trees that are not polynomials (any transcendental, division by a
/// variable, `if`, comparisons, …) the form falls back to the structural
/// spelling, with the arguments of the commutative operators (`add`, `mul`,
/// `min`, `max`) sorted so `a*b` and `b*a` still agree.
pub fn shape(tree: &Node) -> String {
    match polynomial(tree) {
        Some(p) => {
            let terms: Vec<String> = p
                .iter()
                .map(|m| {
                    if m.is_empty() {
                        "c".to_string()
                    } else {
                        m.iter().map(|i| format!("x{}", i)).collect::<Vec<_>>().join("*")
                    }
                })
                .collect();
            format!("poly[{}]", terms.join(", "))
        }
        None => structural(tree),
    }
}

/// A monomial: the input indices it multiplies, with repetition for powers,
/// sorted ascending. The empty monomial is the constant term.
type Monomial = Vec<usize>;

/// Expand `tree` into the set of monomials it contains, or `None` when it is
/// not a polynomial in the inputs. Coefficients are deliberately dropped: every
/// constant is a free parameter (`c`), so two polynomials with the same terms
/// are the same family of laws.
fn polynomial(tree: &Node) -> Option<Vec<Monomial>> {
    let mut m = match tree.op {
        Op::Const => vec![Vec::new()],
        Op::Input => vec![vec![tree.val as usize]],
        Op::Neg => polynomial(child(tree, 0)?)?,
        Op::Add | Op::Sub => {
            let mut v = polynomial(child(tree, 0)?)?;
            v.extend(polynomial(child(tree, 1)?)?);
            v
        }
        Op::Mul => mul(&polynomial(child(tree, 0)?)?, &polynomial(child(tree, 1)?)?),
        // A division is polynomial only by a constant (whose value is free).
        Op::Div => {
            let den = polynomial(child(tree, 1)?)?;
            if den.len() == 1 && den[0].is_empty() {
                polynomial(child(tree, 0)?)?
            } else {
                return None;
            }
        }
        Op::Square => {
            let b = polynomial(child(tree, 0)?)?;
            mul(&b, &b)
        }
        Op::Cube => {
            let b = polynomial(child(tree, 0)?)?;
            mul(&mul(&b, &b), &b)
        }
        // `x^c` is a polynomial only when the exponent is a small integer
        // literal; anything else (including a fitted exponent) is not.
        Op::Pow => {
            let base = polynomial(child(tree, 0)?)?;
            let e = child(tree, 1)?.val;
            let integral = e.is_finite() && e.fract() == 0.0 && (1.0..=8.0).contains(&e);
            if !integral {
                return None;
            }
            let mut acc = vec![Vec::new()];
            for _ in 0..(e as usize) {
                acc = mul(&acc, &base);
            }
            acc
        }
        _ => return None,
    };
    m.sort();
    m.dedup();
    Some(m)
}

fn child(n: &Node, i: usize) -> Option<&Node> {
    n.kids.get(i)
}

/// The product of two polynomial expansions: every monomial against every other.
fn mul(a: &[Monomial], b: &[Monomial]) -> Vec<Monomial> {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for x in a {
        for y in b {
            let mut m = x.clone();
            m.extend_from_slice(y);
            m.sort_unstable();
            out.push(m);
        }
    }
    out
}

/// The structural spelling: constants collapsed to `c`, inputs to `xN`, and the
/// arguments of commutative operators sorted.
fn structural(tree: &Node) -> String {
    match tree.op {
        Op::Const => "c".to_string(),
        Op::Input => format!("x{}", tree.val as usize),
        op => {
            let mut parts: Vec<String> = tree.kids.iter().map(structural).collect();
            if matches!(op, Op::Add | Op::Mul | Op::Min | Op::Max) {
                parts.sort();
            }
            format!("{}({})", op.symbol(), parts.join(", "))
        }
    }
}

/// Input-variable indices appearing in `tree`, ascending and de-duplicated.
pub fn variables(tree: &Node) -> Vec<usize> {
    let mut out = Vec::new();
    collect(tree, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn collect(n: &Node, out: &mut Vec<usize>) {
    if n.op == Op::Input {
        out.push(n.val as usize);
    } else {
        for k in &n.kids {
            collect(k, out);
        }
    }
}

/// A bootstrap resample of `s`: `s.len()` rows drawn **with replacement**.
pub fn resample(s: &Sample, rng: &mut Mt) -> Sample {
    let n = s.len();
    let mut out = Sample::default();
    for _ in 0..n {
        let i = rng.usize_below(n);
        out.x.push(s.x[i].clone());
        out.y.push(s.y[i]);
    }
    out
}

/// One structural form that the search selected, and how often.
#[derive(Clone, Debug)]
pub struct Form {
    pub shape: String,
    pub count: usize,
    /// The simplest thing seen with this shape, in the caller's variable names.
    pub example: String,
    pub complexity: usize,
    /// Highest training-error-free count, i.e. how many runs found it.
    pub frequency: f64,
}

/// The summary of a stability run.
pub struct Stability {
    /// Resamples actually searched.
    pub runs: usize,
    /// Resamples in which the search returned at least one solution.
    pub found: usize,
    /// Structural forms, most frequent first.
    pub forms: Vec<Form>,
    /// `(variable index, count)` for every input, most frequent first.
    pub variables: Vec<(usize, usize)>,
}

impl Stability {
    /// The fraction of runs that selected this structural form.
    pub fn frequency(&self, shape: &str) -> f64 {
        if self.runs == 0 {
            return 0.0;
        }
        self.forms
            .iter()
            .find(|f| f.shape == shape)
            .map(|f| f.count as f64 / self.runs as f64)
            .unwrap_or(0.0)
    }
}

/// Re-run the search on `runs` bootstrap resamples of the training rows.
///
/// The representative of each run is the model the search **recommends**: the
/// member of the resample's Pareto front with the lowest error on the held-out
/// `test` split. Picking by training error instead would let a spurious term win
/// by a numerical hair, because a more flexible model can always shave a little
/// off the residuals it was fitted to; that is exactly the failure this check
/// exists to expose.
///
/// Each sub-search keeps the caller's settings and gets a derived seed, so the
/// result is a deterministic function of `cfg.seed`. Progress callbacks are
/// cleared on the sub-searches so a stability run never pollutes a stream.
pub fn run(cfg: &Config, train: &Sample, test: &Sample, runs: usize, vars: &[String]) -> Stability {
    let mut rng = Mt::new(cfg.seed ^ 0x5741_4c49);
    let mut counts: Vec<Form> = Vec::new();
    let mut var_counts = vec![0usize; vars.len()];
    let mut found = 0usize;
    let mut effective = 0usize;
    for b in 0..runs {
        if train.is_empty() {
            continue;
        }
        let rs = resample(train, &mut rng);
        let mut sub = cfg.clone();
        sub.seed = cfg.seed.wrapping_add(1).wrapping_add((b as u32).wrapping_mul(0x9E37_79B9));
        sub.progress = None;
        let out = search::run(&sub, &rs, test);
        effective += 1;
        let front = out.archive.sorted();
        // `test_errors` is aligned with `front` (search::Outcome contract).
        let pick = front
            .iter()
            .zip(out.test_errors.iter().chain(std::iter::repeat(&f64::INFINITY)))
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
        let Some((sel, _)) = pick else { continue };
        found += 1;
        for v in variables(&sel.tree) {
            if let Some(c) = var_counts.get_mut(v) {
                *c += 1;
            }
        }
        let sh = shape(&expr::simplify(&sel.tree));
        let ex = expr::to_string(&expr::simplify(&sel.tree), vars);
        match counts.iter_mut().find(|f| f.shape == sh) {
            Some(f) => {
                f.count += 1;
                if sel.complexity < f.complexity {
                    f.example = ex;
                    f.complexity = sel.complexity;
                }
            }
            None => counts.push(Form {
                shape: sh,
                count: 1,
                example: ex,
                complexity: sel.complexity,
                frequency: 0.0,
            }),
        }
    }
    counts.sort_by(|a, b| b.count.cmp(&a.count).then(a.shape.cmp(&b.shape)));
    for f in &mut counts {
        f.frequency = if effective == 0 {
            0.0
        } else {
            f.count as f64 / effective as f64
        };
    }
    let mut vs: Vec<(usize, usize)> = var_counts.into_iter().enumerate().collect();
    vs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    Stability {
        runs: effective,
        found,
        forms: counts,
        variables: vs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    #[test]
    fn shape_collapses_constants() {
        // (2*x0 + 1)
        let t = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![constant(2.0), input(0)]),
                constant(1.0),
            ],
        );
        assert_eq!(shape(&t), "poly[c, x0]");
    }

    #[test]
    fn shape_sorts_commutative_arguments() {
        let a = node(Op::Add, vec![constant(1.0), node(Op::Mul, vec![input(0), constant(2.0)])]);
        let b = node(Op::Add, vec![node(Op::Mul, vec![constant(9.0), input(0)]), constant(3.0)]);
        assert_eq!(shape(&a), shape(&b), "a*b and 1 + b*a must share one form");
    }

    /// The search returns algebraically equivalent laws in different shapes; the
    /// signature must not punish it for that.
    #[test]
    fn shape_ignores_algebraic_rearrangement() {
        // c + c*x0
        let plus = node(
            Op::Add,
            vec![constant(1.0), node(Op::Mul, vec![constant(2.0), input(0)])],
        );
        // c - c*x0
        let minus = node(
            Op::Sub,
            vec![constant(1.0), node(Op::Mul, vec![constant(2.0), input(0)])],
        );
        // c * (c + x0)
        let factored = node(
            Op::Mul,
            vec![constant(2.0), node(Op::Add, vec![constant(1.0), input(0)])],
        );
        assert_eq!(shape(&plus), "poly[c, x0]");
        assert_eq!(shape(&minus), shape(&plus), "the sign of a free constant is free");
        assert_eq!(shape(&factored), shape(&plus), "factored forms are the same law");
    }

    #[test]
    fn shape_expands_powers() {
        // x0 * x0 and x0^2 are the same term.
        let product = node(Op::Mul, vec![input(0), input(0)]);
        let power = node(Op::Pow, vec![input(0), constant(2.0)]);
        assert_eq!(shape(&product), "poly[x0*x0]");
        assert_eq!(shape(&power), shape(&product));
    }

    #[test]
    fn shape_keeps_non_polynomials_structural() {
        let s = node(Op::Sin, vec![input(0)]);
        let c = node(Op::Cos, vec![input(0)]);
        assert_ne!(shape(&s), shape(&c), "sin and cos are different laws");
        assert!(shape(&s).starts_with("sin("), "got {}", shape(&s));
    }

    #[test]
    fn variables_are_deduped_and_sorted() {
        let t = node(
            Op::Add,
            vec![
                node(Op::Mul, vec![input(2), input(1)]),
                node(Op::Add, vec![input(1), input(0)]),
            ],
        );
        assert_eq!(variables(&t), vec![0, 1, 2]);
    }

    #[test]
    fn resample_keeps_the_row_count_and_draws_from_the_source() {
        let s = Sample {
            x: vec![vec![0.0], vec![1.0], vec![2.0]],
            y: vec![0.0, 10.0, 20.0],
        };
        let mut rng = Mt::new(7);
        let r = resample(&s, &mut rng);
        assert_eq!(r.len(), s.len());
        for (x, y) in r.x.iter().zip(&r.y) {
            // y is the row label, so it pins the row the sample came from.
            assert_eq!(x[0] * 10.0, *y);
        }
    }

    #[test]
    fn empty_stability_reports_zero_frequency() {
        let st = Stability {
            runs: 0,
            found: 0,
            forms: Vec::new(),
            variables: Vec::new(),
        };
        assert_eq!(st.frequency("+(c, x0)"), 0.0);
    }
}
