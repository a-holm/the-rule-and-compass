//! #1184: the prefix code, the Student-t tail, the evidence and the Occam rule.

use rcsr::expr::{constant, custom_node, input, node, Block, Node, Op};
use rcsr::inference::{normal_cdf, student_t_sf};
use rcsr::occam::{evidence, p_occam, select_occam, Branch, CodeSpec, Composite, Gate, OccamParams, Scored};
use rcsr::selection::RobustMeta;

const TOL: f64 = 1e-12;

fn log2(n: f64) -> f64 {
    n.log2()
}

/// Operators add, mul, greater and custom0 (A = 6), four variables.
fn spec4() -> CodeSpec {
    CodeSpec::v1(&[Op::Add, Op::Mul, Op::Greater, Op::Custom0], 4)
}

#[test]
fn description_bits_match_the_hand_values() {
    let s = spec4();
    let a = log2(6.0);
    let v = 2.0;
    let tier1 = log2(5.0) + log2(41.0);
    let tier4 = log2(5.0) + 1.0 + log2(900_000.0) + log2(19.0);
    let close = |got: f64, want: f64| assert!((got - want).abs() < TOL, "{got} vs {want}");

    // one leaf
    let b = s.description_bits(&input(0)).unwrap();
    close(b.total, a + v);
    close(b.constants, 0.0);

    // greater(x1, 0.5): three nodes, one variable, one tier-1 constant
    let t = node(Op::Greater, vec![input(1), constant(0.5)]);
    let b = s.description_bits(&t).unwrap();
    close(b.structure, 3.0 * a + v);
    close(b.constants, tier1);
    close(b.total, 3.0 * a + v + tier1);

    // mul(add(x0, 3.14159), greater(x2, 1)): seven nodes, two variables, tier 4 and tier 0
    let t = node(
        Op::Mul,
        vec![node(Op::Add, vec![input(0), constant(3.14159)]), node(Op::Greater, vec![input(2), constant(1.0)])],
    );
    let b = s.description_bits(&t).unwrap();
    close(b.total, 7.0 * a + 2.0 * v + tier4 + log2(5.0) + log2(5.0));
    assert_eq!(b.nodes, 7);
    assert_eq!(b.constant_count, 2);

    // two variables
    let t = node(Op::Add, vec![input(0), input(3)]);
    close(s.description_bits(&t).unwrap().total, 3.0 * a + 2.0 * v);

    // custom block: one operator symbol per call, its constant once per slot
    let block = Block {
        name: "f".to_string(),
        params: vec!["a".to_string()],
        body: node(Op::Add, vec![input(0), constant(0.25)]),
    };
    let call = |x: usize| custom_node(Op::Custom0, vec![input(x)], block.clone());
    let t = node(Op::Add, vec![call(0), call(1)]);
    // nodes: add, two calls, two variables = 5; constant 0.25 on tier 2, priced once
    close(s.description_bits(&t).unwrap().total, 5.0 * a + 2.0 * v + log2(5.0) + log2(401.0));
}

#[test]
fn a_tree_outside_the_code_has_no_codeword() {
    let s = spec4();
    assert!(s.description_bits(&node(Op::Sub, vec![input(0), input(1)])).is_err());
    assert!(s.description_bits(&input(4)).is_err());
    assert!(s.description_bits(&constant(0.123_456_789)).is_err());
}

fn trees(n: usize, ops: &[Op], vars: usize, consts: &[f64]) -> Vec<Node> {
    let mut out = Vec::new();
    if n == 1 {
        out.extend((0..vars).map(input));
        out.extend(consts.iter().map(|c| constant(*c)));
        return out;
    }
    let (unary, binary) = (ops[0], ops[1]);
    for k in trees(n - 1, ops, vars, consts) {
        out.push(node(unary, vec![k]));
    }
    for l in 1..n - 1 {
        for a in trees(l, ops, vars, consts) {
            for b in trees(n - 1 - l, ops, vars, consts) {
                out.push(node(binary, vec![a.clone(), b]));
            }
        }
    }
    out
}

#[test]
fn the_code_satisfies_kraft() {
    let s = CodeSpec::v1(&[Op::Neg, Op::Add], 2);
    let consts = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut sum = 0.0;
    let mut count = 0usize;
    for n in 1..=5 {
        for t in trees(n, &[Op::Neg, Op::Add], 2, &consts) {
            sum += 2f64.powf(-s.description_bits(&t).unwrap().total);
            count += 1;
        }
    }
    assert!(count > 1000, "{count}");
    assert!(sum <= 1.0, "Kraft sum {sum}");
    assert!(sum > 0.3, "the bound is not vacuous: {sum}");
}

#[test]
fn the_composite_layer_satisfies_kraft() {
    let s = CodeSpec::v1(&[Op::Neg, Op::Add], 2);
    let mut branches: Vec<(Branch, f64)> = vec![(Branch::Grid, 0.0)];
    for n in 1..=3 {
        for t in trees(n, &[Op::Neg, Op::Add], 2, &[0.0, 1.0]) {
            branches.push((Branch::Tree(s.description_bits(&t).unwrap().total), 0.0));
        }
    }
    // Grid rules: one codeword per (variable, direction, cut) of the grid.
    let grid_words = 2 * 2 * 16;
    let mut words: Vec<Branch> = Vec::new();
    for _ in 0..grid_words {
        words.push(Branch::Grid);
    }
    words.extend(branches.iter().skip(1).map(|(b, _)| *b));
    let w = |c: Composite| 2f64.powf(-s.composite_bits(&c).unwrap());
    let mut sum = 0.0;
    for a in &words {
        sum += w(Composite::C0(*a));
    }
    for g in 0..8 {
        for a in &words {
            sum += w(Composite::C1 { gate: Gate::Menu(g), branch: *a });
            for b in &words {
                sum += w(Composite::C2 { gate: Gate::Menu(g), on: *a, off: *b });
            }
        }
    }
    for g in &words {
        for a in &words {
            sum += w(Composite::C1 { gate: Gate::Searched(*g), branch: *a });
            for b in &words {
                sum += w(Composite::C2 { gate: Gate::Searched(*g), on: *a, off: *b });
            }
        }
    }
    assert!(sum <= 1.0 + 1e-12, "Kraft sum {sum}");
}

#[test]
fn the_gate_menu_has_short_codewords() {
    let s = CodeSpec::v1(&[Op::Neg, Op::Add], 36);
    let base = s.composite_bits(&Composite::C1 { gate: Gate::Menu(0), branch: Branch::Grid }).unwrap();
    for g in 0..8 {
        let b = s.composite_bits(&Composite::C1 { gate: Gate::Menu(g), branch: Branch::Grid }).unwrap();
        assert_eq!(b, base);
    }
    assert!(s.composite_bits(&Composite::C1 { gate: Gate::Menu(8), branch: Branch::Grid }).is_err());
    // type 2 bits + menu 3 bits + cash 1 bit + grid branch (flag 1, variable log2 36, direction 1, cut 4)
    assert!((base - (2.0 + 3.0 + 1.0 + 1.0 + log2(36.0) + 1.0 + 4.0)).abs() < TOL);
}

#[test]
fn composite_costs_match_the_pilot_table() {
    // 14 operators plus two leaf kinds: A = 16.
    let ops = [
        Op::Neg, Op::Exp, Op::Ln, Op::Abs, Op::Sqrt, Op::Tanh, Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Min, Op::Max, Op::Greater, Op::Less,
    ];
    let s = CodeSpec::v1(&ops, 36);
    assert_eq!(s.alphabet_size(), 16);
    let tree5 = log2(16.0) * 5.0 + 2.0 * log2(36.0) + log2(5.0) + log2(41.0);
    let tree3 = 3.0 * 4.0 + 2.0 * log2(36.0);
    let g = Branch::Grid;
    let cases = [
        (Composite::C0(g), 12.2),
        (Composite::C0(Branch::Tree(tree5)), 40.0),
        (Composite::C1 { gate: Gate::Menu(0), branch: g }, 17.2),
        (Composite::C1 { gate: Gate::Menu(0), branch: Branch::Tree(tree5) }, 45.0),
        (Composite::C2 { gate: Gate::Menu(0), on: g, off: g }, 28.3),
        (Composite::C2 { gate: Gate::Menu(0), on: Branch::Tree(tree5), off: Branch::Tree(tree5) }, 84.1),
        (Composite::C1 { gate: Gate::Searched(g), branch: g }, 27.3),
        (Composite::C2 { gate: Gate::Searched(g), on: g, off: g }, 37.5),
        (Composite::C2 { gate: Gate::Searched(Branch::Tree(tree3)), on: g, off: g }, 53.7),
    ];
    for (c, want) in cases {
        let got = s.composite_bits(&c).unwrap();
        assert!((got - want).abs() < 0.1, "{c:?}: {got} vs {want}");
    }
}

#[test]
fn rounding_keeps_the_book_and_lowers_the_bits() {
    let s = CodeSpec::v1(&[Op::Greater], 1);
    let tree = node(Op::Greater, vec![input(0), constant(0.5004)]);
    let words = s.candidates(&tree);
    let tier1 = words.iter().find(|c| c.tier == 1).expect("tier-1 candidate");
    let exact = words.iter().find(|c| c.tier == 4).expect("exact candidate");
    assert_eq!(tier1.tree.kids[1].val, 0.5);
    assert_eq!(exact.tree.kids[1].val, 0.5004);
    assert!(tier1.bits.total < exact.bits.total);
    // The book does not change on rows that avoid [0.5, 0.5004].
    let xs: Vec<f64> = (0..200).map(|i| i as f64 / 100.0 - 1.0).filter(|x| *x < 0.5 || *x > 0.5004).collect();
    for x in xs {
        let a = rcsr::expr::eval(&tier1.tree, &[x]);
        let b = rcsr::expr::eval(&exact.tree, &[x]);
        assert_eq!(a, b, "x = {x}");
    }
    // With equal evidence the rule reports the cheaper code word.
    let series: Vec<f64> = (0..400).map(|i| 0.001 + 0.002 * ((i * 7 % 11) as f64 - 5.0) / 5.0).collect();
    let meta = [RobustMeta { archive_rank: 1, complexity: 3.0, usable: true, objective: 0.0 }];
    let sel = select_occam(&meta, OccamParams { delta: 0.05, min_entries: 1 }, &s, &mut |_| {
        Ok(words
            .iter()
            .filter(|c| c.tier == 1 || c.tier == 4)
            .map(|c| Scored { tier: c.tier, expression_exact: format!("t{}", c.tier), bits: c.bits, series: series.clone(), entries: 50, hold: 1 })
            .collect())
    })
    .unwrap();
    let best = sel.members[0].best.as_ref().unwrap();
    assert_eq!(best.tier, 1, "{best:?}");
}

#[test]
fn the_spec_hash_changes_with_the_code() {
    let a = CodeSpec::v1(&[Op::Add, Op::Mul], 3);
    let b = CodeSpec::v1(&[Op::Mul, Op::Add], 3);
    let c = CodeSpec::v1(&[Op::Add, Op::Mul], 4);
    let d = CodeSpec::v1(&[Op::Add, Op::Mul, Op::Sub], 3);
    assert_eq!(a.sha256(), b.sha256());
    assert_ne!(a.sha256(), c.sha256());
    assert_ne!(a.sha256(), d.sha256());
    assert_eq!(a.file_sha256, rcsr::manifest::sha256_hex(rcsr::occam::SPEC_V1.as_bytes()));
    assert!(a.json().contains(&a.sha256()));
    assert_eq!(a.version, 1);
}

#[test]
fn the_student_t_tail_matches_the_reference() {
    // Reference values: Simpson integration of the density over u = 1/x.
    let pins = [
        (3.0, 4.541, 9.998238064494291e-3),
        (3.0, 40.0, 1.719034039457943e-5),
        (19.0, 2.539, 1.001024139893157e-2),
        (19.0, 10.0, 2.631513081167936e-9),
        (99.0, 5.0, 1.240698006520418e-6),
        (99.0, 6.5, 1.636933231833749e-9),
    ];
    for (df, t, want) in pins {
        let got = student_t_sf(t, df);
        assert!(((got - want) / want).abs() < 1e-8, "df {df} t {t}: {got} vs {want}");
    }
    assert!((student_t_sf(0.0, 7.0) - 0.5).abs() < 1e-15);
    assert!((student_t_sf(-1.3, 12.0) + student_t_sf(1.3, 12.0) - 1.0).abs() < 1e-14);
    for df in [3.0, 19.0, 99.0] {
        for k in 1..=80 {
            let z = k as f64 * 0.1;
            let normal = 1.0 - normal_cdf(z);
            assert!(student_t_sf(z, df) >= normal * (1.0 - 1e-12), "df {df} z {z}");
        }
    }
}

#[test]
fn p_occam_caps_at_one() {
    assert_eq!(p_occam(0.5, 10.0), 1.0);
    assert!((p_occam(1e-6, 10.0) - 1e-6 * 1024.0).abs() < 1e-15);
    assert_eq!(p_occam(0.0, 100.0), 0.0);
}

#[test]
fn evidence_uses_at_least_twenty_blocks() {
    let series: Vec<f64> = (0..1000).map(|i| 0.01 + ((i * 37 % 101) as f64 - 50.0) / 5000.0).collect();
    let ev = evidence(&series, 1).unwrap();
    assert_eq!(ev.block_len, 10);
    assert_eq!(ev.blocks, 100);
    let long_hold = evidence(&series, 400).unwrap();
    assert_eq!(long_hold.block_len, 50);
    assert_eq!(long_hold.blocks, 20);
    let e = evidence(&series[..19], 1).unwrap_err();
    assert!(e.contains("at least 20 blocks") && e.contains("19"), "{e}");
    let flat = evidence(&[0.001; 100], 1).unwrap();
    assert!(flat.z.is_none() && flat.p.is_none());
}

fn meta(rank: usize) -> RobustMeta {
    RobustMeta { archive_rank: rank, complexity: rank as f64, usable: true, objective: 0.0 }
}

fn word(bits: f64, level: f64, entries: usize) -> Scored {
    let series: Vec<f64> = (0..600).map(|i| level + 0.004 * (((i * 53) % 97) as f64 / 48.0 - 1.0)).collect();
    let b = rcsr::occam::Bits { structure: bits, constants: 0.0, total: bits, nodes: 1, variables: 1, constant_count: 0 };
    Scored { tier: 0, expression_exact: "x".to_string(), bits: b, series, entries, hold: 1 }
}

#[test]
fn the_rule_picks_the_lowest_adjusted_value_and_certifies_under_delta() {
    let s = CodeSpec::v1(&[Op::Add], 2);
    let params = OccamParams { delta: 0.05, min_entries: 30 };
    let books = [word(30.0, 0.0004, 100), word(5.0, 0.0004, 100), word(5.0, -0.001, 100), word(5.0, 0.0004, 10)];
    let metas: Vec<RobustMeta> = (1..=4).map(meta).collect();
    let sel = select_occam(&metas, params, &s, &mut |i| Ok(vec![books[i].clone()])).unwrap();
    assert_eq!(sel.chosen, Some(1), "the cheap code wins at equal evidence");
    assert!(sel.members[0].eligible && !sel.members[2].eligible && !sel.members[3].eligible);
    let j = sel.fields_json(None);
    assert!(j.contains("\"rule\":\"occam\"") && j.contains("\"verdict\":\"certified\"") && j.contains("\"chosen_rank\":2"), "{j}");
    assert!(j.contains(&sel.code.sha256()));
    // Priced at 80 bits nothing is certified.
    let dear = [word(80.0, 0.0004, 100)];
    let sel = select_occam(&metas[..1], params, &s, &mut |i| Ok(vec![dear[i].clone()])).unwrap();
    assert_eq!(sel.chosen, None);
    assert!(sel.fields_json(None).contains("\"verdict\":\"no_signal\""));
    assert!(sel.member_json(0).contains("\"p_occam\":1"));
}
