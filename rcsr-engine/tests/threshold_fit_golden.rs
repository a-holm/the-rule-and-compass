//! #1342: a CPU search with `--fit-thresholds` keeps its archive and finalist
//! bit for bit. The pinned values were recorded on the base commit, where every
//! threshold cut is scored by evaluating the whole tree.

use rcsr::data::Sample;
use rcsr::expr::{self, Op};
use rcsr::search::{self, Config};

const ARCHIVE_HASH: u64 = 9326910026421292313;
const FINALIST: &str = "if(x0, (x1 - -1.3036249814810474), (-0.48725127482664243 * x1))";

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for i in 0..n {
        let a = next() * 4.0 - 2.0;
        let b = next() * 3.0 - 1.0;
        let c = (i % 40) as f64 / 40.0;
        let step = if a > 0.35 { 1.5 * b + 1.0 } else { -0.5 * b };
        let gate = if c > 0.6 { 0.75 } else { 0.0 };
        x.push(vec![a, b, c]);
        y.push(step + gate + (next() - 0.5) * 0.02);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

#[test]
fn threshold_search_archive_and_finalist_are_pinned() {
    let train = sample(600);
    let test = sample(200);
    let cfg = Config {
        generations: 8,
        islands: 2,
        pop: 40,
        max_complexity: 14,
        fit_iters: 3,
        patience: 0,
        seed: 11,
        gpu: false,
        fit_thresholds: true,
        ops: Op::parse_list("add,sub,mul,gauss,greater,if"),
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    let vars: Vec<String> = (0..3).map(|i| format!("x{i}")).collect();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for s in &out.archive.items {
        fnv(&mut hash, &s.error.to_bits().to_le_bytes());
        fnv(&mut hash, &s.cost.to_bits().to_le_bytes());
        fnv(&mut hash, &(s.complexity as u64).to_le_bytes());
        fnv(&mut hash, expr::to_string_exact(&s.tree, &vars).as_bytes());
    }
    let finalist = out
        .archive
        .items
        .iter()
        .min_by(|a, b| a.error.total_cmp(&b.error))
        .map(|s| expr::to_string_exact(&s.tree, &vars))
        .expect("an archive member");
    assert!(rcsr::fit::threshold_moves() > 0, "the fixture moves a threshold");
    assert_eq!((hash, finalist.as_str()), (ARCHIVE_HASH, FINALIST));
}
