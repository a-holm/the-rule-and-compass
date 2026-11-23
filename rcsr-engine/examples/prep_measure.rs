// Temporary measurement harness for issue #133 (not part of the library API).
use rcsr::{data, rng::Mt, store};
use std::time::Instant;

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let target = args.get(2).map(|s| s.as_str()).unwrap_or("y60");
    let time = args.get(3).map(|s| s.as_str()).unwrap_or("ts");
    let group = args.get(4).map(|s| s.as_str()).unwrap_or("symbol");
    let t0 = Instant::now();
    let table = if store::is_store_path(path) {
        let fs = store::FileStore::open(path).unwrap();
        fs.to_table().unwrap()
    } else {
        data::read_table(path).unwrap()
    };
    let load = t0.elapsed().as_secs_f64();
    eprintln!("load={:.3}s peak_rss_kb={}", load, rss_kb());
    let names = &table.names;
    let ti = table.index_of(time).unwrap_or(usize::MAX);
    let gi = table.index_of(group).unwrap_or(usize::MAX);
    let yi = table.index_of(target).unwrap();
    let vars: Vec<String> = names
        .iter()
        .enumerate()
        .filter(|(j, _)| {
            table.numeric[*j] && *j != ti && *j != gi && *j != yi
        })
        .map(|(_, n)| n.clone())
        .collect();
    let mut spec = data::SplitSpec {
        mode: if ti == usize::MAX {
            data::SplitMode::Random
        } else {
            data::SplitMode::Temporal
        },
        time_col: if ti == usize::MAX { None } else { Some(ti) },
        group_col: if gi == usize::MAX { None } else { Some(gi) },
        test_frac: 0.25,
        ..Default::default()
    };
    spec.time = data::TimeSpec::default();
    let mut rng = Mt::new(1);
    let mut prep = data::PrepSpec::default();
    match args.get(5).map(|s| s.as_str()) {
        Some("drop") => prep.fill = data::FillMethod::Drop,
        Some("mean") => prep.fill = data::FillMethod::Mean,
        _ => {}
    }
    let t1 = Instant::now();
    let p = data::prepare_full_with(&table, &vars, target, &spec, &mut rng, &prep).unwrap();
    let prep = t1.elapsed().as_secs_f64();
    println!(
        "rows={} cols={} vars={} load={:.3}s prepare={:.3}s total={:.3}s peak_rss_kb={} train={} test={}",
        table.len(), names.len(), vars.len(), load, prep, t0.elapsed().as_secs_f64(),
        rss_kb(), p.train.len(), p.test.len()
    );
}
