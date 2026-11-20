//! #1238: `rcsr protocol verdict` on hand-made sealed reads. Fixture R of the issue: a
//! fixture lock 2030-01-01..2030-01-07, six daily periods and synthetic series; no market
//! data. Ledger lines that only a locked read writes (`test_read`, `regime_dating`) are
//! appended by hand on the hash chain; `protocol freeze` and the final seal run through the
//! CLI.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use scratch::Scratch;
use std::path::PathBuf;

/// 2030-01-01T00:00:00Z.
const D0: i64 = 1_893_456_000;
const DAY: i64 = 86_400;
const NET: [f64; 6] = [0.02, -0.01, 0.03, -0.02, 0.01, 0.01];
const COST: [f64; 6] = [0.001, 0.0, 0.001, 0.0, 0.001, 0.0];
const EXPOSURE: [f64; 6] = [1.0, 0.5, 1.0, 0.5, 1.0, 0.0];
const BH_NET: [f64; 6] = [0.01, -0.02, 0.02, -0.03, 0.0, 0.01];
const STAMP: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

const REGIMES: &str = "label,start,end\nbull,2030-01-01,2030-01-04\nbear,2030-01-04,2030-01-07\n";
const SUBWINDOWS: &str = "label,start,end\nprior_look,2030-01-01,2030-01-05\npristine,2030-01-05,2030-01-07\n";
const YEARS: &str = "label,start,end\ny1,2030-01-01,2030-01-03\ny2,2030-01-03,2030-01-05\ny3,2030-01-05,2030-01-07\n";

fn sha(bytes: &[u8]) -> String {
    rcsr::manifest::sha256_hex(bytes)
}

fn run_sha(k: usize) -> String {
    sha(format!("run-{k}").as_bytes())
}

fn candidate_id(tag: &str) -> String {
    sha(format!("candidate-{tag}").as_bytes())
}

fn arr(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = rcsr_cmd::rcsr().args(args).output().unwrap();
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

fn ok(r: &(i32, String, String)) {
    assert_eq!(r.0, 0, "{}{}", r.1, r.2);
}

/// Appends `body` as one ledger line chained to the last line (the first line names 64 zeros).
fn append(ledger: &str, body: &str) {
    let text = std::fs::read_to_string(ledger).unwrap_or_default();
    let head = text.lines().last().map_or("0".repeat(64), |l| sha(l.as_bytes()));
    std::fs::write(ledger, format!("{text}{{\"prev\":\"{head}\",{body}}}\n")).unwrap();
}

/// The bodies of the ledger lines (without `prev`), in order.
fn bodies(ledger: &str) -> Vec<String> {
    std::fs::read_to_string(ledger).unwrap().lines().map(|l| l[75..l.len() - 1].to_string()).collect()
}

/// One sealed read of fixture R and the switches the refusal tests turn.
#[derive(Clone)]
struct ReadFx {
    tag: String,
    net: Vec<f64>,
    cost: Option<Vec<f64>>,
    p: f64,
    run: Option<String>,
    bps: f64,
    ppy: f64,
    test_read: bool,
    sealed: bool,
    recorded: bool,
}

impl ReadFx {
    fn r(tag: &str, run: usize) -> ReadFx {
        ReadFx {
            tag: tag.to_string(),
            net: NET.to_vec(),
            cost: Some(COST.to_vec()),
            p: 0.01,
            run: Some(run_sha(run)),
            bps: 10.0,
            ppy: 365.0,
            test_read: true,
            sealed: true,
            recorded: true,
        }
    }

    /// The hashed report text and the file text with the members outside the hash.
    fn texts(&self) -> (String, String) {
        let n = self.net.len();
        let time: Vec<f64> = (0..n).map(|d| (D0 + DAY * d as i64) as f64).collect();
        let cost = self.cost.as_ref().map_or(String::new(), |c| format!(",\"period_cost\":[{}]", arr(c)));
        let run = self.run.as_ref().map_or(String::new(), |r| format!(",\"run_sha256\":\"{r}\""));
        let body = format!(
            "{{\"tool\":\"rcsr\",\"mode\":\"walkforward\",\"expression\":\"x\",\"cost_model\":{{\"per_side_bps\":{bps},\"round_trip_bps\":{rt}}},\
\"annualization\":{{\"periods_per_year\":{ppy},\"method\":\"explicit\",\"source\":\"explicit\",\"period_seconds\":86400}},\
\"selection\":{{\"selected_on\":\"validation\",\"candidate_count\":1{run}}},\
\"pooled\":{{\"periods\":{n},\"max_drawdown\":-0.02,\"period_net\":[{net}],\"period_time\":[{time}],\"period_exposure\":[{e}],\"period_bh\":[{b}]{cost},\
\"period_entry_stamp\":[{st}],\"period_bh_net\":[{b}]}},\"split\":{{\"purge_seconds\":0}},\
\"inference\":{{\"bootstrap\":{{\"n\":{n},\"block_len\":2,\"draws\":200,\"seed\":7,\"p_mean_positive\":{p}}}}}}}",
            bps = self.bps,
            rt = 2.0 * self.bps,
            ppy = self.ppy,
            net = arr(&self.net),
            time = arr(&time),
            e = arr(&EXPOSURE),
            b = arr(&BH_NET),
            st = arr(&STAMP),
            p = self.p,
        );
        let protocol = format!(
            ",\"protocol\":{{\"lockbox\":{{\"rows\":{n}}},\"candidate_id\":\"{}\",\"report_sha256\":\"{}\",\"test_read\":{},\"ledger\":null}}",
            candidate_id(&self.tag),
            sha(body.as_bytes()),
            self.test_read
        );
        let file = format!("{}{protocol},\"ledger\":{{\"command\":\"walkforward\",\"wall_s\":1.5}}}}", &body[..body.len() - 1]);
        (body, file)
    }
}

/// One pick of the frozen candidate table.
#[derive(Clone)]
struct PickFx {
    run: Option<String>,
    s4: f64,
    p1: f64,
    cap: f64,
    dsr: f64,
}

impl PickFx {
    fn r(run: usize) -> PickFx {
        PickFx { run: Some(run_sha(run)), s4: 1.0, p1: 0.5, cap: 0.2, dsr: 0.97 }
    }
}

#[derive(Clone)]
struct Fx {
    reads: Vec<ReadFx>,
    picks: Vec<PickFx>,
    /// SPA p and PBO of the thresholds table and of a frozen development table.
    table: (f64, f64),
    development: Option<(f64, f64)>,
    freeze_subwindows: bool,
    /// Population files the table names; the verdict never opens them.
    population: Vec<String>,
    /// #1269: `spa.p_family` of a thresholds or development table written with `--family-table`.
    table_family: Option<f64>,
    development_family: Option<f64>,
}

impl Fx {
    /// Fixture R: one read, pick 1 with S4 floor 1.0, P1 floor 0.5, cap 0.20 and DSR 0.97,
    /// a development table with SPA p 0.03 and PBO 0.10 (the thresholds table holds 0.5 and
    /// 0.9, so a verdict that read it for D2 or D3 would fail them).
    fn r() -> Fx {
        Fx {
            reads: vec![ReadFx::r("a", 1)],
            picks: vec![PickFx::r(1)],
            table: (0.5, 0.9),
            development: Some((0.03, 0.10)),
            freeze_subwindows: true,
            population: Vec::new(),
            table_family: None,
            development_family: None,
        }
    }
}

/// #1269: the `trials` member with a family block and the `spa.p_family` member of a table
/// written with `--family-table`, or two empty strings; NaN gives the family block without
/// `spa.p_family`.
fn family_members(p_family: Option<f64>) -> (String, String) {
    p_family.map_or((String::new(), String::new()), |p| {
        (
            ",\"trials\":{\"nominal\":4,\"n_eff\":3,\"family\":{\"tables\":[{\"sha256\":\"f1\",\"n_eff\":5,\"nominal\":6,\"spa_p\":0.4}],\
\"sha256\":\"f\",\"n_eff_total\":8,\"nominal_total\":10}}"
                .to_string(),
            if p.is_nan() { String::new() } else { format!(",\"p_family\":{p}") },
        )
    })
}

fn table_json(picks: &[PickFx], (spa, pbo): (f64, f64), population: &[String], family: Option<f64>) -> String {
    let files: Vec<String> =
        population.iter().enumerate().map(|(i, f)| format!("{{\"file\":\"{f}\",\"sha256\":\"{}\",\"path\":{},\"periods\":6}}", sha(f.as_bytes()), i + 1)).collect();
    let picks_json: Vec<String> = picks
        .iter()
        .enumerate()
        .map(|(k, p)| {
            format!(
                "{{\"pick\":{},\"row\":{k},\"label\":\"row{k}\",\"dsr\":{{\"n_eff\":3,\"value\":{}}},\"composite\":null,\"freeze\":{{\"schema\":\"rcsr-trials-freeze/1\",\
\"configuration\":\"F{k}\",\"cost_bps\":10,\"periods_per_year\":365,\"period_seconds\":86400,\"population\":{{\"groups\":3,\"test_groups\":2,\"paths\":{},\
\"files\":[{}]}},\"s4_floor\":{{\"window_days\":3,\"percentile\":10,\"windows\":8,\"rank\":1,\"value\":{}}},\"p1_floor\":{{\"window_days\":2,\"percentile\":10,\
\"windows\":10,\"rank\":1,\"value\":{}}},\"drawdown_cap\":{{\"window_days\":3,\"percentile\":90,\"windows\":8,\"rank\":8,\"percentile_value\":0.01,\"floor\":0.2,\
\"value\":{}}},\"run\":\"final-{k}.run.json\",\"run_sha256\":{}}}}}",
                k + 1,
                p.dsr,
                population.len(),
                files.join(","),
                p.s4,
                p.p1,
                p.cap,
                p.run.as_ref().map_or("null".to_string(), |r| format!("\"{r}\""))
            )
        })
        .collect();
    let rows: Vec<String> = (0..picks.len())
        .map(|k| format!("{{\"row\":{k},\"label\":\"row{k}\",\"occam\":{{\"run\":\"o.json\",\"source\":\"occam\",\"bits\":{},\"p_occam\":0.25,\"bit_budget\":40}}}}", 12 + k))
        .collect();
    let (trials, p_family) = family_members(family);
    format!(
        "{{\"schema\":\"rcsr-trials-table/1\",\"rule\":{{\"schema\":\"rcsr-trials-rule/1\",\"pick\":{},\"sha256\":\"{rule}\"}},\
\"freeze\":{{\"schema\":\"rcsr-trials-freeze/1\",\"rule_sha256\":\"{rule}\",\"map\":\"map.json\",\"map_sha256\":\"{map}\",\"method\":{{}}}}{trials},\
\"picks\":[{}],\"spa\":{{\"benchmark\":\"cash\",\"p_consistent\":{spa}{p_family}}},\"pbo\":{{\"value\":{pbo}}},\"rows\":[{}]}}\n",
        picks.len(),
        picks_json.join(","),
        rows.join(","),
        rule = sha(b"rule"),
        map = sha(b"map"),
    )
}

fn development_json((spa, pbo): (f64, f64), family: Option<f64>) -> String {
    let (trials, p_family) = family_members(family);
    format!(
        "{{\"schema\":\"rcsr-trials-table/1\"{trials},\"picks\":[],\"spa\":{{\"benchmark\":\"ladder\",\"p_consistent\":{spa}{p_family}}},\"pbo\":{{\"value\":{pbo}}},\"rows\":[]}}\n"
    )
}

struct Setup {
    _scratch: Scratch,
    dir: PathBuf,
    ledger: String,
    sealed: String,
    thresholds: String,
    development: Option<String>,
    subwindows: String,
    years: String,
    regimes: String,
    reads: Vec<String>,
    bodies: Vec<String>,
}

impl Setup {
    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }
}

/// Writes the frozen files, the ledger (candidates, freeze events, final seal, reads and the
/// after-read dating of the lock calendar) and the read reports of `fx`.
fn setup(tag: &str, fx: &Fx) -> Setup {
    let scratch = Scratch::new(tag);
    let dir = scratch.path().to_path_buf();
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let ledger = p("ledger.jsonl");
    let population: Vec<String> = fx.population.iter().map(|f| p(f)).collect();
    for f in &population {
        std::fs::write(f, "{\"mode\":\"cpcv_path\"}\n").unwrap();
    }
    let thresholds = p("table.json");
    std::fs::write(&thresholds, table_json(&fx.picks, fx.table, &population, fx.table_family)).unwrap();
    let development = fx.development.map(|d| {
        let path = p("development.json");
        std::fs::write(&path, development_json(d, fx.development_family)).unwrap();
        path
    });
    let (subwindows, years, regimes) = (p("subwindows.csv"), p("years.csv"), p("regimes_lock.csv"));
    std::fs::write(&subwindows, SUBWINDOWS).unwrap();
    std::fs::write(&years, YEARS).unwrap();
    std::fs::write(&regimes, REGIMES).unwrap();
    for r in &fx.reads {
        append(
            &ledger,
            &format!(
                "\"event\":\"candidate\",\"id\":\"{}\",\"spec_sha256\":\"spec-{t}\",\"data_sha256\":\"data\",\"build_commit\":\"build\",\
\"report_sha256\":\"dev-{t}\",\"test_read\":false,\"expression\":\"x\"",
                candidate_id(&r.tag),
                t = r.tag
            ),
        );
    }
    let mut frozen = vec![("thresholds", thresholds.clone()), ("years", years.clone())];
    if fx.freeze_subwindows {
        frozen.push(("subwindows", subwindows.clone()));
    }
    if let Some(d) = &development {
        frozen.push(("development", d.clone()));
    }
    for (role, file) in &frozen {
        ok(&rcsr(&["protocol", "freeze", "--ledger", &ledger, "--role", role, "--file", file]));
    }
    let sealed = p("locked_candidates.json");
    let ids: Vec<String> = fx.reads.iter().filter(|r| r.sealed).map(|r| candidate_id(&r.tag)).collect();
    let mut seal = vec!["protocol", "seal", "--final", "--max-reads", "3", "--ledger", ledger.as_str(), "--out", sealed.as_str()];
    for id in &ids {
        seal.extend_from_slice(&["--id", id.as_str()]);
    }
    ok(&rcsr(&seal));
    let (mut reads, mut hashed, mut count) = (Vec::new(), Vec::new(), 0);
    for (k, r) in fx.reads.iter().enumerate() {
        let (body, file) = r.texts();
        if r.recorded {
            count += 1;
            append(
                &ledger,
                &format!(
                    "\"event\":\"test_read\",\"id\":\"{}\",\"report_sha256\":\"{}\",\"read_count\":{count},\"test_read\":true",
                    candidate_id(&r.tag),
                    sha(body.as_bytes())
                ),
            );
        }
        let path = p(&format!("read-{}.json", k + 1));
        std::fs::write(&path, file).unwrap();
        reads.push(path);
        hashed.push(body);
    }
    append(&ledger, &format!("\"event\":\"regime_dating\",\"calendar_sha256\":\"{}\",\"params_sha256\":\"{}\",\"test_read\":false", sha(REGIMES.as_bytes()), sha(b"params")));
    Setup { _scratch: scratch, dir, ledger, sealed, thresholds, development, subwindows, years, regimes, reads, bodies: hashed }
}

fn verdict(s: &Setup) -> (i32, String, String) {
    let (out, md) = (s.path("verdict.json"), s.path("verdict.md"));
    let mut a: Vec<&str> = vec![
        "protocol", "verdict", "--ledger", &s.ledger, "--sealed", &s.sealed, "--thresholds", &s.thresholds, "--subwindows", &s.subwindows,
        "--pristine-label", "pristine", "--years", &s.years, "--regimes", &s.regimes, "--out", &out, "--markdown", &md,
    ];
    if let Some(d) = &s.development {
        a.extend_from_slice(&["--development", d]);
    }
    for r in &s.reads {
        a.extend_from_slice(&["--read", r]);
    }
    rcsr(&a)
}

fn result(s: &Setup) -> Json {
    Json::parse(std::fs::read_to_string(s.path("verdict.json")).unwrap().trim()).expect("verdict JSON")
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    path.iter().fold(j, |cur, k| cur.get(k).unwrap_or_else(|| panic!("field '{}' missing in {:?}", k, cur)))
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{} is not a number: {:?}", path.join("."), at(j, path)))
}

fn text<'a>(j: &'a Json, path: &[&str]) -> &'a str {
    at(j, path).as_str().unwrap_or_else(|| panic!("{} is not a string", path.join(".")))
}

fn items(j: &Json) -> &[Json] {
    match j {
        Json::Arr(a) => a,
        other => panic!("not an array: {other:?}"),
    }
}

fn candidate(j: &Json, k: usize) -> &Json {
    &items(at(j, &["candidates"]))[k]
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12, "{what}: {got} vs {want}");
}

fn refused(tag: &str, fx: &Fx, needle: &str) {
    let s = setup(tag, fx);
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 2, "{tag}: {out}{err}");
    assert!(err.contains(needle), "{tag}: '{needle}' not in: {err}");
    assert!(!std::path::Path::new(&s.path("verdict.json")).exists(), "{tag}: a refused verdict writes nothing");
}

#[test]
fn verdict_gate_values_match_hand_fixture() {
    let s = setup("verdict-r", &Fx::r());
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 0, "{out}{err}");
    let j = result(&s);
    assert_eq!(text(&j, &["schema"]), "rcsr-verdict/1");
    let c = candidate(&j, 0);
    let g = |gate: &str, key: &str| num(at(c, &["gates", gate]), &[key]);
    close(g("S1", "net10"), 2.4333333333333333, "S1 net10");
    close(g("S1", "net20"), 2.2508333333333333, "S1 net20");
    close(g("S2", "alpha_bh"), 2.1291666666666667, "S2");
    close(g("S3", "max_drawdown"), -0.02, "S3");
    close(g("S3", "cap"), 0.2, "S3 cap");
    close(g("S4", "net10"), 2.4333333333333333, "S4");
    close(g("S4", "floor"), 1.0, "S4 floor");
    close(g("S5", "p"), 0.01, "S5");
    close(g("R1", "net10"), 4.8666666666666667, "R1 net10");
    close(g("R1", "net20"), 4.6233333333333333, "R1 net20");
    close(g("R1", "alpha_bh"), 2.4333333333333333, "R1 alpha");
    close(g("R1", "periods"), 3.0, "R1 periods");
    close(g("R2", "excess"), 2.4333333333333333, "R2 excess");
    close(g("R2", "alpha_bh"), 1.825, "R2 alpha");
    close(g("P1", "net10"), 3.65, "P1");
    close(g("P1", "floor"), 0.5, "P1 floor");
    close(g("P1", "periods"), 2.0, "P1 periods");
    close(g("D1", "dsr"), 0.97, "D1");
    close(g("D2", "p"), 0.03, "D2 reads the development table");
    close(g("D3", "pbo"), 0.10, "D3 reads the development table");
    for gate in ["S1", "S2", "S3", "S4", "S5", "R1", "R2", "P1", "D1", "D2", "D3"] {
        assert_eq!(at(c, &["gates", gate, "pass"]), &Json::Bool(true), "{gate}");
    }
    assert_eq!(text(c, &["verdict"]), "good");
    assert_eq!(text(c, &["role"]), "primary");
    assert_eq!(text(c, &["candidate_id"]), candidate_id("a"));
    assert_eq!(text(c, &["run_sha256"]), run_sha(1));
    assert_eq!(text(c, &["read_report_sha256"]), sha(s.bodies[0].as_bytes()));
    let file_sha = |p: &str| sha(&std::fs::read(p).unwrap());
    assert_eq!(text(&j, &["frozen", "thresholds"]), file_sha(&s.thresholds));
    assert_eq!(text(&j, &["frozen", "development"]), file_sha(s.development.as_ref().unwrap()));
    assert_eq!(text(&j, &["frozen", "subwindows"]), file_sha(&s.subwindows));
    assert_eq!(text(&j, &["frozen", "years"]), file_sha(&s.years));
    assert_eq!(text(&j, &["regimes_calendar_sha256"]), file_sha(&s.regimes));
    assert_eq!(text(&j, &["sealed_sha256"]), file_sha(&s.sealed));
    assert_eq!(text(&j, &["rule_sha256"]), sha(b"rule"));
    // Report-only lines: the full lock, the prior-look part and the pristine part, then the
    // years and the strata of the lock calendar; cash is 0.
    let parts = items(at(c, &["parts"]));
    let labels: Vec<&str> = parts.iter().map(|p| text(p, &["label"])).collect();
    assert_eq!(labels, ["full", "prior_look", "pristine"]);
    close(num(&parts[0], &["net10"]), 2.4333333333333333, "full net10");
    close(num(&parts[0], &["entries"]), 3.0, "full entries");
    close(num(&parts[1], &["net10"]), 1.825, "prior-look net10");
    close(num(&parts[1], &["bh"]), -1.825, "prior-look buy-and-hold");
    close(num(&parts[2], &["net10"]), 3.65, "pristine net10");
    for p in parts {
        assert_eq!(num(p, &["cash"]), 0.0);
    }
    let names = |key: &str| items(at(c, &[key])).iter().map(|p| text(p, &["label"]).to_string()).collect::<Vec<_>>();
    assert_eq!(names("years"), ["y1", "y2", "y3"]);
    assert_eq!(names("strata"), ["bull", "bear"]);
    close(num(&items(at(c, &["strata"]))[0], &["net10"]), 4.8666666666666667, "bull stratum");
    let ro = at(c, &["report_only"]);
    assert_eq!((num(ro, &["bits"]), num(ro, &["p_occam"])), (12.0, 0.25));
    assert!(matches!(at(ro, &["p_family"]), Json::Null));
    assert_eq!(num(ro, &["dsr_trials"]), 51.0);
    let dsr: Vec<&str> = items(at(ro, &["dsr"])).iter().map(|d| text(d, &["part"])).collect();
    assert_eq!(dsr, ["full", "prior_look"]);
    assert_eq!(at(&j, &["secondaries", "tested"]), &Json::Bool(false));
    // Same inputs, same bytes.
    let (json1, md1) = (std::fs::read(s.path("verdict.json")).unwrap(), std::fs::read(s.path("verdict.md")).unwrap());
    let (code, _, err) = verdict(&s);
    assert_eq!(code, 0, "{err}");
    assert_eq!(std::fs::read(s.path("verdict.json")).unwrap(), json1);
    assert_eq!(std::fs::read(s.path("verdict.md")).unwrap(), md1);
    assert!(String::from_utf8(md1).unwrap().contains("good"));
}

fn class_of(tag: &str, fx: &Fx) -> (String, Json) {
    let s = setup(tag, fx);
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 0, "{tag}: {out}{err}");
    let j = result(&s);
    (text(candidate(&j, 0), &["verdict"]).to_string(), candidate(&j, 0).clone())
}

#[test]
fn verdict_classes_follow_design_table() {
    let mut fx = Fx::r();
    fx.reads[0].p = 0.08;
    fx.picks[0].dsr = 0.85;
    let (class, c) = class_of("verdict-promising", &fx);
    assert_eq!(class, "promising");
    assert_eq!((at(&c, &["gates", "S5", "pass"]), at(&c, &["gates", "D1", "pass"])), (&Json::Bool(false), &Json::Bool(false)));

    let mut fx = Fx::r();
    fx.picks[0].s4 = 3.0;
    let (class, c) = class_of("verdict-s4", &fx);
    assert_eq!(class, "not_confirmed");
    assert_eq!(at(&c, &["gates", "S4", "pass"]), &Json::Bool(false));

    let mut fx = Fx::r();
    fx.reads[0].net = NET.iter().map(|x| -x).collect();
    assert_eq!(class_of("verdict-negated", &fx).0, "falsified");

    let mut fx = Fx::r();
    fx.reads[0].net = vec![0.01, -0.01, 0.0, 0.0, 0.0, 0.0];
    let (class, c) = class_of("verdict-zero", &fx);
    assert_eq!(num(&c, &["gates", "S1", "net10"]), 0.0);
    assert_eq!(class, "not_confirmed");
}

/// #1269: a table written with `--family-table` carries `trials.family`; D2 then reads its
/// Bonferroni `spa.p_family` and names the source.
#[test]
fn verdict_d2_reads_the_family_p() {
    let fx = Fx { development: Some((0.02, 0.10)), development_family: Some(0.08), ..Fx::r() };
    let s = setup("verdict-d2-family", &fx);
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 0, "{out}{err}");
    let c = candidate(&result(&s), 0).clone();
    let d2 = at(&c, &["gates", "D2"]);
    close(num(d2, &["p"]), 0.08, "D2 reads spa.p_family");
    assert_eq!(text(d2, &["p_source"]), "family");
    assert_eq!(at(d2, &["pass"]), &Json::Bool(false));
    assert_eq!(text(&c, &["verdict"]), "not_confirmed", "D2 fails, so neither good nor promising");
    let md = std::fs::read_to_string(s.path("verdict.md")).unwrap();
    assert!(md.contains("| D2 | p 0.08, p_source family,"), "{md}");

    let (class, c) = class_of("verdict-d2-consistent", &Fx { development_family: None, ..fx.clone() });
    let d2 = at(&c, &["gates", "D2"]);
    close(num(d2, &["p"]), 0.02, "D2 reads spa.p_consistent");
    assert_eq!(text(d2, &["p_source"]), "consistent");
    assert_eq!(at(d2, &["pass"]), &Json::Bool(true));
    assert_eq!(class, "good");

    // Without a frozen development table D2 reads the family p of the thresholds table.
    let (class, c) = class_of("verdict-d2-thresholds", &Fx { development: None, table: (0.02, 0.10), table_family: Some(0.08), ..Fx::r() });
    let d2 = at(&c, &["gates", "D2"]);
    close(num(d2, &["p"]), 0.08, "D2 reads the thresholds table's spa.p_family");
    assert_eq!(text(d2, &["p_source"]), "family");
    assert_eq!(class, "not_confirmed");

    // A family block without spa.p_family fails D2 with a reason.
    let (class, c) = class_of("verdict-d2-no-p", &Fx { development_family: Some(f64::NAN), ..fx.clone() });
    let d2 = at(&c, &["gates", "D2"]);
    assert!(matches!(at(d2, &["p"]), Json::Null), "{d2:?}");
    assert_eq!(text(d2, &["p_source"]), "family");
    assert_eq!(text(d2, &["reason"]), "missing in the frozen table");
    assert_eq!(class, "not_confirmed");
}

#[test]
fn verdict_secondaries_holm() {
    let mut fx = Fx::r();
    fx.reads = vec![ReadFx::r("a", 1), ReadFx { p: 0.04, ..ReadFx::r("b", 2) }, ReadFx { p: 0.01, ..ReadFx::r("c", 3) }];
    fx.picks = vec![PickFx::r(1), PickFx::r(2), PickFx::r(3)];
    let s = setup("verdict-holm", &fx);
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 0, "{out}{err}");
    let j = result(&s);
    assert_eq!(text(candidate(&j, 0), &["verdict"]), "good");
    let sec = at(&j, &["secondaries"]);
    assert_eq!(at(sec, &["tested"]), &Json::Bool(true));
    let holm = items(at(sec, &["holm"]));
    assert_eq!(holm.len(), 2);
    assert_eq!((num(&holm[0], &["pick"]), num(&holm[1], &["pick"])), (2.0, 3.0));
    close(num(&holm[0], &["p"]), 0.04, "p pick 2");
    close(num(&holm[0], &["p_holm"]), 0.04, "Holm pick 2");
    close(num(&holm[1], &["p_holm"]), 0.02, "Holm pick 3");
    for (k, want) in [(1, 0.04), (2, 0.02)] {
        let c = candidate(&j, k);
        assert_eq!(text(c, &["role"]), "secondary");
        assert_eq!(at(c, &["tested"]), &Json::Bool(true));
        close(num(c, &["p_holm"]), want, "candidate p_holm");
    }

    fx.picks[0].s4 = 3.0;
    let s = setup("verdict-holm-untested", &fx);
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 0, "{out}{err}");
    let j = result(&s);
    assert_eq!(text(candidate(&j, 0), &["verdict"]), "not_confirmed", "secondaries never change the primary");
    let sec = at(&j, &["secondaries"]);
    assert_eq!(at(sec, &["tested"]), &Json::Bool(false));
    assert!(text(sec, &["reason"]).contains("primary"));
    assert!(items(at(sec, &["holm"])).is_empty());
    for k in [1, 2] {
        let c = candidate(&j, k);
        assert_eq!(at(c, &["tested"]), &Json::Bool(false));
        assert!(matches!(at(c, &["p_holm"]), Json::Null));
        assert_eq!(text(c, &["verdict"]), "good", "their own numbers stay as context");
    }
}

#[test]
fn verdict_refuses_unfrozen_or_changed_inputs() {
    // The thresholds file changed by one byte after its freeze.
    let s = setup("verdict-changed", &Fx::r());
    let t = std::fs::read_to_string(&s.thresholds).unwrap();
    assert_eq!(t.matches("\"value\":0.97}").count(), 1);
    std::fs::write(&s.thresholds, t.replace("\"value\":0.97}", "\"value\":0.98}")).unwrap();
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 2, "{out}{err}");
    assert!(err.contains("--thresholds") && err.contains("differs from the frozen"), "{err}");

    // An unfrozen subwindows file.
    refused("verdict-unfrozen", &Fx { freeze_subwindows: false, ..Fx::r() }, "--subwindows");

    // A hand-built ledger whose thresholds freeze line follows the seal, chain rebuilt.
    let s = setup("verdict-late-freeze", &Fx::r());
    let mut lines = bodies(&s.ledger);
    let freeze = lines.iter().position(|b| b.contains("\"event\":\"freeze\"") && b.contains("\"role\":\"thresholds\"")).unwrap();
    let moved = lines.remove(freeze);
    let seal = lines.iter().position(|b| b.contains("\"event\":\"seal\"")).unwrap();
    lines.insert(seal + 1, moved);
    std::fs::remove_file(&s.ledger).unwrap();
    for b in &lines {
        append(&s.ledger, b);
    }
    ok(&rcsr(&["protocol", "verify", "--ledger", &s.ledger]));
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 2, "{out}{err}");
    assert!(err.contains("--thresholds") && err.contains("after the final seal"), "{err}");

    // A lock calendar whose SHA-256 is not the one the after-read dating recorded.
    let s = setup("verdict-calendar", &Fx::r());
    std::fs::write(&s.regimes, format!("{REGIMES}# edited\n")).unwrap();
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 2, "{out}{err}");
    assert!(err.contains("--regimes") && err.contains("regime_dating"), "{err}");

    // A frozen development table that is not passed.
    let s = setup("verdict-dev-missing", &Fx::r());
    let s = Setup { development: None, ..s };
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 2, "{out}{err}");
    assert!(err.contains("--development"), "{err}");
}

#[test]
fn verdict_reads_only_locked_reads() {
    // A development walk-forward report.
    let mut fx = Fx::r();
    fx.reads[0].test_read = false;
    refused("verdict-dev-report", &fx, "protocol.test_read");

    // A read of a candidate outside the sealed list (its read line was written by hand).
    let mut fx = Fx::r();
    fx.reads.push(ReadFx { sealed: false, ..ReadFx::r("z", 2) });
    fx.picks.push(PickFx::r(2));
    refused("verdict-unsealed", &fx, "sealed list");

    // A read whose report hash has no test_read in the ledger.
    let mut fx = Fx::r();
    fx.reads.push(ReadFx { recorded: false, ..ReadFx::r("y", 2) });
    fx.picks.push(PickFx::r(2));
    refused("verdict-unrecorded", &fx, "records no test_read");

    // A read of another procedure: its run matches no pick, or it names no run.
    let mut fx = Fx::r();
    fx.reads[0].run = Some(run_sha(9));
    refused("verdict-unbound", &fx, "matches no pick");
    let mut fx = Fx::r();
    fx.reads[0].run = None;
    refused("verdict-no-run", &fx, "selection.run_sha256");

    // Two reads bound to one pick.
    let mut fx = Fx::r();
    fx.reads.push(ReadFx::r("b", 1));
    refused("verdict-two-reads", &fx, "two reads");

    // A read at another cost or annualisation than the frozen thresholds.
    let mut fx = Fx::r();
    fx.reads[0].bps = 20.0;
    refused("verdict-cost", &fx, "cost");
    let mut fx = Fx::r();
    fx.reads[0].ppy = 252.0;
    refused("verdict-ppy", &fx, "periods_per_year");

    // A read without the series the gates need.
    let mut fx = Fx::r();
    fx.reads[0].cost = None;
    refused("verdict-no-cost", &fx, "pooled.period_cost");

    // A primary without a read.
    let mut fx = Fx::r();
    fx.reads = vec![ReadFx::r("b", 2)];
    fx.picks = vec![PickFx::r(1), PickFx::r(2)];
    refused("verdict-no-primary", &fx, "pick 1");

    // A read edited after it was recorded.
    let s = setup("verdict-edited", &Fx::r());
    let t = std::fs::read_to_string(&s.reads[0]).unwrap();
    std::fs::write(&s.reads[0], t.replacen("\"period_net\":[0.02,", "\"period_net\":[0.05,", 1)).unwrap();
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 2, "{out}{err}");
    assert!(err.contains("does not hash"), "{err}");
}

#[test]
fn verdict_uses_frozen_thresholds_as_is() {
    let mut fx = Fx::r();
    fx.picks[0].s4 = 0.123456789012345678;
    fx.picks[0].p1 = 0.0987654321098765;
    fx.picks[0].cap = 0.2345678901234567;
    fx.population = vec!["path_1.json".to_string(), "path_2.json".to_string()];
    let s = setup("verdict-as-is", &fx);
    for f in &fx.population {
        std::fs::remove_file(s.path(f)).unwrap();
    }
    let (code, out, err) = verdict(&s);
    assert_eq!(code, 0, "{out}{err}");
    let table = Json::parse(std::fs::read_to_string(&s.thresholds).unwrap().trim()).unwrap();
    let frozen = at(&items(at(&table, &["picks"]))[0], &["freeze"]);
    let j = result(&s);
    let c = candidate(&j, 0);
    for (gate, key, field) in [("S4", "floor", "s4_floor"), ("P1", "floor", "p1_floor"), ("S3", "cap", "drawdown_cap")] {
        assert_eq!(num(c, &["gates", gate, key]).to_bits(), num(frozen, &[field, "value"]).to_bits(), "{gate}");
    }
}

/// A plain run.json of a search with --objective: finalist `x`, threshold 0, 10 bp per side.
fn run_json(t0: i64) -> String {
    let spec = "{\"column\":\"y\",\"name\":\"y\",\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":null,\"return_units\":\"simple\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"return\",\"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}";
    format!(
        "{{\"tool\":\"rcsr\",\"target\":\"y\",\"variables\":[\"x\"],\"split\":{{\"mode\":\"temporal\",\"train_time\":[{t0},{}],\"required_purge_seconds\":3600,\
\"purge_seconds\":3600}},\"target_spec\":{spec},\"search\":{{\"candidates_evaluated\":3}},\
\"objective\":{{\"spec\":{{\"threshold\":0.0,\"per_side_bps\":10.0,\"max_position\":1.0,\"cash_return\":0.0,\"min_holding\":1}},\"cost\":{{\"source\":\"flag\"}},\
\"finalist\":{{\"expression\":\"x\",\"expression_exact\":\"x\"}}}},\
\"pareto\":[{{\"archive_rank\":1,\"expression\":\"x\",\"expression_exact\":\"x\",\"complexity\":1}}]}}\n",
        t0 + 100 * 3600
    )
}

/// `protocol.report_sha256` is the SHA-256 of the report without its unhashed members, for a
/// plain, a `--from-run` and a `--front` report; a `--from-run` report names its run file's
/// SHA-256 and the candidate id does not depend on it.
#[test]
fn hashed_report_text_reproduces_the_report_hash() {
    let scratch = Scratch::new("verdict-hash");
    let dir = scratch.path();
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let t0: i64 = 1_704_067_200;
    let mut csv = String::from("ts,x,y\n");
    for i in 0..400i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        csv.push_str(&format!("{},{},{}\n", t0 + i * 3600, x, y));
    }
    let data = p("d.csv");
    std::fs::write(&data, csv).unwrap();
    rcsr::lockbox::write_sidecar(&data, &rcsr::lockbox::Lockbox { start: t0 + 300 * 3600, end: t0 + 400 * 3600 }, "test").unwrap();
    let run = p("run.json");
    std::fs::write(&run, run_json(t0)).unwrap();
    let respelled = p("run2.json");
    std::fs::write(&respelled, format!("{} \n", run_json(t0).trim_end())).unwrap();
    let w = format!("dev,{},{},{},{}", t0, t0 + 150 * 3600, t0 + 160 * 3600, t0 + 250 * 3600);
    let base = ["walkforward", "--data", data.as_str(), "--target", "y", "--time", "ts", "--window", w.as_str(), "--emit-series", "--bootstrap", "100"];
    let report = |name: &str, extra: &[&str]| -> (String, Json) {
        let out = p(name);
        let mut a = base.to_vec();
        a.extend_from_slice(extra);
        a.extend_from_slice(&["--json", "--out", out.as_str()]);
        ok(&rcsr(&a));
        let t = std::fs::read_to_string(&out).unwrap();
        let j = Json::parse(t.trim()).unwrap();
        let hashed = rcsr::verdict::hashed_report_text(&t).unwrap();
        assert_eq!(sha(hashed.as_bytes()), text(&j, &["protocol", "report_sha256"]), "{name}");
        (hashed, j)
    };
    let (plain, j) = report("plain.json", &["--expr", "x", "--threshold-fixed", "0", "--cost-bps", "10"]);
    assert!(at(&j, &["selection"]).get("run_sha256").is_none() && !plain.contains("\"ledger\":"));
    let ledger = p("ledger.jsonl");
    let (hashed, j) = report("from_run.json", &["--from-run", run.as_str(), "--ledger", ledger.as_str()]);
    assert_eq!(text(&j, &["selection", "run_sha256"]), sha(&std::fs::read(&run).unwrap()));
    assert_eq!(text(&j, &["selection", "run"]), run);
    assert!(!hashed.contains(&run), "the run path is outside the hash");
    let (_, j2) = report("from_run2.json", &["--from-run", respelled.as_str(), "--ledger", ledger.as_str()]);
    assert_ne!(text(&j2, &["selection", "run_sha256"]), text(&j, &["selection", "run_sha256"]));
    assert_eq!(text(&j2, &["protocol", "candidate_id"]), text(&j, &["protocol", "candidate_id"]), "the run file hash is not part of the candidate id");
    let (hashed, j) = report("front.json", &["--front", run.as_str(), "--validation-frac", "0.2"]);
    assert_eq!(text(&j, &["selection", "run"]), run);
    assert!(!hashed.contains(&run));
    let removed = rcsr::verdict::remove_member(&hashed, &["selection", "run_sha256"]).unwrap();
    assert!(!removed.contains("run_sha256") && Json::parse(&removed).is_some());
    assert_eq!(rcsr::verdict::remove_member(&removed, &["selection", "run_sha256"]).unwrap(), removed, "an absent member leaves the text");
}
