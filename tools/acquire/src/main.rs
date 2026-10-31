use rcsr_acquire::calendar::{self, Month};
use rcsr_acquire::transport::Curl;
use rcsr_acquire::{fetch, json, listing, universe, verify, DATA_URL, KLINES_PREFIX, LIST_URL};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

const HELP: &str = "\
rcsr-acquire: Binance Vision spot kline archive, with curl as the transport

USAGE:
  rcsr-acquire list   [--out <dir>] [--interval <iv>] [--symbols <A,B,..>] [--jobs <n>]
  rcsr-acquire fetch  --interval <iv> --from <YYYY-MM> (--symbols <A,B,..> | --symbols-file <file>)
                      [--to <YYYY-MM>] [--out <dir>] [--jobs <n>] [--skip-404]
  rcsr-acquire verify --interval <iv> [--dir <dir>] [--symbols <A,B,..>] [--from <YYYY-MM>]
                      [--to <YYYY-MM>] [--snapshot <file>] [--through <YYYY-MM-DD>] [--jobs <n>]
  rcsr-acquire universe --archive <file> [--products <prod.json>] [--equities <f1,f2,..>] [--config <file>]
                      [--tickers <spot.json> --top-by-quote-volume <n>] [--out <dir>]

list    Lists the archive bucket. Without --symbols it writes <out>/all_symbols_archive.txt.
        With --interval it also writes <out>/keys_<iv>.json, the months published per symbol.
fetch   Downloads <out>/raw_<iv>/<SYM>/<SYM>-<iv>-YYYY-MM.zip for every month in the range.
        A zip larger than 1000 bytes is kept and not requested again. Every request appends
        a row to <out>/downloads_<iv>.csv (symbol,interval,month,http_status,bytes,fetched_at).
        --to defaults to the previous month; a month that is not published yet is refused.
        --skip-404 does not request months an earlier run recorded as 404.
verify  Fetches the .CHECKSUM sibling of every zip under <dir>/raw_<iv>/, compares it with the
        local SHA-256 and writes the snapshot JSON for `rcsr panel build --snapshot`
        (default <dir>/snapshot_<iv>.json). Exit 1 when a zip is
        not verified. `complete` is true only when all zips verify and coverage_last is at
        least --through.

universe Filters the USDT symbols of an archive list to genuine crypto pairs. Every input is a file;
        nothing is downloaded. Writes <out>/universe_all_usdt_pit.txt (sorted) and
        <out>/symbol_exclusions.csv (symbol,rule,detail, one row per matched rule). --products is the
        exchange product list (JSON with a \"data\" array of {s,tags}); a symbol listed there is not
        checked by the equity name heuristic. --equities are NASDAQ-format ticker files. --config
        replaces keys of the default rules (tools/acquire/universe-defaults.toml). With --tickers
        (JSON array of {symbol,quoteVolume}) and --top-by-quote-volume N it also writes
        <out>/universe_top_N_by_quote_volume.txt, the N kept symbols by quote volume.

Defaults: --out and --dir are the current directory, --jobs is 4. Requests retry 3 times.
Exit codes: 0 success, 1 failed requests or unverified zips, 2 usage error.
";

struct Args {
    flags: BTreeMap<String, String>,
    switches: Vec<String>,
}

const VALUE_FLAGS: &[&str] = &[
    "--out", "--dir", "--interval", "--symbols", "--symbols-file", "--from", "--to", "--jobs",
    "--snapshot", "--through", "--archive", "--products", "--equities", "--config", "--tickers",
    "--top-by-quote-volume",
];
const SWITCHES: &[&str] = &["--skip-404"];

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut flags = BTreeMap::new();
    let mut switches = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if SWITCHES.contains(&a) {
            switches.push(a.to_string());
        } else if VALUE_FLAGS.contains(&a) {
            i += 1;
            let v = args.get(i).ok_or_else(|| format!("{a} needs a value"))?;
            if flags.insert(a.to_string(), v.clone()).is_some() {
                return Err(format!("{a} given twice"));
            }
        } else {
            return Err(format!("unknown argument '{a}'"));
        }
        i += 1;
    }
    Ok(Args { flags, switches })
}

impl Args {
    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }
    fn switch(&self, k: &str) -> bool {
        self.switches.iter().any(|s| s == k)
    }
    fn path(&self, k: &str) -> PathBuf {
        PathBuf::from(self.get(k).unwrap_or("."))
    }
    fn jobs(&self) -> Result<usize, String> {
        match self.get("--jobs") {
            None => Ok(4),
            Some(v) => v.parse::<usize>().ok().filter(|n| *n >= 1).ok_or(format!("invalid --jobs '{v}'")),
        }
    }
    fn month(&self, k: &str) -> Result<Option<Month>, String> {
        self.get(k).map(Month::parse).transpose()
    }
    fn interval(&self) -> Result<String, String> {
        let iv = self.get("--interval").ok_or("--interval is required")?;
        valid_name(iv, "--interval")
    }
    fn symbols(&self) -> Result<Option<Vec<String>>, String> {
        let list: Vec<String> = match (self.get("--symbols"), self.get("--symbols-file")) {
            (Some(_), Some(_)) => return Err("--symbols and --symbols-file are exclusive".into()),
            (Some(s), None) => s.split(',').map(|x| x.trim().to_string()).collect(),
            (None, Some(f)) => fs::read_to_string(f)
                .map_err(|e| format!("cannot read {f}: {e}"))?
                .lines()
                .map(|x| x.trim().to_string())
                .collect(),
            (None, None) => return Ok(None),
        };
        let mut out = Vec::new();
        for s in list.into_iter().filter(|s| !s.is_empty()) {
            out.push(valid_name(&s, "symbol")?);
        }
        Ok(Some(out))
    }
}

fn valid_name(s: &str, what: &str) -> Result<String, String> {
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric()) {
        Ok(s.to_string())
    } else {
        Err(format!("invalid {what} '{s}'"))
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first().map(String::as_str) else {
        eprint!("{HELP}");
        return ExitCode::from(2);
    };
    if matches!(cmd, "--help" | "-h" | "help") {
        print!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let result = parse_args(&argv[1..]).and_then(|args| match cmd {
        "list" => cmd_list(&args),
        "fetch" => cmd_fetch(&args),
        "verify" => cmd_verify(&args),
        "universe" => cmd_universe(&args),
        other => Err(format!("unknown subcommand '{other}'")),
    });
    match result {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("rcsr-acquire: {e}\nrun `rcsr-acquire --help` for usage");
            ExitCode::from(2)
        }
    }
}

fn cmd_list(args: &Args) -> Result<u8, String> {
    let out = args.path("--out");
    let jobs = args.jobs()?;
    let interval = args.get("--interval").map(|i| valid_name(i, "--interval")).transpose()?;
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let curl = Curl::default();
    let scratch = out.join(".list-tmp");
    fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let result = (|| {
        let symbols = match args.symbols()? {
            Some(s) => s,
            None => {
                let (_, prefixes) = listing::list_all(&curl, LIST_URL, KLINES_PREFIX, true, &scratch.join("symbols.xml"), 3, 1500)?;
                let syms = listing::symbols(&prefixes, KLINES_PREFIX);
                let text: String = syms.iter().map(|s| format!("{s}\n")).collect();
                fs::write(out.join("all_symbols_archive.txt"), text).map_err(|e| e.to_string())?;
                println!("{} symbols -> {}", syms.len(), out.join("all_symbols_archive.txt").display());
                syms
            }
        };
        if let Some(iv) = &interval {
            let index = index_months(&curl, &symbols, iv, jobs, &scratch)?;
            let mut s = String::from("{");
            for (i, (sym, months)) in index.iter().enumerate() {
                s.push_str(if i == 0 { "\n  " } else { ",\n  " });
                let list: Vec<String> = months.iter().map(|m| json::quote(&m.to_string())).collect();
                s.push_str(&format!("{}: [{}]", json::quote(sym), list.join(", ")));
            }
            s.push_str("\n}\n");
            let path = out.join(format!("keys_{iv}.json"));
            fs::write(&path, s).map_err(|e| e.to_string())?;
            let n: usize = index.values().map(Vec::len).sum();
            println!("{} symbols, {n} monthly zips -> {}", index.len(), path.display());
        }
        Ok::<u8, String>(0)
    })();
    let _ = fs::remove_dir(&scratch);
    result
}

fn index_months(
    curl: &Curl,
    symbols: &[String],
    interval: &str,
    jobs: usize,
    scratch: &Path,
) -> Result<BTreeMap<String, Vec<Month>>, String> {
    let next = AtomicUsize::new(0);
    let done = Mutex::new((BTreeMap::new(), None::<String>));
    thread::scope(|scope| {
        for w in 0..jobs.clamp(1, symbols.len().max(1)) {
            let (next, done) = (&next, &done);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(sym) = symbols.get(i) else { break };
                let prefix = format!("{KLINES_PREFIX}{sym}/{interval}/");
                let res = listing::list_all(curl, LIST_URL, &prefix, false, &scratch.join(format!("page{w}.xml")), 3, 1500);
                let mut d = done.lock().expect("index lock");
                match res {
                    Ok((keys, _)) => {
                        d.0.insert(sym.clone(), listing::months_from_keys(&keys, interval));
                    }
                    Err(e) => {
                        d.1.get_or_insert(format!("{sym}: {e}"));
                        break;
                    }
                }
            });
        }
    });
    let (map, err) = done.into_inner().expect("index lock");
    err.map_or(Ok(map), Err)
}

fn cmd_fetch(args: &Args) -> Result<u8, String> {
    let interval = args.interval()?;
    let symbols = args.symbols()?.ok_or("--symbols or --symbols-file is required")?;
    if symbols.is_empty() {
        return Err("no symbols given".into());
    }
    let from = args.month("--from")?.ok_or("--from is required")?;
    let now = calendar::current_month();
    let to = match args.month("--to")? {
        Some(t) => t,
        None => {
            let (y, m) = if now.month == 1 { (now.year - 1, 12) } else { (now.year, now.month - 1) };
            Month { year: y, month: m }
        }
    };
    calendar::check_published(to, now)?;
    if from > to {
        return Err(format!("--from {from} is after --to {to}"));
    }
    let cfg = fetch::FetchConfig {
        base_url: DATA_URL.to_string(),
        out: args.path("--out"),
        interval,
        symbols,
        from,
        to,
        jobs: args.jobs()?,
        skip_404: args.switch("--skip-404"),
        retries: 3,
        backoff_ms: 1500,
    };
    let s = fetch::run(&cfg, &Curl::default()).map_err(|e| format!("fetch: {e}"))?;
    println!(
        "planned {} present {} known-404 {} requests {} ok {} http-404 {} failed {} -> {}",
        s.planned,
        s.present,
        s.known_missing,
        s.requests,
        s.ok,
        s.missing,
        s.failed,
        fetch::csv_path(&cfg.out, &cfg.interval).display()
    );
    Ok(u8::from(s.failed > 0))
}

fn cmd_verify(args: &Args) -> Result<u8, String> {
    let interval = args.interval()?;
    let dir = args.path("--dir");
    let through = args.get("--through").map(str::to_string);
    if let Some(t) = &through {
        let ok = t.len() == 10 && Month::parse(&t[..7]).is_ok() && t.as_bytes()[7] == b'-' && t[8..].bytes().all(|b| b.is_ascii_digit());
        if !ok {
            return Err(format!("invalid --through '{t}', expected YYYY-MM-DD"));
        }
    }
    let cfg = verify::VerifyConfig {
        base_url: DATA_URL.to_string(),
        dir: dir.clone(),
        interval: interval.clone(),
        symbols: args.symbols()?,
        from: args.month("--from")?,
        to: args.month("--to")?,
        jobs: args.jobs()?,
        retries: 3,
        backoff_ms: 1500,
    };
    let entries = verify::run(&cfg, &Curl::default()).map_err(|e| format!("verify: {e}"))?;
    let json = verify::snapshot_json(&entries, &interval, through.as_deref(), &calendar::now_iso());
    let path = args.get("--snapshot").map(PathBuf::from).unwrap_or_else(|| dir.join(format!("snapshot_{interval}.json")));
    fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    let verified = entries.iter().filter(|e| e.verified).count();
    println!("verified {verified}/{} -> {}", entries.len(), path.display());
    Ok(u8::from(entries.is_empty() || verified != entries.len()))
}

fn cmd_universe(args: &Args) -> Result<u8, String> {
    let top = args
        .get("--top-by-quote-volume")
        .map(|v| v.parse::<usize>().ok().filter(|n| *n >= 1).ok_or(format!("invalid --top-by-quote-volume '{v}'")))
        .transpose()?;
    let cfg = universe::UniverseConfig {
        archive: PathBuf::from(args.get("--archive").ok_or("--archive is required")?),
        products: args.get("--products").map(PathBuf::from),
        equities: args.get("--equities").map(|v| v.split(',').filter(|p| !p.is_empty()).map(PathBuf::from).collect()).unwrap_or_default(),
        config: args.get("--config").map(PathBuf::from),
        tickers: args.get("--tickers").map(PathBuf::from),
        top,
        out: args.path("--out"),
    };
    let s = universe::run(&cfg)?;
    let rules: Vec<String> = s.by_rule.iter().map(|(r, n)| format!("{r} {n}")).collect();
    println!(
        "archive usdt {} kept {} excluded rows {} ({}) -> {}",
        s.usdt,
        s.kept,
        s.by_rule.values().sum::<usize>(),
        rules.join(", "),
        cfg.out.display()
    );
    if let Some(f) = &s.top_file {
        println!("top {} -> {}", s.top_count, f.display());
    }
    Ok(0)
}
