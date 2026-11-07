//! `rcsr dispatch-plan`: the route of every cell of a fixed size grid under one
//! calibration, and the cells whose route differs under another (#861).
//!
//! The command reads no clock and opens no device. Both calibrations are
//! pinned: the corpus anchors or a file written by `rcsr calibrate`.

use crate::dispatch::{
    Calibration, CalibrationRequest, CalibrationSource, Dispatcher, Route, Stage, Workload,
};

/// Training or fit rows of the grid, from tiny to the pilot's largest store.
pub const ROWS: &[usize] = &[
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 128920, 262144, 556472, 1_000_000,
    1_600_000, 3_911_395,
];
/// Candidates per batch.
pub const CANDIDATES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 252, 504, 1024];
/// Batch maximum complexity of an eval stage.
pub const NODES: &[usize] = &[1, 3, 7, 14, 31];
/// Constants per candidate of a fit stage.
pub const CONSTANTS: &[usize] = &[1, 2, 4, 8, 14];
/// Fit sweeps.
pub const SWEEPS: &[usize] = &[3, 8];

/// One planned call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub stage: Stage,
    /// The stage already routed to the device, so the hysteresis holds it.
    pub held: bool,
    pub candidates: usize,
    pub rows: usize,
    /// Batch complexity (eval) or constants (fit).
    pub shape: usize,
    /// Fit sweeps; `0` for an eval.
    pub sweeps: usize,
}

impl Cell {
    fn workload(&self) -> Workload {
        match self.stage {
            Stage::Eval => Workload::eval(self.candidates, self.rows, self.shape),
            _ => Workload::fit(self.candidates, self.rows, self.shape, self.sweeps)
                .with_min_rows(32 * self.shape),
        }
    }

    /// Row evaluations of an eval cell, or fits times rows of a fit cell.
    pub fn work(&self) -> usize {
        match self.stage {
            Stage::Eval => self.candidates * self.rows * self.shape,
            _ => self.candidates * self.rows,
        }
    }

    fn label(&self) -> String {
        match self.stage {
            Stage::Eval => format!(
                "eval held={} W={} N={} m={}",
                self.held as u8, self.candidates, self.rows, self.shape
            ),
            _ => format!(
                "fit held={} W={} N={} p={} s={}",
                self.held as u8, self.candidates, self.rows, self.shape, self.sweeps
            ),
        }
    }
}

/// Every cell of the grid, eval then fit, first from the CPU and then held on
/// the device.
pub fn grid() -> Vec<Cell> {
    let mut cells = Vec::new();
    for held in [false, true] {
        for &rows in ROWS {
            for &candidates in CANDIDATES {
                for &shape in NODES {
                    cells.push(Cell { stage: Stage::Eval, held, candidates, rows, shape, sweeps: 0 });
                }
            }
        }
        for &rows in ROWS {
            for &candidates in CANDIDATES {
                for &shape in CONSTANTS {
                    for &sweeps in SWEEPS {
                        if rows >= 32 * shape {
                            cells.push(Cell { stage: Stage::Fit, held, candidates, rows, shape, sweeps });
                        }
                    }
                }
            }
        }
    }
    cells
}

/// The route of one cell and the two wall estimates behind it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Routed {
    pub route: Route,
    pub device_s: f64,
    pub cpu_s: f64,
}

impl Routed {
    fn wall(&self, route: Route) -> f64 {
        if route == Route::Device { self.device_s } else { self.cpu_s }
    }
}

/// Plan `cell` under `calibration`.
pub fn route(calibration: &Calibration, cell: &Cell) -> Routed {
    let mut dispatcher = Dispatcher::new(*calibration);
    if cell.held {
        let seed = match cell.stage {
            Stage::Eval => Workload::eval(1024, 4_000_000, 31),
            _ => Workload::fit(1024, 4_000_000, 4, 8),
        };
        dispatcher.plan(cell.stage, &seed);
    }
    let decision = dispatcher.plan(cell.stage, &cell.workload());
    let cost = decision.cost;
    Routed {
        route: decision.route,
        device_s: cost.device_s + cost.fit_s + cost.transfer_s + cost.launch_s,
        cpu_s: cost.cpu_s,
    }
}

/// A cell whose route differs between the two calibrations.
#[derive(Clone, Copy, Debug)]
pub struct Flip {
    pub cell: Cell,
    pub first: Routed,
    pub second: Routed,
    /// Wall seconds per call the first route costs over the second, both
    /// priced with the second calibration.
    pub extra_s: f64,
}

/// The grid routed under two calibrations.
pub struct Plan {
    pub rows: Vec<(Cell, Routed, Routed)>,
    pub flips: Vec<Flip>,
}

impl Plan {
    pub fn largest_flip_cost(&self) -> f64 {
        self.flips.iter().map(|f| f.extra_s.abs()).fold(0.0, f64::max)
    }

    pub fn largest_flip_work(&self) -> usize {
        self.flips.iter().map(|f| f.cell.work()).max().unwrap_or(0)
    }
}

/// Route the whole grid under `first` and `second`.
pub fn compare(first: &Calibration, second: &Calibration) -> Plan {
    let mut rows = Vec::new();
    let mut flips = Vec::new();
    for cell in grid() {
        let a = route(first, &cell);
        let b = route(second, &cell);
        if a.route != b.route {
            flips.push(Flip {
                cell,
                first: a,
                second: b,
                extra_s: b.wall(a.route) - b.wall(b.route),
            });
        }
        rows.push((cell, a, b));
    }
    Plan { rows, flips }
}

/// `rcsr dispatch-plan --help`.
pub const USAGE: &str = "rcsr dispatch-plan [--calibration <anchors|path>] [--against <anchors|path>] [--json]

Print the route (device or cpu) of eval and fit calls over a fixed grid of
sizes, 4950 cells, under one calibration, then the cells whose route differs
under another. A route that no cell flips is not decided by the difference
between the two. Reads no clock and opens no device.

  --calibration <v>   the calibration to route with: anchors (default) or the
                      path of a file written by `rcsr calibrate`
  --against <v>       a second calibration; without it the table shows one
                      source and no cell flips
  --json              one JSON object: both sources, the flips and the summary

Each line is `<stage> held=<0|1> W=<candidates> N=<rows> m=<complexity>|p=<constants> s=<sweeps>
cal=<route> <device s>/<cpu s> against=<route> <device s>/<cpu s>`. held=1 plans the stage
as one the device already holds, so the hysteresis band applies. The summary line gives the
cells, the flips, the largest flip cost in wall seconds per call (the first route priced with
the second calibration, minus the second route) and the largest flipped work (row-evals of an
eval, fits times rows of a fit). Exit 0; 2 on a usage error or an unusable file.";

fn source(arg: &str) -> Result<(String, Calibration), String> {
    let resolved = CalibrationRequest::parse(arg);
    if resolved == CalibrationRequest::Probe {
        return Err("--calibration probe reads a clock; dispatch-plan takes anchors or a file".to_string());
    }
    let src: CalibrationSource = resolved.resolve(None)?;
    let label = if src == CalibrationSource::Anchors { "anchors" } else { "file" };
    Ok((label.to_string(), Calibration::pinned(&src)))
}

fn rates_json(c: &Calibration) -> String {
    format!(
        "{{\"r_eval\":{:e},\"r_fit\":{:e},\"bw\":{:e},\"launch_s\":{:e},\"cpu_eval\":{:e},\"cpu_fit\":{:e},\"hysteresis\":{}}}",
        c.r_eval, c.r_fit, c.bw, c.launch_s, c.cpu_eval, c.cpu_fit, c.hysteresis
    )
}

/// `rcsr dispatch-plan`.
pub fn run(args: &[String]) -> ! {
    let mut first = "anchors".to_string();
    let mut second: Option<String> = None;
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", USAGE);
                std::process::exit(0);
            }
            "--json" => json = true,
            flag @ ("--calibration" | "--against") => {
                i += 1;
                let Some(v) = args.get(i) else {
                    eprintln!("error: {} needs a value", flag);
                    std::process::exit(2);
                };
                if flag == "--calibration" {
                    first = v.clone();
                } else {
                    second = Some(v.clone());
                }
            }
            other => {
                eprintln!("error: unknown dispatch-plan argument '{}'\n\n{}", other, USAGE);
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let load = |arg: &str| {
        source(arg).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(2)
        })
    };
    let (first_label, a) = load(&first);
    let (second_label, b) = match &second {
        Some(arg) => load(arg),
        None => (first_label.clone(), a),
    };
    let plan = compare(&a, &b);
    let summary = format!(
        "grid cells {}, flips {}, largest flip cost {:.3e} s, largest flipped work {}",
        plan.rows.len(),
        plan.flips.len(),
        plan.largest_flip_cost(),
        plan.largest_flip_work()
    );
    if json {
        let flips: Vec<String> = plan
            .flips
            .iter()
            .map(|f| {
                format!(
                    "{{\"cell\":\"{}\",\"work\":{},\"cal\":\"{}\",\"against\":\"{}\",\"extra_s\":{:e}}}",
                    f.cell.label(),
                    f.cell.work(),
                    f.first.route.name(),
                    f.second.route.name(),
                    f.extra_s
                )
            })
            .collect();
        println!(
            "{{\"schema\":\"rcsr-dispatch-plan/1\",\"calibration\":{{\"source\":\"{}\",\"rates\":{}}},\"against\":{{\"source\":\"{}\",\"rates\":{}}},\"cells\":{},\"flips\":[{}],\"largest_flip_cost_s\":{:e},\"largest_flip_work\":{}}}",
            first_label,
            rates_json(&a),
            second_label,
            rates_json(&b),
            plan.rows.len(),
            flips.join(","),
            plan.largest_flip_cost(),
            plan.largest_flip_work()
        );
    } else {
        println!("rates cal={} {}", first_label, rates_json(&a));
        println!("rates against={} {}", second_label, rates_json(&b));
        for (cell, x, y) in &plan.rows {
            println!(
                "{} cal={} {:.4e}/{:.4e} against={} {:.4e}/{:.4e}",
                cell.label(),
                x.route.name(),
                x.device_s,
                x.cpu_s,
                y.route.name(),
                y.device_s,
                y.cpu_s
            );
        }
        println!("{}", summary);
    }
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Calibration {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/calibration_rtx3060.json");
        source(path).expect("the fixture resolves").1
    }

    #[test]
    fn the_grid_has_4950_cells() {
        assert_eq!(grid().len(), 4950);
    }

    /// One of the six rates scaled, the others at the anchors.
    fn scaled(term: usize, by: f64) -> Calibration {
        let mut c = Calibration::default();
        match term {
            0 => c.r_eval *= by,
            1 => c.r_fit *= by,
            2 => {
                c.bw *= by;
                c.bw_h2d *= by;
                c.bw_d2h *= by;
            }
            3 => c.launch_s /= by,
            4 => c.cpu_eval *= by,
            _ => c.cpu_fit *= by,
        }
        c
    }

    const TERMS: [&str; 6] = ["r_eval", "r_fit", "bw", "launch_s", "cpu_eval", "cpu_fit"];

    /// The pilot's eval and fit calls: 1 to 1024 candidates, at least 65,536
    /// rows, every complexity and constant count.
    fn pilot_cells() -> Vec<Cell> {
        grid()
            .into_iter()
            .filter(|c| {
                !c.held && c.rows >= 65_536 && matches!(c.candidates, 1 | 4 | 64 | 504 | 1024)
                    && match c.stage {
                        Stage::Eval => true,
                        _ => matches!(c.shape, 1 | 4 | 14),
                    }
            })
            .collect()
    }

    #[test]
    fn pilot_sizes_route_to_the_device_under_a_tenfold_rate_error() {
        let cells = pilot_cells();
        assert!(cells.len() > 300, "{} pilot cells", cells.len());
        for term in 0..TERMS.len() {
            for by in [0.1, 10.0] {
                let c = scaled(term, by);
                for cell in &cells {
                    let routed = route(&c, cell);
                    assert_eq!(
                        routed.route,
                        Route::Device,
                        "{} x{} flips {} ({:.3e} device, {:.3e} cpu)",
                        TERMS[term],
                        by,
                        cell.label(),
                        routed.device_s,
                        routed.cpu_s
                    );
                }
            }
        }
    }

    #[test]
    fn anchor_flips_stay_below_the_launch_scale() {
        let plan = compare(&Calibration::default(), &fixture());
        assert!(plan.rows.len() == 4950);
        assert!(plan.flips.len() <= 60, "{} flips", plan.flips.len());
        for f in &plan.flips {
            assert!(f.cell.work() <= 4096, "{} flips with work {}", f.cell.label(), f.cell.work());
            assert!(f.extra_s.abs() <= 8.0e-6, "{} costs {:.3e} s", f.cell.label(), f.extra_s);
        }
    }

    #[test]
    fn a_calibration_against_itself_has_no_flip() {
        let plan = compare(&fixture(), &fixture());
        assert!(plan.flips.is_empty());
    }
}
