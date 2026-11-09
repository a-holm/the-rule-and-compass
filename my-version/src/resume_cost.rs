//! Cost of a chunked search (#923): the seconds spent before this process,
//! kept in `<save path>.cost` next to the checkpoint so that checkpoint bytes
//! stay free of timestamps.

use crate::devlog::Json;

/// The backend that scored the generations of a search (#962), over chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Search {
    /// A process that ran no generation: the neutral element of `merge`.
    None,
    Cpu,
    Gpu,
    Mixed,
    Unknown,
}

impl Search {
    pub fn name(self) -> &'static str {
        match self {
            Search::None => "unknown",
            Search::Cpu => "cpu",
            Search::Gpu => "gpu",
            Search::Mixed => "mixed",
            Search::Unknown => "unknown",
        }
    }

    pub fn of(kind: crate::backend::Kind) -> Search {
        match kind {
            crate::backend::Kind::Cpu => Search::Cpu,
            crate::backend::Kind::Gpu => Search::Gpu,
        }
    }

    fn parse(s: &str) -> Search {
        match s {
            "cpu" => Search::Cpu,
            "gpu" => Search::Gpu,
            "mixed" => Search::Mixed,
            _ => Search::Unknown,
        }
    }

    /// An unknown part makes the whole unknown: an old chunk of unknown origin must not claim a backend.
    pub fn merge(self, other: Search) -> Search {
        match (self, other) {
            (Search::None, x) | (x, Search::None) => x,
            (Search::Unknown, _) | (_, Search::Unknown) => Search::Unknown,
            (a, b) if a == b => a,
            _ => Search::Mixed,
        }
    }
}

/// Wall and device cost of a search, summed over chunks.
#[derive(Clone, Debug, PartialEq)]
pub struct Cost {
    pub chunks: u64,
    pub search_s: f64,
    /// `None` when some chunk had no GPU ledger or its value was not saved.
    pub ledger_wall_s: Option<f64>,
    pub device_s: Option<f64>,
    pub backend: Search,
}

pub const INCOMPLETE_JSON: &str = "{\"complete\":false}";

fn opt(v: Option<f64>) -> String {
    match v {
        Some(v) if v.is_finite() => format!("{}", v),
        _ => "null".to_string(),
    }
}

fn read_opt(j: &Json, key: &str) -> Option<f64> {
    j.get(key).and_then(Json::as_f64)
}

pub fn sidecar_path(checkpoint: &str) -> String {
    format!("{}.cost", checkpoint)
}

impl Cost {
    pub fn process(search_s: f64, ledger_wall_s: Option<f64>, device_s: Option<f64>, backend: Search) -> Cost {
        Cost { chunks: 1, search_s, ledger_wall_s, device_s, backend }
    }

    /// `self` is the cost before this process, `process` the cost so far in it.
    pub fn plus(&self, process: &Cost) -> Cost {
        let both = |a: Option<f64>, b: Option<f64>| a.zip(b).map(|(a, b)| a + b);
        Cost {
            chunks: self.chunks + process.chunks,
            search_s: self.search_s + process.search_s,
            ledger_wall_s: both(self.ledger_wall_s, process.ledger_wall_s),
            device_s: both(self.device_s, process.device_s),
            backend: self.backend.merge(process.backend),
        }
    }

    /// The `search.cumulative` block of run.json.
    pub fn json(&self) -> String {
        format!(
            "{{\"complete\":true,\"chunks\":{},\"elapsed_seconds\":{},\"ledger_wall_s\":{},\"device_hours\":{},\"backend\":\"{}\"}}",
            self.chunks,
            opt(Some(self.search_s)),
            opt(self.ledger_wall_s),
            opt(self.device_s.map(|d| d / 3600.0)),
            self.backend.name()
        )
    }

    fn sidecar_json(&self, generation: usize) -> String {
        format!(
            "{{\"v\":2,\"generation\":{},\"chunks\":{},\"search_seconds\":{},\"ledger_wall_s\":{},\"device_seconds_incl_fit\":{},\"backend\":\"{}\"}}",
            generation,
            self.chunks,
            opt(Some(self.search_s)),
            opt(self.ledger_wall_s),
            opt(self.device_s),
            self.backend.name()
        )
    }
}

/// Atomically write the sidecar of a checkpoint saved at `generation`.
pub fn write(checkpoint: &str, generation: usize, cost: &Cost) -> Result<(), String> {
    crate::atomic::stage(sidecar_path(checkpoint), cost.sidecar_json(generation).as_bytes())?.publish()
}

/// The cost recorded for the checkpoint at `generation`, or `None` when the
/// sidecar is missing, unreadable or belongs to another generation.
pub fn load(checkpoint: &str, generation: usize) -> Option<Cost> {
    let text = std::fs::read_to_string(sidecar_path(checkpoint)).ok()?;
    let j = Json::parse(text.trim())?;
    if j.get("generation")?.as_usize()? != generation {
        return None;
    }
    Some(Cost {
        chunks: j.get("chunks")?.as_usize()? as u64,
        search_s: read_opt(&j, "search_seconds")?,
        ledger_wall_s: read_opt(&j, "ledger_wall_s"),
        device_s: read_opt(&j, "device_seconds_incl_fit"),
        backend: j.get("backend").and_then(Json::as_str).map_or(Search::Unknown, Search::parse),
    })
}

/// `search.cumulative` for a process that started from `base` (`None` for a
/// fresh run, which is its own total).
pub fn total(resumed: bool, base: Option<&Cost>, process: &Cost) -> Option<Cost> {
    match (resumed, base) {
        (false, _) => Some(process.clone()),
        (true, Some(b)) => Some(b.plus(process)),
        (true, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("rcsr-resume-cost-{}-{}", name, std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join("cp.txt").to_string_lossy().into_owned()
    }

    #[test]
    fn sidecar_round_trips_and_ignores_other_generations() {
        let path = dir("round");
        let c = Cost { chunks: 3, search_s: 12.5, ledger_wall_s: Some(11.0), device_s: None, backend: Search::Gpu };
        write(&path, 40, &c).unwrap();
        assert_eq!(load(&path, 40), Some(c));
        assert_eq!(load(&path, 41), None);
        assert_eq!(load(&format!("{}.none", path), 40), None);
    }

    #[test]
    fn sidecar_v2_round_trips_backend() {
        for backend in [Search::Cpu, Search::Gpu, Search::Mixed, Search::Unknown] {
            let path = dir(&format!("backend-{}", backend.name()));
            let c = Cost { chunks: 2, search_s: 1.0, ledger_wall_s: None, device_s: None, backend };
            write(&path, 7, &c).unwrap();
            assert!(std::fs::read_to_string(sidecar_path(&path)).unwrap().contains("\"v\":2"));
            assert_eq!(load(&path, 7).unwrap().backend, backend);
            assert!(c.json().contains(&format!("\"backend\":\"{}\"", backend.name())));
        }
    }

    #[test]
    fn sidecar_v1_loads_with_unknown_backend() {
        let path = dir("v1");
        std::fs::write(
            sidecar_path(&path),
            "{\"v\":1,\"generation\":5,\"chunks\":2,\"search_seconds\":3.5,\"ledger_wall_s\":null,\"device_seconds_incl_fit\":null}",
        )
        .unwrap();
        let c = load(&path, 5).expect("v1 sidecar loads");
        assert_eq!((c.chunks, c.search_s, c.backend), (2, 3.5, Search::Unknown));
    }

    #[test]
    fn merge_gpu_cpu_is_mixed_and_unknown_dominates() {
        assert_eq!(Search::Gpu.merge(Search::Cpu), Search::Mixed);
        assert_eq!(Search::Cpu.merge(Search::Gpu), Search::Mixed);
        assert_eq!(Search::Gpu.merge(Search::Gpu), Search::Gpu);
        assert_eq!(Search::Mixed.merge(Search::Gpu), Search::Mixed);
        for x in [Search::Cpu, Search::Gpu, Search::Mixed, Search::Unknown] {
            assert_eq!(Search::Unknown.merge(x), Search::Unknown);
            assert_eq!(x.merge(Search::Unknown), Search::Unknown);
        }
    }

    #[test]
    fn zero_generation_process_does_not_change_backend() {
        for base in [Search::Cpu, Search::Gpu, Search::Mixed, Search::Unknown] {
            let b = Cost { chunks: 1, search_s: 1.0, ledger_wall_s: None, device_s: None, backend: base };
            let t = total(true, Some(&b), &Cost::process(0.5, None, None, Search::None)).unwrap();
            assert_eq!(t.backend, base);
        }
    }

    #[test]
    fn totals_add_the_base_once_and_drop_unknown_parts() {
        let base = Cost { chunks: 2, search_s: 10.0, ledger_wall_s: Some(9.0), device_s: Some(7200.0), backend: Search::Gpu };
        let process = Cost::process(5.0, Some(4.0), None, Search::Gpu);
        let t = total(true, Some(&base), &process).unwrap();
        assert_eq!((t.chunks, t.search_s, t.ledger_wall_s, t.device_s), (3, 15.0, Some(13.0), None));
        assert_eq!(total(true, None, &process), None);
        assert_eq!(total(false, Some(&base), &process), Some(process));
    }
}
