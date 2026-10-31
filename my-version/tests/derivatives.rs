use rcsr::derivatives::{self, Snapshot};
use rcsr::enrich;

const H: f64 = 3600.0;
const D: f64 = 86_400.0;

fn snap(rows: &[(&str, &str, f64, &str)]) -> Snapshot {
    let mut s = String::from("field,symbol,publish_time,value\n");
    for (f, sym, t, v) in rows {
        s.push_str(&format!("{f},{sym},{t},{v}\n"));
    }
    Snapshot::from_csv(&s).unwrap()
}

#[test]
fn every_reserved_column_is_unavailable_with_inventory_fields() {
    let reg = enrich::default_registry();
    assert_eq!(derivatives::NAMES.len(), 10);
    for name in derivatives::NAMES {
        let e = reg.iter().find(|e| e.name == *name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(e.family, "derivatives");
        assert!(!e.available);
        assert!(e.reason.as_deref().unwrap().contains("missing raw field"));
    }
    let json = enrich::registry_json();
    let inv = derivatives::inventory_json();
    for key in ["required_source", "publication_lag", "required_raw_field", "flip_condition"] {
        assert_eq!(json.matches(&format!("\"{key}\":")).count(), 10, "{key} in registry");
        assert_eq!(inv.matches(&format!("\"{key}\":")).count(), 10, "{key} in inventory");
    }
    assert_eq!(inv.matches("\"available\":false,\"source\":null").count(), 10);
    assert!(inv.contains("\"coverage_condition\":"));
    assert!(rcsr::schema::json().contains("\"required_raw_field\":\"fundingRate\""));
}

#[test]
fn default_manifest_block_is_unavailable_and_empty() {
    assert_eq!(derivatives::manifest_json(None), "{\"available\":false,\"source\":null,\"columns\":[]}");
}

#[test]
fn requests_are_refused_naming_the_raw_field() {
    let e = enrich::expand("funding_rate").unwrap_err();
    assert!(e.contains("unavailable") && e.contains("fundingRate"), "{e}");
    let e = enrich::expand("basis_bps").unwrap_err();
    assert!(e.contains("markPrice+spotClose"), "{e}");
    let e = enrich::expand("derivatives").unwrap_err();
    assert!(e.contains("unavailable") && e.contains("openInterest") == false && e.contains("sumOpenInterest"), "{e}");
    assert!(enrich::expand("r1,liq_long").unwrap_err().contains("liqLongNotional"));
    let e = enrich::expand("funding_rate_8h").unwrap_err();
    assert!(e.contains("unknown"), "{e}");
    assert!(derivatives::resolve(&["funding_rate"], None).unwrap_err().contains("fundingRate"));
    assert!(derivatives::resolve(&["funding_rate_8h"], None).unwrap_err().contains("unknown"));
}

#[test]
fn funding_join_uses_only_published_values() {
    // Funding for 08:00 is published at 08:05; the 16:00 value is published at 16:05.
    let s = snap(&[
        ("fundingRate", "AAAUSDT", 0.0 * H + 300.0, "0.0001"),
        ("fundingRate", "AAAUSDT", 8.0 * H + 300.0, "0.0002"),
        ("fundingRate", "AAAUSDT", 16.0 * H + 300.0, "0.0003"),
        ("fundingRate", "AAAUSDT", 24.0 * H + 300.0, "0.0004"),
    ]);
    let t = [0.0, 300.0, 8.0 * H, 8.0 * H + 300.0, 15.0 * H, 16.0 * H + 299.0, 16.0 * H + 300.0];
    let c = derivatives::column(&s, "funding_rate", "AAAUSDT", &t).unwrap();
    assert!(c[0].is_nan(), "before any publication");
    assert_eq!(c[1], 0.0001);
    assert_eq!(c[2], 0.0001, "the 08:05 value is not visible at 08:00");
    assert_eq!(c[3], 0.0002);
    assert_eq!(c[4], 0.0002);
    assert_eq!(c[5], 0.0002, "future-published value is not used");
    assert_eq!(c[6], 0.0003);
    // staleness beyond two cadences is NaN, never a carried value
    let c = derivatives::column(&s, "funding_rate", "AAAUSDT", &[24.0 * H + 300.0 + 17.0 * H]).unwrap();
    assert!(c[0].is_nan());
    // carry: sum of the three latest published payments
    let c = derivatives::column(&s, "funding_carry_24h", "AAAUSDT", &[16.0 * H, 16.0 * H + 300.0, 24.0 * H + 300.0]).unwrap();
    assert!(c[0].is_nan(), "only two payments published");
    assert!((c[1] - 0.0006).abs() < 1e-15);
    assert!((c[2] - 0.0009).abs() < 1e-15);
    // unknown symbol: NaN
    assert!(derivatives::column(&s, "funding_rate", "ZZZUSDT", &[1e6]).unwrap()[0].is_nan());
}

#[test]
fn open_interest_change_uses_known_snapshots_only() {
    let mut rows = Vec::new();
    for h in 0..30 {
        rows.push(("sumOpenInterest", "AAAUSDT", h as f64 * H, if h == 12 { "" } else { "100" }));
    }
    let s = snap(&rows);
    let c = derivatives::column(&s, "oi_chg_1h", "AAAUSDT", &[5.0 * H, 12.0 * H, 13.0 * H]).unwrap();
    assert_eq!(c[0], 0.0);
    assert!(c[1].is_nan() || c[1] == 0.0);
    assert!(c[2].is_nan() || c[2] == 0.0);
    let c = derivatives::column(&s, "oi_chg_24h", "AAAUSDT", &[10.0 * H]).unwrap();
    assert!(c[0].is_nan(), "no snapshot 24h earlier");
}

#[test]
fn basis_with_one_leg_missing_is_nan() {
    let s = snap(&[
        ("markPrice", "AAAUSDT", 0.0, "101"),
        ("spotClose", "AAAUSDT", 0.0, "100"),
        ("markPrice", "AAAUSDT", H, "102"),
        ("spotClose", "AAAUSDT", 2.0 * H, "100"),
        ("markPrice", "AAAUSDT", 3.0 * H, "103"),
        ("spotClose", "AAAUSDT", 3.0 * H, ""),
    ]);
    let c = derivatives::column(&s, "basis_bps", "AAAUSDT", &[0.0, H, 2.0 * H, 3.0 * H, 4.0 * H]).unwrap();
    assert!((c[0] - 100.0).abs() < 1e-9);
    assert!(c[1].is_nan(), "spot leg missing");
    assert!(c[2].is_nan(), "mark leg missing, not spliced from t-1h");
    assert!(c[3].is_nan(), "empty spot cell is a missing leg");
    assert!(c[4].is_nan());
}

#[test]
fn conflicting_duplicates_and_bad_headers_are_errors() {
    assert!(Snapshot::from_csv("a,b\n").is_err());
    let e = Snapshot::from_csv("field,symbol,publish_time,value\nfundingRate,A,10,1\nfundingRate,A,10,2\n").unwrap_err();
    assert!(e.contains("conflicting"), "{e}");
}

fn full_rows(symbols: &[&str], days: usize, skip_symbol_day: Option<(&str, usize)>) -> Vec<(String, String, f64, String)> {
    let mut v = Vec::new();
    for sym in symbols {
        for d in 0..days {
            if skip_symbol_day.map_or(false, |(s, k)| s == *sym && d >= k) {
                continue;
            }
            for h in 0..3 {
                let t = d as f64 * D + h as f64 * 8.0 * H + 300.0;
                v.push(("fundingRate".to_string(), sym.to_string(), t, "0.0001".to_string()));
            }
        }
    }
    v
}

fn build(rows: &[(String, String, f64, String)]) -> (Snapshot, Vec<u8>) {
    let mut s = String::from("field,symbol,publish_time,value\n");
    for (f, sym, t, v) in rows {
        s.push_str(&format!("{f},{sym},{t},{v}\n"));
    }
    (Snapshot::from_csv(&s).unwrap(), s.into_bytes())
}

#[test]
fn verification_flips_only_covered_columns_and_is_recorded() {
    let syms = vec!["AAAUSDT".to_string(), "BBBUSDT".to_string()];
    let (s, bytes) = build(&full_rows(&["AAAUSDT", "BBBUSDT"], 30, None));
    let v = derivatives::verify(&s, "snap.csv", &bytes, &syms, 0.0, 30.0 * D - 1.0).unwrap();
    assert_eq!(v.columns, vec!["funding_rate", "funding_carry_24h"]);
    assert!(derivatives::resolve(&["funding_rate"], Some(&v)).is_ok());
    assert!(derivatives::resolve(&["basis_bps"], Some(&v)).unwrap_err().contains("markPrice"));
    let m = derivatives::manifest_json(Some(&v));
    assert!(m.starts_with("{\"available\":true,\"source\":\"snap.csv\",\"sha256\":\""), "{m}");
    assert!(m.contains("\"columns\":[\"funding_rate\",\"funding_carry_24h\"]"));
    // the default registry stays unavailable
    assert!(enrich::default_registry().iter().filter(|e| e.family == "derivatives").all(|e| !e.available));
}

#[test]
fn verification_fails_on_missing_symbol_or_low_coverage() {
    let syms = vec!["AAAUSDT".to_string(), "BBBUSDT".to_string()];
    let (s, bytes) = build(&full_rows(&["AAAUSDT"], 30, None));
    let e = derivatives::verify(&s, "snap.csv", &bytes, &syms, 0.0, 30.0 * D - 1.0).unwrap_err();
    assert!(e.contains("missing raw field fundingRate for BBBUSDT"), "{e}");
    let (s, bytes) = build(&full_rows(&["AAAUSDT", "BBBUSDT"], 30, Some(("BBBUSDT", 20))));
    let e = derivatives::verify(&s, "snap.csv", &bytes, &syms, 0.0, 30.0 * D - 1.0).unwrap_err();
    assert!(e.contains("BBBUSDT") && e.contains("% of days"), "{e}");
}
