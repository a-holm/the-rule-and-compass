//! #868: `device_launches` counts successful kernel runs only.

#[path = "common/launches.rs"]
mod launches;
#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

use launches::{device_failures, device_launches};

fn report(batches: u64, failed: u64, fit_calls: u64) -> String {
    format!("{{\"backend\":\"gpu\",\"gpu\":{{\"batches\":{batches},\"failed_batches\":{failed},\"fit\":{{\"calls\":{fit_calls}}}}}}}")
}

#[test]
fn a_failed_batch_is_not_a_launch() {
    assert_eq!(device_launches(&report(5, 0, 2)), 7);
    assert_eq!(device_launches(&report(5, 5, 0)), 0, "every batch failed: no kernel ran");
    assert_eq!(device_launches(&report(5, 2, 1)), 4);
    assert_eq!(device_failures(&report(5, 2, 1)), 2);
    assert_eq!(device_failures(&report(5, 0, 1)), 0);
}

#[test]
fn a_report_without_a_gpu_block_or_not_json_has_no_launches() {
    assert_eq!(device_launches("{\"backend\":\"cpu\",\"gpu\":null}"), 0);
    assert_eq!(device_launches("not json"), 0);
    assert_eq!(device_failures("not json"), 0);
}

/// A child routed to the CPU reports no launch, so a forced-device test asserting `> 0` fails.
#[test]
fn a_cpu_child_reports_zero_launches() {
    let scratch = Scratch::new("launches-cpu");
    let mut csv = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        csv.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    let d = scratch.path().join("d.csv");
    std::fs::write(&d, csv).unwrap();
    let out = rcsr_cmd::rcsr()
        .args(["--data", d.to_str().unwrap(), "--target", "y", "--generations", "2", "--pop", "20", "--json", "--cpu"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(device_launches(&String::from_utf8_lossy(&out.stdout)), 0);
}
