//! Test-only helper for `scripts/tests/stall_watch.sh`: blocks forever on a channel.

use std::sync::mpsc;

fn main() {
    let _ = rcsr::attach::linked();
    let (tx, rx) = mpsc::channel::<u8>();
    let _keep = tx;
    let _ = rx.recv();
}
