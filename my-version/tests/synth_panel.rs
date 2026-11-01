//! The synthetic panel generator is byte-deterministic (#483).

#[path = "common/synth_panel.rs"]
mod synth_panel;

#[test]
fn same_symbols_bars_seed_give_the_same_csv_bytes() {
    let a = synth_panel::synth_panel_csv(3, 200, 11);
    let b = synth_panel::synth_panel_csv(3, 200, 11);
    assert_eq!(a, b);
    assert_eq!(rcsr::manifest::sha256_hex(&a), "42ea9e77e3c00b2b75515894c5c7284ed977d80a3e9a2019e57cab074b2f29e8");
    assert_ne!(a, synth_panel::synth_panel_csv(3, 200, 12));
    assert_eq!(a.iter().filter(|&&c| c == b'\n').count(), 3 * 200 + 1);
}
