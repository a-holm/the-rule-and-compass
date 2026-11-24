//! The h1-mini run fixture (#1444): a seeded synthetic hourly panel of 40,000 rows in 8 groups.
#![allow(dead_code)]

#[path = "synth_panel.rs"]
mod synth_panel;

pub const H1_MINI_SYMBOLS: usize = 8;
pub const H1_MINI_BARS: usize = 5_000;

/// CSV `ts,symbol,open,high,low,close,volume` with `H1_MINI_SYMBOLS * H1_MINI_BARS` rows. The same
/// seed gives the same bytes.
pub fn h1_mini_panel_csv(seed: u64) -> Vec<u8> {
    synth_panel::synth_panel_csv(H1_MINI_SYMBOLS, H1_MINI_BARS, seed)
}
