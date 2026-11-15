//! #1216: a hand-written search `run.json` for the ladder, with or without the
//! regime gate (`objective.spec.pnl_mask`) the run priced its books under.
#![allow(dead_code)]

/// A threshold-book run on columns `f1`, `f2`. It saw the raw time keys `train_start..=train_end`.
pub fn run_json(masked: bool, train_start: f64, train_end: f64) -> String {
    let mask = if masked { ",\"pnl_mask\":{\"expr\":\"f1\",\"band\":0.5,\"dwell\":3}" } else { "" };
    format!(
        "{{\"variables\":[\"f1\",\"f2\"],\"pareto\":[],\
\"target_spec\":{{\"column\":\"y\",\"name\":\"raw_target\",\"kind\":\"forward_return\",\"deprecated\":false,\
\"return_column\":\"r\",\"return_units\":\"log\",\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
\"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\
\"clip\":null,\"units\":\"log_return\",\"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}},\
\"split\":{{\"train_time\":[{train_start},{train_end}],\"required_purge_seconds\":3600}},\
\"objective\":{{\"spec\":{{\"threshold\":0.0,\"max_position\":1.0,\"cash_return\":0.0,\"per_side_bps\":5.0,\"min_holding\":1{mask}}},\
\"finalist\":{{\"expression\":\"f1\",\"expression_exact\":\"f1\"}}}}}}"
    )
}
