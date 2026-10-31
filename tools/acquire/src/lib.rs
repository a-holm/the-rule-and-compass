//! Binance Vision archive acquisition with `curl` as the only transport.
//!
//! The crate has no dependencies. Network access is behind the [`transport::Transport`]
//! trait, so every code path except the `curl` call is testable offline.

pub mod calendar;
pub mod fetch;
pub mod json;
pub mod listing;
pub mod sha256;
pub mod transport;
pub mod verify;

/// Public download host of the archive.
pub const DATA_URL: &str = "https://data.binance.vision";
/// S3 listing endpoint of the same bucket.
pub const LIST_URL: &str = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision";
/// Key prefix of the spot monthly kline archive.
pub const KLINES_PREFIX: &str = "data/spot/monthly/klines/";
