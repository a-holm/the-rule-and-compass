---
type: lessons
description: Lessons that hold for any OHLC or OHLCV dataset, each with its source.
---

# Lessons: OHLC and OHLCV data

One lesson per entry: a present-tense rule, then `Source:` with the date and the artefact kind. A lesson that holds only for one dataset or one machine belongs in the local area, not here. No entry holds a value from inside a locked range or a row of raw data.

1. Check the store start against the first raw bar before every build. A start date copied from an older runbook cut four years of history unnoticed. Source: 2026-11-14, store build review.
2. Declare the lock on the first store and extend it backwards only while no run has read the new range. Source: 2026-11-14, lock revision.
3. A grep with a wide glob can read a sealed manifest. Keep every glob inside the staged and store directories, and disclose any exposure of names, hashes or sizes in the pre-registration. Source: 2026-11-21, store build.
4. Count cross-section members per split group before choosing a floor. With the default floor a period with fewer members than the floor has no market column at all. Source: 2026-11-21, member counts.
5. Build floored cross-section columns as a second, chained enrich and use the same two calls for every later store of the family. Source: 2026-11-21, store build.
6. Zero-volume bars that repeat the previous price stay in the store. Source: 2026-11-21, venue README check.
7. Venue CSVs without trade and taker fields get no flow columns; the view drops them and says so. Source: 2026-11-20, runbook gap triage.
8. Use the engine's `--embargo-seconds auto`. A hand table undercounted the longest lookback. Source: 2026-11-14, merge review.
9. A split whose span reaches the lock is trimmed to the purge hole by the engine and exits 0. A refusal drill must probe a window that reaches the lock, not a span. Source: 2026-11-21, drills.
10. Strided labels need one anchor for all symbols, on a week boundary, or a per-timestamp book loses its meaning. Source: 2026-11-14, design.
11. Build every store of one campaign with one binary, and record its sha256 with the store. Source: 2026-11-22, roadmap triage.
12. Profile a search configuration on a synthetic panel of the same shape before the pre-registration, so no development result is seen early. Source: 2026-11-22, device profile.
13. Device steps of a store build run through the lock, after the memcheck run of the same code is clean, and end with zero new Xid lines since the step started. Source: 2026-10-31 and 2026-11-20, GPU safety.
14. A source without a volume column still gets a `volume` column of zeros in the store. Decide the variant from the source columns. Source: 2026-11-22, code read of the bar parser.
