# Time parsing

The time column given by `--time` decides the order, the temporal split and the
lag series. It is parsed strictly; nothing is guessed silently.

## Accepted forms

| input | meaning |
|---|---|
| `1704187800`, `1704187800.0` | numeric epoch, unit detected (see below) |
| `2024-01-02T09:30:00Z` | 09:30 UTC |
| `2024-01-02T09:30:00-05:00` | 14:30 UTC |
| `2024-01-02 09:30:00+06:00` | 03:30 UTC |
| `2024-01-02T09:30:00` | 09:30 UTC; the assumption is **reported** |
| `2024-01-02` | date-only; policy required (see below) |
| `2024/01/02` | same as `2024-01-02`, and reported as a slash format |

Calendar dates are validated (month lengths and leap years). Seconds may carry
a fraction (`09:30:00.250`); leap second `60` is accepted.

## Offsets and time zones

An explicit `Z` or `±HH:MM`/`±HHMM` offset is applied: the stored instant is
always UTC. Without an offset, `--tz` supplies a **fixed** offset (`UTC`,
`+02:00`, `-0530`). Named zones such as `America/New_York` are refused with a
suggestion to pass a fixed offset, because the crate has no tz database and will
not guess DST. The repeated hour on a DST fall-back day is therefore explicit:
`01:30-04:00` and `01:30-05:00` are two different instants, one hour apart.

A column that mixes offset-bearing and offset-less timestamps is refused unless
`--tz` says what the offset-less values mean. A column that mixes syntactic
formats (`2024-1-2` with `2024-01-02`, date-only with date-time, `-` with `/`)
is refused with the data row number: mixing breaks naive sorting and hides
timezone changes.

## Date-only policy

`--date-only` chooses the instant a bare date denotes:

| value | meaning |
|---|---|
| `error` (default) | refuse: a date alone is under-specified |
| `utc-midnight` | 00:00:00 UTC |
| `session-close` | 21:00 UTC (16:00 New York in EST) |

The default refuses rather than silently treating a daily equity bar as
midnight, which would place it before the session it summarises.

## Numeric units

Numeric columns are matched against four disjoint epoch windows (1973–2103
after conversion):

| unit | window | multiplier to seconds |
|---|---|---|
| `s` | `1e8 ..= 4.2e9` | 1 |
| `ms` | `1e11 ..= 4.2e12` | 1e-3 |
| `us` | `1e14 ..= 4.2e15` | 1e-6 |
| `ns` | `1e17 ..= 4.2e18` | 1e-9 |

A column outside every window (including a column that mixes seconds and
milliseconds) is refused, and so is one that fits more than one unit. Pass
`--time-unit s|ms|us|ns` to state the unit. Values below `1e8` are treated as a
plain index/seconds axis, so synthetic `0,1,2` time columns keep their old
meaning. Pinned plain-epoch-seconds fixtures are bit-identical: the conversion
factor is 1 and no rounding occurs.

Every conversion and assumption appears in the split notes and the JSON, e.g.

```text
# time: format=numeric unit=ms date-only=error offset_rows=0 scaled_rows=1200
# time: numeric time column read as ms and scaled to seconds (1200 row(s))
```

## What is refused

- a date-only column without a policy;
- a numeric column with an ambiguous or mixed unit;
- mixed formats, mixed offset conventions without `--tz`;
- an impossible date or clock (2024-02-30, 25:00).

See also [`data-preparation.md`](data-preparation.md) for the split-side
consequences, and [`data-splitting.md`](data-splitting.md) for the embargo.
