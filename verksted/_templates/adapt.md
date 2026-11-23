# Adaptation recipe: <id>

Copy to `output/adapt.md` of the type stage and fill in. The keys are those of `../06_data/references/type-stage-contract.md`. One `key: value` per line.

type: <folder name of the type stage, for example 04_ohlcv>
variant: <column variant inside the type>
layout: <file layout that was detected>
bar_seconds: <spacing of the store rows, or none>
source_bar_seconds: <spacing of the raw rows>
first_bar: <first row of the source, ISO UTC>
store_first_bar: <first row of the store, ISO UTC>
approved_start: none
dev_end: <first instant that is not development data, ISO UTC>
locked_test: <START,END, or none with the reason>
series: <single or panel>
symbols: <number of series>
raw_columns: <raw value columns the source holds>
zero_volume_rows: <count>
zero_volume_policy: <keep or drop, with the reason>
feature_groups: <--features value of stage 06>
view_excludes: <names or families the variant cannot support>
labels: <--label values of stage 06>
periods_per_year: <31536000 / max(horizon_seconds, stride_seconds)>
store: <path of the base store>
deviations: <everything skipped with the reason, or none>
