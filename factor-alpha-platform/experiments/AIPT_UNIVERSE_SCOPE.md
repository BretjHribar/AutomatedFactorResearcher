# AIPT Universe Scope Notes

Generated during the strict AIPT replication/cost experiments.

## `equity_smallcap_d0` / `equity_smallcap_d1`

- Universe file: `experiments/data/aipt_universes/MCAP_100M_500M_PITV2.parquet`
- Source matrices: `data/fmp_cache/matrices_pit_v2`
- Construction: point-in-time membership at `t`, `close[t] > 0`, and `100MM <= market_cap[t] < 500MM`
- Raw PIT universe file: 4,102 dates, 6,706 columns, 3,798 unique smallcap members, max 1,394 active names/day
- Scenario experiment range: 2018-01-02 to 2026-04-24

Default frozen-cohort loader:

- To avoid future-name lookahead, the default loader freezes the tradable column set at the first fit date.
- First fit date for smallcap: 2019-01-03.
- Active names on that date: 631, so the effective experiment cohort is 631 columns, not 650.
- Full-period active names/day: mean 327, median 318, min 114, max 631.
- Post-warmup active names/day: mean 308, median 300, min 114, max 631.
- Average active names/day by year: 2019 501, 2020 431, 2021 354, 2022 305, 2023 254, 2024 203, 2025 160, 2026 YTD 127.
- ADV60 on initial fit date: median about $1.05MM, mean about $2.31MM, 75th percentile about $2.38MM, 95th about $7.54MM.
- ADV60 across active selected name-days: median about $935K, mean about $2.12MM, 75th percentile about $2.12MM, 95th about $6.86MM.
- Market cap across active selected name-days: median about $276.6MM, 5th/95th about $120.7MM/$469.3MM.

Dynamic PIT ADV variant:

- Added `--dynamic-universe` to `experiments/aipt_unconstrained.py`.
- Dynamic loader uses same-day PIT `adv60[t]` to keep up to `max_names` names per date.
- Dynamic smallcap d0 active columns ever used: 3,026.
- Dynamic active names/day: mean 593, median 643, min 299, max 650.
- Dynamic post-warmup active names/day: mean 596, median 650, min 299, max 650.
- Dynamic ADV60 across active name-days: median about $1.34MM, mean about $3.23MM, 75th percentile about $3.08MM, 95th about $10.70MM.

Interpretation:

- The default frozen cohort is strict no-lookahead, but narrower than a true rolling smallcap universe.
- Static dynamic-universe splits produced high VAL+TEST Sharpe, but strict monthly walk-forward selection collapsed to low live Sharpe. This suggests rolling-entrant selection is materially less stable than the frozen-cohort result and should not be used without stronger selection regularization.
