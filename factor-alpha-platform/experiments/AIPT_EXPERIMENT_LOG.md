# AIPT Experiment Log

This is the human-readable ledger for the SSRN 4388526 replication work. The generated, file-by-file registry is:

- `experiments/AIPT_EXPERIMENT_REGISTRY.md`
- `experiments/AIPT_EXPERIMENT_REGISTRY.json`
- `experiments/AIPT_SELECTION_OVERFIT.md`

The registry is regenerated with:

```powershell
python experiments\aipt_experiment_registry.py
```

## Record Rules

- Every run under `experiments/results/aipt*` is recorded, including smokes, stopped runs, diagnostic runs, and superseded runs.
- The generated registry records TRAIN, VAL, TEST, VAL+TEST, and FULL Sharpe for surfaced candidates.
- The registry records both candidates selected by TRAIN Sharpe and candidates selected by VAL+TEST/FULL Sharpe, with `TRAIN - VAL+TEST` and `TRAIN - TEST` overfit gaps.
- Runs before the registry existed are marked `not captured; reconstructed from output specs` when the exact launch command was not provable.
- Future AIPT runners write `run_manifest.json` at startup with `argv`, parsed args, cwd, script, timestamp, and mode.
- Strict equity results must use `data/fmp_cache/matrices_pit_v2` and experiment-local PIT universes from `experiments/data/aipt_universes`.
- Cost/constrained runs are diagnostic until the unconstrained paper SDF baseline is complete.

## Current Primary Run

`experiments/results/aipt_unconstrained_strict_main`

Purpose: paper-matching baseline, unconstrained SDF only.

Command:

```powershell
python experiments\aipt_unconstrained.py --scenarios equity_smallcap_d0 equity_smallcap_d1 equity_top1000_d0 equity_top1000_d1 kucoin_top100 --source-sets default --p-grid 64,256,1024 --z-grid 0.00001,0.0001,0.001,0.01,0.1,1 --seeds 1,2,3 --out-dir experiments/results/aipt_unconstrained_strict_main
```

Status: running as of the latest registry generation. This is the run to use for matching the paper before costs.

## Supporting Artifacts

- Paper text extraction: `experiments/ssrn_4388526_extracted.txt`
- Theory/cost note: `experiments/aipt_execution_cost_theory.md`
- No-lookahead audit script: `experiments/aipt_no_lookahead_audit.py`
- Strict audit report: `experiments/results/aipt_no_lookahead_audit_strict.json`
- PIT universe builder: `experiments/aipt_build_pit_universes.py`
- PIT universe manifest: `experiments/data/aipt_universes/manifest.json`
- Result registry generator: `experiments/aipt_experiment_registry.py`
- Selection overfit report: `experiments/AIPT_SELECTION_OVERFIT.md`

## Recorded Result Folders

The generated registry currently records these result folders:

- `experiments/results/aipt_smoke`
- `experiments/results/aipt_smoke_eq`
- `experiments/results/aipt_smoke_p256`
- `experiments/results/aipt_smoke_p1024`
- `experiments/results/aipt_extended`
- `experiments/results/aipt_extended2`
- `experiments/results/aipt_unconstrained_smoke`
- `experiments/results/aipt_unconstrained_main`
- `experiments/results/aipt_unconstrained_pit_smoke`
- `experiments/results/aipt_unconstrained_pit_main`
- `experiments/results/aipt_unconstrained_strict_smoke`
- `experiments/results/aipt_unconstrained_strict_main`
- `experiments/results/aipt_unconstrained_projection_smoke`
- `experiments/results/aipt_unconstrained_projection_smoke_proj`
- `experiments/results/aipt_stepwise_smoke`

## Interpretation Notes

- `aipt_extended*` and `aipt_smoke*` are constrained/cost diagnostics. They explain why the first reported Sharpes were low, but they are not the paper replication target.
- `aipt_unconstrained_main` is legacy and was stopped after the PIT fundamental issue was identified.
- `aipt_unconstrained_pit_main` used PIT matrices but was stopped after the legacy equity universe survivorship channel was identified.
- `aipt_unconstrained_strict_main` uses PIT matrices and PIT universes and is the accepted unconstrained baseline.
- `aipt_stepwise_smoke` is the first post-baseline no-dollar-neutrality constraint/cost decomposition.
