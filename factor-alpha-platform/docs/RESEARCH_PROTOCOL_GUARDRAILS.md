**Research Protocol Guardrails**

Alpha discovery must not select, rank, tune, or iterate candidates using
validation or test performance. Parameter sweeps over expression variants are
not evidence; they are overfit diagnostics at best.

Required flow:

1. Define candidate family, universe, costs, train window, validation window,
   and test window before running.
2. Run discovery on TRAIN only. Candidate scan outputs must not contain VAL,
   TEST, VAL+TEST, or FULL metrics unless explicitly marked diagnostic.
3. Freeze the selected expression list and all hyperparameters from TRAIN-only
   evidence.
4. Use validation once for acceptance/rejection of the frozen set, not for
   ranking thousands of alternatives.
5. Use test once, after acceptance, as a final untouched holdout.
6. Quarantine any run that sorts, filters, or promotes based on validation or
   test metrics.

Tooling notes:

- `tools/run_kucoin_btc_terms_discovery.py` now defaults to
  `TRAIN_ONLY_DISCOVERY` and writes only TRAIN metrics.
- Non-TRAIN ranking requires `--allow-oos-diagnostics`; such output is
  diagnostic only and must not be used for selection or promotion.
- `tools/analyze_kucoin_btc_terms_selection.py` now defaults to
  `TRAIN_ONLY_SELECTION`.
- VAL-based or TEST-based selection sets require explicit diagnostic flags and
  are not promotion-eligible.
