"""
Per-alpha preprocessing — config-driven replacement for the ad-hoc
`proc_signal_subind` (equity) and `signal_to_portfolio` (crypto) helpers
that lived in eval scripts.

A single `apply_preprocess(signal, **opts)` covers all variants:

    EQUITY canonical (eval_smallcap_d0_final.proc_signal):
        universe_mask=True, demean_method='subindustry',
        normalize='l1', clip_max_w=0.02

    CRYPTO canonical (update_wq_alphas_db.signal_to_portfolio):
        universe_mask=False, demean_method='cross_section',
        normalize='l1', clip_max_w=None

A test suite in tools/test_preprocessing_byteeq.py asserts both produce
DataFrames identical to the legacy helpers on real data.
"""
from __future__ import annotations
from typing import Optional, Literal

import numpy as np
import pandas as pd

DemeanMethod = Literal["none", "cross_section", "subindustry"]
NormalizeMethod = Literal["none", "l1", "zscore"]


def apply_preprocess(
    signal: pd.DataFrame,
    *,
    universe_mask: bool = False,
    universe: Optional[pd.DataFrame] = None,
    demean_method: DemeanMethod = "cross_section",
    classifications: Optional[pd.Series] = None,
    normalize: NormalizeMethod = "l1",
    clip_max_w: Optional[float] = None,
) -> pd.DataFrame:
    """Standardize a raw alpha signal into a portfolio-shaped weight matrix.

    Stages (run in this order, any can be skipped via opts):
      1. universe_mask     : set values outside `universe` mask to NaN per bar
      2. demean            : "cross_section" (per-row) | "subindustry" (per-row,
                              per-group) | "none"
      3. normalize         : "l1" (rows sum |w|=1) | "zscore" (per-row z-score)
                              | "none"
      4. clip_max_w        : clip to ±max_w  (None = no clip)
      5. fillna(0)

    Args:
        signal: DataFrame (date × ticker) of raw signal values.
        universe_mask: if True, mask signal where `universe` is False per bar.
                       Requires `universe` to be passed.
        universe: bool DataFrame (date × ticker), used only if universe_mask.
        demean_method:
            "none"          - no demean
            "cross_section" - subtract row mean
            "subindustry"   - subtract per-subindustry mean (requires
                              `classifications`).
        classifications: Series mapping ticker -> subindustry group, used
                         only if demean_method == "subindustry".
        normalize:
            "none"   - no normalization
            "l1"     - divide each row by sum |x| (so |w|.sum() == 1)
            "zscore" - subtract row mean, divide by row std
        clip_max_w: optional clip range. None = no clip.

    Returns:
        DataFrame with same shape as input, dtype float, no NaN
        (final fillna(0)).
    """
    s = signal.replace([np.inf, -np.inf], np.nan).astype(float)

    # 1. universe mask (optional)
    if universe_mask:
        if universe is None:
            raise ValueError("universe_mask=True requires `universe` DataFrame")
        s = s.where(universe, np.nan)

    # 2. demean
    if demean_method == "cross_section":
        s = s.sub(s.mean(axis=1), axis=0)
    elif demean_method == "subindustry":
        if classifications is None:
            raise ValueError("demean_method='subindustry' requires `classifications`")
        for g in classifications.dropna().unique():
            cols = (classifications == g).values
            if cols.any():
                sub = s.iloc[:, cols]
                s.iloc[:, cols] = sub.sub(sub.mean(axis=1), axis=0)
    elif demean_method == "none":
        pass
    else:
        raise ValueError(f"unknown demean_method={demean_method!r}")

    # 3. normalize
    if normalize == "l1":
        ab = s.abs().sum(axis=1).replace(0, np.nan)
        s = s.div(ab, axis=0)
    elif normalize == "zscore":
        sd = s.std(axis=1).replace(0, np.nan)
        s = s.div(sd, axis=0)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"unknown normalize={normalize!r}")

    # 4. clip
    if clip_max_w is not None:
        s = s.clip(lower=-float(clip_max_w), upper=float(clip_max_w))

    # 5. fillna
    return s.fillna(0)
