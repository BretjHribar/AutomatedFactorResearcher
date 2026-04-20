# Lookahead Bias Test: Forward vs Backward Returns

```
Loading augmented data (K=58)...
  Panel: T=11892, N=49, K=58

====================================================================================================
  LOOKAHEAD BIAS TEST: Forward vs Backward cum_ret
  K=58, P=500, z=1e-5, 5 seeds, QP(tc=0.005, rb=1, to=0.05)
====================================================================================================

  ORIGINAL: features[t-1] -> ret[t+1:t+h]  (FUTURE - not realized at t)
  CAUSAL:   features[t-h] -> ret[t-h+1:t]   (PAST  - realized at t)


  === h=1  ORIGINAL (no multi-h, no bias possible) ===
    QP train: SR@5=-0.911 TO=0.0828
    QP val  : SR@5=-0.545 TO=0.1003
    QP test : SR@5=+2.024 TO=0.1007
    Time: 272s

  === h=1  CAUSAL  (should match original) ===
    QP train: SR@5=-0.911 TO=0.0828
    QP val  : SR@5=-0.545 TO=0.1003
    QP test : SR@5=+2.024 TO=0.1007
    Time: 259s

  === h=6  ORIGINAL (forward-looking, BIASED) ===
    QP train: SR@5=+6.928 TO=0.0805
    QP val  : SR@5=+5.514 TO=0.0941
    QP test : SR@5=+5.414 TO=0.0999
    Time: 275s

  === h=6  CAUSAL  (backward-looking, CLEAN) ===
    QP train: SR@5=+0.048 TO=0.0845
    QP val  : SR@5=-0.115 TO=0.1095
    QP test : SR@5=-0.021 TO=0.1026
    Time: 267s

  === h=12 ORIGINAL (forward-looking, BIASED) ===
    QP train: SR@5=+10.064 TO=0.0792
    QP val  : SR@5=+7.708 TO=0.0911
    QP test : SR@5=+7.941 TO=0.0927
    Time: 267s

  === h=12 CAUSAL  (backward-looking, CLEAN) ===
    QP train: SR@5=-0.802 TO=0.0811
    QP val  : SR@5=-2.559 TO=0.0970
    QP test : SR@5=-0.394 TO=0.1012
    Time: 328s

  === h=24 ORIGINAL (forward-looking, BIASED) ===
    QP train: SR@5=+11.817 TO=0.0776
    QP val  : SR@5=+11.142 TO=0.0814
    QP test : SR@5=+8.583 TO=0.0773
    Time: 244s

  === h=24 CAUSAL  (backward-looking, CLEAN) ===
    QP train: SR@5=-0.929 TO=0.0844
    QP val  : SR@5=-0.559 TO=0.1069
    QP test : SR@5=-0.627 TO=0.0961
    Time: 262s

====================================================================================================
  SUMMARY: ORIGINAL (biased) vs CAUSAL (clean)
====================================================================================================
  Config                                             |   Train |     Val |    Test |  Test TO
  -----------------------------------------------------------------------------------------------
  h=1  ORIGINAL (no multi-h, no bias possible)       |   -0.91 |   -0.55 |   +2.02 |   0.1007
  h=1  CAUSAL  (should match original)               |   -0.91 |   -0.55 |   +2.02 |   0.1007
  h=6  ORIGINAL (forward-looking, BIASED)            |   +6.93 |   +5.51 |   +5.41 |   0.0999
  h=6  CAUSAL  (backward-looking, CLEAN)             |   +0.05 |   -0.11 |   -0.02 |   0.1026
  h=12 ORIGINAL (forward-looking, BIASED)            |  +10.06 |   +7.71 |   +7.94 |   0.0927
  h=12 CAUSAL  (backward-looking, CLEAN)             |   -0.80 |   -2.56 |   -0.39 |   0.1012
  h=24 ORIGINAL (forward-looking, BIASED)            |  +11.82 |  +11.14 |   +8.58 |   0.0773
  h=24 CAUSAL  (backward-looking, CLEAN)             |   -0.93 |   -0.56 |   -0.63 |   0.0961

  BIAS MAGNITUDE (ORIGINAL - CAUSAL):
    h= 1 val  : -0.55 - -0.55 = +0.00 bias
    h= 1 test : +2.02 - +2.02 = +0.00 bias
    h= 6 val  : +5.51 - -0.11 = +5.63 bias
    h= 6 test : +5.41 - -0.02 = +5.44 bias
    h=12 val  : +7.71 - -2.56 = +10.27 bias
    h=12 test : +7.94 - -0.39 = +8.34 bias
    h=24 val  : +11.14 - -0.56 = +11.70 bias
    h=24 test : +8.58 - -0.63 = +9.21 bias

  Total time: 2356s
```
