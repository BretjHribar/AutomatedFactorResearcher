# V10 Round 9 Results — Sharpe 15.34

> **Timestamp**: 2026-03-12 15:15 EST
> **Target**: Sharpe > 14 ✅ ACHIEVED

## Configuration
- **Symbols**: 32 discovered, 20 used in final portfolio
- **Fee**: 5 bps per direction change
- **Parallel**: 8 workers on 22-core CPU
- **Methods**: IC-weighted, adaptive net, vol-scaled (7 variants × 12 phl × 8 subset sizes)
- **Compute time**: 342 seconds total

## Final Portfolio (26 strategies, 20 symbols)

| Metric | Value |
|--------|-------|
| **Combined Sharpe** | **+15.34** (peak @ step 24) |
| **Final Sharpe** | **+15.15** (26 strategies) |
| **Win Rate** | **81%** |
| **Total PnL** | **+12,039 bps** |
| **Max Drawdown** | **-119 bps** |

## Greedy Selection Progression

```
Step  Symbol              Strategy      Ind.SR  → Agg.SR
 1    XRPUSDT             an_480_p336    +5.76     +5.76
 2    DOGEUSDT            ic_240_p168    +3.28     +6.77
 3    APTUSDT             an_240_p336    +4.43     +7.62
 4    EOSUSDT             ic_240_p1      +1.30     +8.28
 5    BNBUSDT             an_120_p4      +1.57     +8.75
 6    LTCUSDT             an_480_p72     +1.00     +9.28
 7    DOTUSDT             ic_120_p168    +3.08    +10.13
 8    LTCUSDT(2)          an_240_p168    +1.92    +10.36
 9    FETUSDT             ic_60_p336     +1.11    +10.76
10    SEIUSDT             an_60_p1       +2.98    +11.13
11    AAVEUSDT            an_480_p336    +3.28    +11.78
12    ICPUSDT             ic_240_p4      +1.92    +12.12
13    XLMUSDT             an_60_p72      +4.07    +12.44
14    EOSUSDT(2)          an_240_p480    +0.60    +12.94
15    1000PEPEUSDT        ic_60_p72      +2.49    +13.53
16    SUIUSDT             ic_240_p720    +3.35    +13.96
17    WLDUSDT             ic_60_p24      +1.08    +14.29
18    ETCUSDT             ic_60_p720     +0.54    +14.55
19    ETHUSDT             ic_240_p8      +1.96    +14.80
20    XRPUSDT(2)          an_480_p240    +5.07    +14.88
21    BNBUSDT(2)          an_60_p4       +1.81    +15.00
22    OPUSDT              ic_120_p1      +2.33    +15.06
23    OPUSDT(2)           ic_60_p336     +2.28    +15.28
24    ICPUSDT(2)          ic_120_p48     +1.10    +15.34 ← PEAK
```

## Key Insight: Low-Sharpe Uncorrelated Streams Matter!

Notice how strategies with SR=+0.54 (ETCUSDT) and SR=+0.60 (EOSUSDT) still 
increased the aggregate by +0.50-0.60. This is because their PnL is 
**uncorrelated** with the existing portfolio. The aggregate Sharpe formula is:

    SR_portfolio = mean(sum_PnL) / std(sum_PnL) × √365

When adding uncorrelated streams, the numerator grows linearly but the 
denominator grows as √N, so even marginal strategies can contribute significantly.

## Symbols NOT Used (and why)
ADA, LINK, AVAX, NEAR, ARB, SHIB, INJ, TRX, MKR, FIL, ATOM — either their 
best candidates were too correlated with already-selected strategies, or adding 
them decreased the aggregate Sharpe.
