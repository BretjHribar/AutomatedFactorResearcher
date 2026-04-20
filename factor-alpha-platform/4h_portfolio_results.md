# 4H Alpha Portfolio Evaluation Results (22 Alphas)
**Timestamp**: 2026-04-14 15:30 (UTC)
**Universe**: BINANCE_TOP50 | **Interval**: 4h
**Active Alphas**: 22

## 📝 Detailed Strategy Performance (@ 7 bps Fees)

### Risk Parity + QP (Top Performer)
| Split | Sharpe | Ret% | RetAnn% | MaxDD% | TO | Fitness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | +0.723 | +72.08% | +24.85% | -27.90% | 0.0693 | 1.37 |
| **Val** | -0.182 | -2.19% | -4.53% | -14.92% | 0.0640 | -0.15 |
| **Test** | **+2.948** | **+31.41%** | **+85.11%** | **-7.30%** | **0.0541** | **11.69** |

### Billion Alphas + QP
| Split | Sharpe | Ret% | RetAnn% | MaxDD% | TO | Fitness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | +0.308 | +31.13% | +10.73% | -50.54% | 0.1176 | 0.29 |
| **Val** | -0.928 | -13.31% | -27.48% | -25.36% | 0.1425 | -1.29 |
| **Test** | **+2.612** | **+31.77%** | **+86.08%** | **-9.93%** | **0.1085** | **7.36** |

---

## 📉 Cumulative Performance at 7 bps Fees (Summary)
See full chart: `4h_portfolio_7bps.png`

| Method | Sharpe | RetAnn% | MaxDD% | TO |
| :--- | :---: | :---: | :---: | :---: |
| **Risk Parity + QP** | **+2.95** | **+85.1%** | **-7.3%** | **0.054** |
| Billion Alphas + QP | +2.61 | +86.1% | -9.9% | 0.108 |
| Equal Weight + QP | +2.22 | +64.9% | -12.6% | 0.051 |
| Adaptive + QP | +1.67 | +47.5% | -12.8% | 0.068 |
| Factor MAX (10d) + QP | +0.05 | +1.7% | -22.1% | 0.118 |
| Factor MAX (5d) + QP | -2.05 | -71.7% | -35.8% | 0.143 |

---

## 📂 Artifact Locations
### Performance Charts
- [4h_portfolio_0bps.png](file:///c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/4h_portfolio_0bps.png)
- [4h_portfolio_2bps.png](file:///c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/4h_portfolio_2bps.png)
- [4h_portfolio_5bps.png](file:///c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/4h_portfolio_5bps.png)
- [4h_portfolio_7bps.png](file:///c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/4h_portfolio_7bps.png)
- [4h_portfolio_test_sharpe_comparison.png](file:///c:/Users/breth/PycharmProjects/AutomatedFactorResearcher/factor-alpha-platform/4h_portfolio_test_sharpe_comparison.png)

### Raw Simulation Output
Alphas loaded:
- Alphas #10 to #32 active.
- New alpha #32 added significant value to "Billion Alphas" methodology.
