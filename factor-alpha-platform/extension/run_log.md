[10:53:02] # Step 1 — Enumerate delisted universe
[10:53:07]   page 10: cumulative 1000 entries
[10:53:13]   page 20: cumulative 2000 entries
[10:53:18]   page 30: cumulative 3000 entries
[10:53:24]   page 40: cumulative 4000 entries
[10:53:29]   page 50: cumulative 5000 entries
[10:53:35]   page 60: cumulative 6000 entries
[10:53:39]   page 70: cumulative 7000 entries
[10:53:45]   page 80: cumulative 8000 entries
[10:53:50]   page 90: cumulative 8942 entries
[10:53:50]   Total delisted entries pulled: 8942
[10:53:50] 
## Filter breakdown:
[10:53:50]   non_us_exchange            3,923
[10:53:50]   etf_or_fund                  809
[10:53:50]   delisted_pre_2010              8
[10:53:50]   missing_dates                  0
[10:53:50]   kept                       4,202
[10:53:50] 
## Kept: 4202 US common stocks delisted 2010+
[10:53:50] 
## Delisting count by year:
[10:53:50]   2010:     5
[10:53:50]   2011:     2
[10:53:50]   2012:     1
[10:53:50]   2013:     2
[10:53:50]   2014:     2
[10:53:50]   2015:     5
[10:53:50]   2016:   100
[10:53:50]   2017:   124
[10:53:50]   2018:   156
[10:53:50]   2019:    32
[10:53:50]   2020:    28
[10:53:50]   2021:   270
[10:53:50]   2022:   748
[10:53:50]   2023: 1,058
[10:53:50]   2024:   745
[10:53:50]   2025:   680
[10:53:50]   2026:   244
[10:53:50] 
## Exchange breakdown of kept universe:
[10:53:50]   NASDAQ       2,857
[10:53:50]   NYSE         1,244
[10:53:50]   AMEX           101
[10:53:50] 
## Saved 4202 entries to C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\delisted_universe.json
[10:54:49] # Step 6 — Restate-detection probe
[10:54:49]   Comparing fresh fetch (today) vs cached values for 20 tickers
[10:54:49]   Average cached-file age: 58.5 days
[10:54:49] 
## AAPL
[10:54:51] 
## MSFT
[10:54:53] 
## JPM
[10:54:54] 
## XOM
[10:54:56] 
## WMT
[10:54:58] 
## GE
[10:54:59] 
## T
[10:55:01] 
## PFE
[10:55:02] 
## BAC
[10:55:04] 
## C
[10:55:05] 
## GS
[10:55:07] 
## GOOGL
[10:55:08] 
## AMZN
[10:55:09] 
## TSLA
[10:55:11] 
## META
[10:55:13] 
## HD
[10:55:14] 
## CSCO
[10:55:15] 
## DIS
[10:55:17] 
## VZ
[10:55:18] 
## INTC
[10:55:20] 
## Aggregate restate rate by field:
[10:55:20] 
## OVERALL: 0 / 0 = 0.00% of (ticker, quarter, field) values changed since last fetch
[10:55:20] 
Saved report to C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\extension\restate_probe_report.json
[10:55:53] # Step 6 — Restate-detection probe
[10:55:53]   Comparing fresh fetch (today) vs cached values for 20 tickers
[10:55:53]   Average cached-file age: 58.5 days
[10:55:53] 
## AAPL
[10:55:53]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:10/40 eps:0/40 weightedAverageShsOut:0/40]
[10:55:54]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:55:54]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:55:54] 
## MSFT
[10:55:55]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:55:55]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:39/40]
[10:55:56]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:55:56] 
## JPM
[10:55:56]   income-statement          common=40  diff_fields=[revenue:1/40 grossProfit:0/40 operatingIncome:0/40 netIncome:1/40 ebitda:1/40 eps:0/40 weightedAverageShsOut:0/40]
[10:55:57]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:1/40]
[10:55:57]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:55:57] 
## XOM
[10:55:58]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:55:58]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:55:59]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:55:59] 
## WMT
[10:55:59]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:00]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:1/40]
[10:56:00]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:00] 
## GE
[10:56:01]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:01]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:02]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:02] 
## T
[10:56:02]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:03]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:03]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:03] 
## PFE
[10:56:04]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:1/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:04]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:1/40 totalDebt:1/40]
[10:56:05]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:1/40 capitalExpenditure:0/40 freeCashFlow:1/40]
[10:56:05] 
## BAC
[10:56:05]   income-statement          common=40  diff_fields=[revenue:1/40 grossProfit:1/40 operatingIncome:1/40 netIncome:1/40 ebitda:1/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:06]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:06]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:06] 
## C
[10:56:06]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:07]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:07]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:07] 
## GS
[10:56:08]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:1/40 operatingIncome:1/40 netIncome:0/40 ebitda:1/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:08]   balance-sheet-statement   common=40  diff_fields=[totalAssets:1/40 totalEquity:1/40 totalDebt:1/40]
[10:56:09]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:09] 
## GOOGL
[10:56:09]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:10]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:1/40]
[10:56:10]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:10] 
## AMZN
[10:56:11]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:1/40 netIncome:0/40 ebitda:0/40 eps:1/40 weightedAverageShsOut:1/40]
[10:56:11]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:12]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:12] 
## TSLA
[10:56:12]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:13]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:13]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:13] 
## META
[10:56:14]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:14]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:15]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:15] 
## HD
[10:56:15]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:16]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:1/40]
[10:56:16]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:16] 
## CSCO
[10:56:17]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:17]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:18]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:18] 
## DIS
[10:56:18]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:18]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:19]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:19] 
## VZ
[10:56:19]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:20]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:20]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:20] 
## INTC
[10:56:21]   income-statement          common=40  diff_fields=[revenue:0/40 grossProfit:0/40 operatingIncome:0/40 netIncome:0/40 ebitda:0/40 eps:0/40 weightedAverageShsOut:0/40]
[10:56:21]   balance-sheet-statement   common=40  diff_fields=[totalAssets:0/40 totalEquity:0/40 totalDebt:0/40]
[10:56:22]   cash-flow-statement       common=40  diff_fields=[operatingCashFlow:0/40 capitalExpenditure:0/40 freeCashFlow:0/40]
[10:56:22] 
## Aggregate restate rate by field:
[10:56:22]   totalDebt                         45/800   =   5.62%
[10:56:22]   ebitda                            14/800   =   1.75%
[10:56:22]   operatingIncome                    3/800   =   0.38%
[10:56:22]   revenue                            2/800   =   0.25%
[10:56:22]   grossProfit                        2/800   =   0.25%
[10:56:22]   netIncome                          2/800   =   0.25%
[10:56:22]   totalEquity                        2/800   =   0.25%
[10:56:22]   eps                                1/800   =   0.12%
[10:56:22]   weightedAverageShsOut              1/800   =   0.12%
[10:56:22]   totalAssets                        1/800   =   0.12%
[10:56:22]   operatingCashFlow                  1/800   =   0.12%
[10:56:22]   freeCashFlow                       1/800   =   0.12%
[10:56:22]   capitalExpenditure                 0/800   =   0.00%
[10:56:22] 
## OVERALL: 75 / 10400 = 0.72% of (ticker, quarter, field) values changed since last fetch
[10:56:22] 
Saved report to C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\extension\restate_probe_report.json
[10:57:22] # Step 2 — Bulk fetch prices for augmented universe (back to 2010-01-01)
[10:57:22]   Delisted entries: 4173
[10:57:22]   Currently cached (active): 2564
[10:57:22]   Augmented universe size: 6712
[10:58:21]   [100/6712] skipped=0 fetched=99 failed=1  rate=101/min  ETA=65min
[10:59:19]   [200/6712] skipped=0 fetched=199 failed=1  rate=102/min  ETA=64min
[11:00:19]   [300/6712] skipped=0 fetched=299 failed=1  rate=102/min  ETA=63min
[11:01:19]   [400/6712] skipped=0 fetched=399 failed=1  rate=101/min  ETA=62min
[11:02:21]   [500/6712] skipped=0 fetched=499 failed=1  rate=100/min  ETA=62min
[11:03:20]   [600/6712] skipped=0 fetched=599 failed=1  rate=101/min  ETA=61min
[11:04:21]   [700/6712] skipped=0 fetched=699 failed=1  rate=100/min  ETA=60min
[11:05:22]   [800/6712] skipped=0 fetched=798 failed=2  rate=100/min  ETA=59min
[11:06:23]   [900/6712] skipped=0 fetched=898 failed=2  rate=100/min  ETA=58min
[11:07:23]   [1000/6712] skipped=0 fetched=998 failed=2  rate=100/min  ETA=57min
[11:08:27]   [1100/6712] skipped=0 fetched=1098 failed=2  rate=99/min  ETA=57min
[11:09:32]   [1200/6712] skipped=0 fetched=1198 failed=2  rate=99/min  ETA=56min
[11:10:32]   [1300/6712] skipped=0 fetched=1298 failed=2  rate=99/min  ETA=55min
[11:11:32]   [1400/6712] skipped=0 fetched=1398 failed=2  rate=99/min  ETA=54min
[11:12:34]   [1500/6712] skipped=0 fetched=1498 failed=2  rate=99/min  ETA=53min
[11:13:33]   [1600/6712] skipped=0 fetched=1598 failed=2  rate=99/min  ETA=52min
[11:14:36]   [1700/6712] skipped=0 fetched=1698 failed=2  rate=99/min  ETA=51min
[11:15:36]   [1800/6712] skipped=0 fetched=1798 failed=2  rate=99/min  ETA=50min
[11:16:40]   [1900/6712] skipped=0 fetched=1898 failed=2  rate=98/min  ETA=49min
[11:17:37]   [2000/6712] skipped=0 fetched=1997 failed=3  rate=99/min  ETA=48min
[11:18:41]   [2100/6712] skipped=0 fetched=2097 failed=3  rate=99/min  ETA=47min
[11:19:46]   [2200/6712] skipped=0 fetched=2197 failed=3  rate=98/min  ETA=46min
[11:20:48]   [2300/6712] skipped=0 fetched=2297 failed=3  rate=98/min  ETA=45min
[11:21:51]   [2400/6712] skipped=0 fetched=2396 failed=4  rate=98/min  ETA=44min
[11:22:48]   [2500/6712] skipped=0 fetched=2496 failed=4  rate=98/min  ETA=43min
[11:23:46]   [2600/6712] skipped=0 fetched=2596 failed=4  rate=98/min  ETA=42min
[11:24:45]   [2700/6712] skipped=0 fetched=2696 failed=4  rate=99/min  ETA=41min
[11:25:44]   [2800/6712] skipped=0 fetched=2796 failed=4  rate=99/min  ETA=40min
[11:26:46]   [2900/6712] skipped=0 fetched=2896 failed=4  rate=99/min  ETA=39min
[11:27:46]   [3000/6712] skipped=0 fetched=2996 failed=4  rate=99/min  ETA=38min
[11:28:47]   [3100/6712] skipped=0 fetched=3096 failed=4  rate=99/min  ETA=37min
[11:29:47]   [3200/6712] skipped=0 fetched=3196 failed=4  rate=99/min  ETA=36min
[11:30:54]   [3300/6712] skipped=0 fetched=3296 failed=4  rate=98/min  ETA=35min
[11:31:53]   [3400/6712] skipped=0 fetched=3396 failed=4  rate=99/min  ETA=34min
[11:32:58]   [3500/6712] skipped=0 fetched=3496 failed=4  rate=98/min  ETA=33min
[11:33:57]   [3600/6712] skipped=0 fetched=3596 failed=4  rate=98/min  ETA=32min
[11:34:55]   [3700/6712] skipped=0 fetched=3696 failed=4  rate=99/min  ETA=31min
[11:36:00]   [3800/6712] skipped=0 fetched=3796 failed=4  rate=98/min  ETA=30min
[11:37:03]   [3900/6712] skipped=0 fetched=3896 failed=4  rate=98/min  ETA=29min
[11:38:03]   [4000/6712] skipped=0 fetched=3996 failed=4  rate=98/min  ETA=28min
[11:39:09]   [4100/6712] skipped=0 fetched=4096 failed=4  rate=98/min  ETA=27min
[11:40:11]   [4200/6712] skipped=0 fetched=4195 failed=5  rate=98/min  ETA=26min
[11:41:12]   [4300/6712] skipped=0 fetched=4295 failed=5  rate=98/min  ETA=25min
[11:42:11]   [4400/6712] skipped=0 fetched=4395 failed=5  rate=98/min  ETA=24min
[11:43:14]   [4500/6712] skipped=0 fetched=4495 failed=5  rate=98/min  ETA=23min
[11:44:18]   [4600/6712] skipped=0 fetched=4595 failed=5  rate=98/min  ETA=22min
[11:45:18]   [4700/6712] skipped=0 fetched=4695 failed=5  rate=98/min  ETA=21min
[11:46:21]   [4800/6712] skipped=0 fetched=4795 failed=5  rate=98/min  ETA=20min
[11:47:21]   [4900/6712] skipped=0 fetched=4895 failed=5  rate=98/min  ETA=18min
[11:48:25]   [5000/6712] skipped=0 fetched=4995 failed=5  rate=98/min  ETA=17min
[11:49:28]   [5100/6712] skipped=0 fetched=5095 failed=5  rate=98/min  ETA=16min
[11:50:30]   [5200/6712] skipped=0 fetched=5195 failed=5  rate=98/min  ETA=15min
[11:51:32]   [5300/6712] skipped=0 fetched=5295 failed=5  rate=98/min  ETA=14min
[11:52:36]   [5400/6712] skipped=0 fetched=5395 failed=5  rate=98/min  ETA=13min
[11:53:34]   [5500/6712] skipped=0 fetched=5495 failed=5  rate=98/min  ETA=12min
[11:54:36]   [5600/6712] skipped=0 fetched=5595 failed=5  rate=98/min  ETA=11min
[11:55:38]   [5700/6712] skipped=0 fetched=5695 failed=5  rate=98/min  ETA=10min
[11:56:40]   [5800/6712] skipped=0 fetched=5795 failed=5  rate=98/min  ETA=9min
[11:57:42]   [5900/6712] skipped=0 fetched=5895 failed=5  rate=98/min  ETA=8min
[11:58:43]   [6000/6712] skipped=0 fetched=5995 failed=5  rate=98/min  ETA=7min
[11:59:43]   [6100/6712] skipped=0 fetched=6095 failed=5  rate=98/min  ETA=6min
[12:00:51]   [6200/6712] skipped=0 fetched=6194 failed=6  rate=98/min  ETA=5min
[12:01:54]   [6300/6712] skipped=0 fetched=6294 failed=6  rate=98/min  ETA=4min
[12:02:56]   [6400/6712] skipped=0 fetched=6394 failed=6  rate=98/min  ETA=3min
[12:03:57]   [6500/6712] skipped=0 fetched=6494 failed=6  rate=98/min  ETA=2min
[12:05:04]   [6600/6712] skipped=0 fetched=6594 failed=6  rate=97/min  ETA=1min
[12:06:20]   [6700/6712] skipped=0 fetched=6694 failed=6  rate=97/min  ETA=0min
[12:06:28] 
## DONE in 69.1min
[12:06:28]   Skipped (already covered 2010+): 0
[12:06:28]   Fetched: 6706
[12:06:28]   Failed: 6
[12:06:28]   First 30 failures: [('ACICU', 'empty'), ('BDSIW', 'empty'), ('DWL', 'empty'), ('FLE', 'empty'), ('MULN', "write_err:('PyLong is too large to fit int64', 'Conversion failed for column open with type object')"), ('UB', 'empty')]
[12:09:01] # Step 3 — Bulk fundamentals fetch
[12:09:01]   Universe size: 6712 (6706 active + 4173 delisted)
[12:09:01]   Endpoints per ticker: ['income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'key-metrics']  → up to 26848 API calls
[12:11:07]   [50/6712] calls=200 fetched=198 skipped=0 failed=2  rate=95/min  ETA=280min
[12:12:58]   [100/6712] calls=400 fetched=398 skipped=0 failed=2  rate=102/min  ETA=260min
[12:14:44]   [150/6712] calls=600 fetched=598 skipped=0 failed=2  rate=105/min  ETA=250min
[12:16:28]   [200/6712] calls=800 fetched=793 skipped=0 failed=7  rate=107/min  ETA=243min
[12:18:10]   [250/6712] calls=1000 fetched=976 skipped=0 failed=24  rate=109/min  ETA=236min
[12:19:55]   [300/6712] calls=1200 fetched=1168 skipped=0 failed=32  rate=110/min  ETA=233min
[12:21:37]   [350/6712] calls=1400 fetched=1364 skipped=0 failed=36  rate=111/min  ETA=229min
[12:23:21]   [400/6712] calls=1600 fetched=1564 skipped=0 failed=36  rate=112/min  ETA=226min
[12:25:04]   [450/6712] calls=1800 fetched=1760 skipped=0 failed=40  rate=112/min  ETA=223min
[12:26:44]   [500/6712] calls=2000 fetched=1948 skipped=0 failed=52  rate=113/min  ETA=220min
[12:28:30]   [550/6712] calls=2200 fetched=2144 skipped=0 failed=56  rate=113/min  ETA=218min
[12:30:14]   [600/6712] calls=2400 fetched=2336 skipped=0 failed=64  rate=113/min  ETA=216min
[12:31:55]   [650/6712] calls=2600 fetched=2532 skipped=0 failed=68  rate=114/min  ETA=214min
[12:33:39]   [700/6712] calls=2800 fetched=2732 skipped=0 failed=68  rate=114/min  ETA=212min
[12:35:23]   [750/6712] calls=3000 fetched=2912 skipped=0 failed=88  rate=114/min  ETA=210min
[12:37:07]   [800/6712] calls=3200 fetched=3096 skipped=0 failed=104  rate=114/min  ETA=208min
[12:38:48]   [850/6712] calls=3400 fetched=3290 skipped=0 failed=110  rate=114/min  ETA=205min
[12:40:32]   [900/6712] calls=3600 fetched=3490 skipped=0 failed=110  rate=114/min  ETA=204min
[12:42:14]   [950/6712] calls=3800 fetched=3681 skipped=0 failed=119  rate=114/min  ETA=201min
[12:43:57]   [1000/6712] calls=4000 fetched=3869 skipped=0 failed=131  rate=115/min  ETA=200min
[12:45:40]   [1050/6712] calls=4200 fetched=4049 skipped=0 failed=151  rate=115/min  ETA=198min
[12:47:24]   [1100/6712] calls=4400 fetched=4244 skipped=0 failed=156  rate=115/min  ETA=196min
[12:49:06]   [1150/6712] calls=4600 fetched=4440 skipped=0 failed=160  rate=115/min  ETA=194min
[12:50:46]   [1200/6712] calls=4800 fetched=4638 skipped=0 failed=162  rate=115/min  ETA=192min
[12:52:27]   [1250/6712] calls=5000 fetched=4830 skipped=0 failed=170  rate=115/min  ETA=190min
[12:54:10]   [1300/6712] calls=5200 fetched=5024 skipped=0 failed=176  rate=115/min  ETA=188min
[12:55:48]   [1350/6712] calls=5400 fetched=5210 skipped=0 failed=190  rate=115/min  ETA=186min
[12:57:27]   [1400/6712] calls=5600 fetched=5390 skipped=0 failed=210  rate=116/min  ETA=184min
[12:59:06]   [1450/6712] calls=5800 fetched=5590 skipped=0 failed=210  rate=116/min  ETA=182min
[13:00:41]   [1500/6712] calls=6000 fetched=5782 skipped=0 failed=218  rate=116/min  ETA=179min
[13:02:16]   [1550/6712] calls=6200 fetched=5982 skipped=0 failed=218  rate=116/min  ETA=177min
[13:03:52]   [1600/6712] calls=6400 fetched=6178 skipped=0 failed=222  rate=117/min  ETA=175min
[13:05:27]   [1650/6712] calls=6600 fetched=6374 skipped=0 failed=226  rate=117/min  ETA=173min
[13:07:14]   [1700/6712] calls=6800 fetched=6570 skipped=0 failed=230  rate=117/min  ETA=172min
[13:09:20]   [1750/6712] calls=7000 fetched=6770 skipped=0 failed=230  rate=116/min  ETA=171min
[13:11:36]   [1800/6712] calls=7200 fetched=6970 skipped=0 failed=230  rate=115/min  ETA=171min
[13:13:58]   [1850/6712] calls=7400 fetched=7170 skipped=0 failed=230  rate=114/min  ETA=171min
[13:16:18]   [1900/6712] calls=7600 fetched=7366 skipped=0 failed=234  rate=113/min  ETA=170min
[13:18:35]   [1950/6712] calls=7800 fetched=7550 skipped=0 failed=250  rate=112/min  ETA=170min
[13:20:22]   [2000/6712] calls=8000 fetched=7748 skipped=0 failed=252  rate=112/min  ETA=168min
[13:22:12]   [2050/6712] calls=8200 fetched=7924 skipped=0 failed=276  rate=112/min  ETA=166min
[13:24:12]   [2100/6712] calls=8400 fetched=8116 skipped=0 failed=284  rate=112/min  ETA=165min
[13:26:26]   [2150/6712] calls=8600 fetched=8310 skipped=0 failed=290  rate=111/min  ETA=164min
[13:28:41]   [2200/6712] calls=8800 fetched=8498 skipped=0 failed=302  rate=110/min  ETA=163min
[13:30:27]   [2250/6712] calls=9000 fetched=8677 skipped=0 failed=323  rate=111/min  ETA=161min
[13:32:32]   [2300/6712] calls=9200 fetched=8857 skipped=0 failed=343  rate=110/min  ETA=160min
[13:34:25]   [2350/6712] calls=9400 fetched=9049 skipped=0 failed=351  rate=110/min  ETA=158min
[13:36:32]   [2400/6712] calls=9600 fetched=9233 skipped=0 failed=367  rate=110/min  ETA=157min
[13:38:33]   [2450/6712] calls=9800 fetched=9422 skipped=0 failed=378  rate=109/min  ETA=156min
[13:40:06]   [2500/6712] calls=10000 fetched=9621 skipped=0 failed=379  rate=110/min  ETA=153min
[13:41:39]   [2550/6712] calls=10200 fetched=9801 skipped=0 failed=399  rate=110/min  ETA=151min
[13:43:13]   [2600/6712] calls=10400 fetched=9988 skipped=0 failed=412  rate=110/min  ETA=149min
[13:45:01]   [2650/6712] calls=10600 fetched=10176 skipped=0 failed=424  rate=110/min  ETA=147min
[13:46:59]   [2700/6712] calls=10800 fetched=10376 skipped=0 failed=424  rate=110/min  ETA=146min
[13:49:21]   [2750/6712] calls=11000 fetched=10576 skipped=0 failed=424  rate=110/min  ETA=145min
[13:52:10]   [2800/6712] calls=11200 fetched=10760 skipped=0 failed=440  rate=109/min  ETA=144min
[13:55:18]   [2850/6712] calls=11400 fetched=10948 skipped=0 failed=452  rate=107/min  ETA=144min
[13:58:32]   [2900/6712] calls=11600 fetched=11148 skipped=0 failed=452  rate=106/min  ETA=144min
[14:01:44]   [2950/6712] calls=11800 fetched=11340 skipped=0 failed=460  rate=105/min  ETA=144min
[14:05:00]   [3000/6712] calls=12000 fetched=11540 skipped=0 failed=460  rate=103/min  ETA=144min
[14:09:17]   [3050/6712] calls=12200 fetched=11740 skipped=0 failed=460  rate=101/min  ETA=144min
[14:13:52]   [3100/6712] calls=12400 fetched=11923 skipped=0 failed=477  rate=99/min  ETA=145min
[14:18:34]   [3150/6712] calls=12600 fetched=12123 skipped=0 failed=477  rate=97/min  ETA=146min
[14:22:20]   [3200/6712] calls=12800 fetched=12319 skipped=0 failed=481  rate=96/min  ETA=146min
[14:26:05]   [3250/6712] calls=13000 fetched=12515 skipped=0 failed=485  rate=95/min  ETA=146min
[14:30:43]   [3300/6712] calls=13200 fetched=12715 skipped=0 failed=485  rate=93/min  ETA=147min
[14:35:26]   [3350/6712] calls=13400 fetched=12905 skipped=0 failed=495  rate=92/min  ETA=147min
[14:39:46]   [3400/6712] calls=13600 fetched=13105 skipped=0 failed=495  rate=90/min  ETA=147min
[14:44:02]   [3450/6712] calls=13800 fetched=13301 skipped=0 failed=499  rate=89/min  ETA=147min
[14:48:16]   [3500/6712] calls=14000 fetched=13493 skipped=0 failed=507  rate=88/min  ETA=146min
[14:52:32]   [3550/6712] calls=14200 fetched=13681 skipped=0 failed=519  rate=87/min  ETA=146min
[14:56:04]   [3600/6712] calls=14400 fetched=13851 skipped=0 failed=549  rate=86/min  ETA=144min
[14:59:14]   [3650/6712] calls=14600 fetched=14043 skipped=0 failed=557  rate=86/min  ETA=143min
[15:02:31]   [3700/6712] calls=14800 fetched=14231 skipped=0 failed=569  rate=85/min  ETA=141min
[15:05:53]   [3750/6712] calls=15000 fetched=14431 skipped=0 failed=569  rate=85/min  ETA=140min
[15:09:09]   [3800/6712] calls=15200 fetched=14619 skipped=0 failed=581  rate=84/min  ETA=138min
[15:12:34]   [3850/6712] calls=15400 fetched=14819 skipped=0 failed=581  rate=84/min  ETA=136min
[15:16:23]   [3900/6712] calls=15600 fetched=15009 skipped=0 failed=591  rate=83/min  ETA=135min
[15:21:03]   [3950/6712] calls=15800 fetched=15192 skipped=0 failed=608  rate=82/min  ETA=134min
[15:25:21]   [4000/6712] calls=16000 fetched=15370 skipped=0 failed=630  rate=81/min  ETA=133min
[15:29:05]   [4050/6712] calls=16200 fetched=15562 skipped=0 failed=638  rate=81/min  ETA=131min
[15:32:26]   [4100/6712] calls=16400 fetched=15762 skipped=0 failed=638  rate=81/min  ETA=130min
[15:35:48]   [4150/6712] calls=16600 fetched=15954 skipped=0 failed=646  rate=80/min  ETA=128min
[15:39:23]   [4200/6712] calls=16800 fetched=16137 skipped=0 failed=663  rate=80/min  ETA=126min
[15:43:04]   [4250/6712] calls=17000 fetched=16328 skipped=0 failed=672  rate=79/min  ETA=124min
[15:46:41]   [4300/6712] calls=17200 fetched=16524 skipped=0 failed=676  rate=79/min  ETA=122min
[15:50:08]   [4350/6712] calls=17400 fetched=16716 skipped=0 failed=684  rate=79/min  ETA=120min
[15:53:26]   [4400/6712] calls=17600 fetched=16908 skipped=0 failed=692  rate=78/min  ETA=118min
[15:57:01]   [4450/6712] calls=17800 fetched=17100 skipped=0 failed=700  rate=78/min  ETA=116min
[16:00:38]   [4500/6712] calls=18000 fetched=17280 skipped=0 failed=720  rate=78/min  ETA=114min
[16:04:11]   [4550/6712] calls=18200 fetched=17479 skipped=0 failed=721  rate=77/min  ETA=112min
[16:07:58]   [4600/6712] calls=18400 fetched=17675 skipped=0 failed=725  rate=77/min  ETA=110min
[16:11:45]   [4650/6712] calls=18600 fetched=17875 skipped=0 failed=725  rate=77/min  ETA=108min
[16:15:32]   [4700/6712] calls=18800 fetched=18063 skipped=0 failed=737  rate=76/min  ETA=106min
[16:19:23]   [4750/6712] calls=19000 fetched=18255 skipped=0 failed=745  rate=76/min  ETA=103min
[16:23:13]   [4800/6712] calls=19200 fetched=18441 skipped=0 failed=759  rate=76/min  ETA=101min
[16:27:06]   [4850/6712] calls=19400 fetched=18631 skipped=0 failed=769  rate=75/min  ETA=99min
[16:30:56]   [4900/6712] calls=19600 fetched=18827 skipped=0 failed=773  rate=75/min  ETA=97min
[16:34:38]   [4950/6712] calls=19800 fetched=19023 skipped=0 failed=777  rate=75/min  ETA=95min
[16:38:21]   [5000/6712] calls=20000 fetched=19222 skipped=0 failed=778  rate=74/min  ETA=92min
[16:42:08]   [5050/6712] calls=20200 fetched=19414 skipped=0 failed=786  rate=74/min  ETA=90min
[16:46:01]   [5100/6712] calls=20400 fetched=19598 skipped=0 failed=802  rate=74/min  ETA=88min
[16:50:04]   [5150/6712] calls=20600 fetched=19794 skipped=0 failed=806  rate=73/min  ETA=85min
[16:53:43]   [5200/6712] calls=20800 fetched=19986 skipped=0 failed=814  rate=73/min  ETA=83min
[16:57:27]   [5250/6712] calls=21000 fetched=20182 skipped=0 failed=818  rate=73/min  ETA=80min
[17:01:16]   [5300/6712] calls=21200 fetched=20378 skipped=0 failed=822  rate=73/min  ETA=78min
[17:04:55]   [5350/6712] calls=21400 fetched=20566 skipped=0 failed=834  rate=72/min  ETA=75min
[17:08:27]   [5400/6712] calls=21600 fetched=20762 skipped=0 failed=838  rate=72/min  ETA=73min
[17:12:01]   [5450/6712] calls=21800 fetched=20961 skipped=0 failed=839  rate=72/min  ETA=70min
[17:15:41]   [5500/6712] calls=22000 fetched=21149 skipped=0 failed=851  rate=72/min  ETA=68min
[17:19:25]   [5550/6712] calls=22200 fetched=21345 skipped=0 failed=855  rate=72/min  ETA=65min
[17:23:05]   [5600/6712] calls=22400 fetched=21541 skipped=0 failed=859  rate=71/min  ETA=62min
[17:26:44]   [5650/6712] calls=22600 fetched=21737 skipped=0 failed=863  rate=71/min  ETA=60min
[17:30:12]   [5700/6712] calls=22800 fetched=21923 skipped=0 failed=877  rate=71/min  ETA=57min
[17:33:34]   [5750/6712] calls=23000 fetched=22123 skipped=0 failed=877  rate=71/min  ETA=54min
[17:37:09]   [5800/6712] calls=23200 fetched=22315 skipped=0 failed=885  rate=71/min  ETA=52min
[17:40:39]   [5850/6712] calls=23400 fetched=22507 skipped=0 failed=893  rate=71/min  ETA=49min
[17:44:10]   [5900/6712] calls=23600 fetched=22695 skipped=0 failed=905  rate=70/min  ETA=46min
[17:47:28]   [5950/6712] calls=23800 fetched=22886 skipped=0 failed=914  rate=70/min  ETA=43min
[17:51:01]   [6000/6712] calls=24000 fetched=23082 skipped=0 failed=918  rate=70/min  ETA=41min
[17:54:54]   [6050/6712] calls=24200 fetched=23274 skipped=0 failed=926  rate=70/min  ETA=38min
[17:58:44]   [6100/6712] calls=24400 fetched=23458 skipped=0 failed=942  rate=70/min  ETA=35min
[18:02:40]   [6150/6712] calls=24600 fetched=23658 skipped=0 failed=942  rate=70/min  ETA=32min
[18:06:24]   [6200/6712] calls=24800 fetched=23850 skipped=0 failed=950  rate=69/min  ETA=30min
[18:10:12]   [6250/6712] calls=25000 fetched=24050 skipped=0 failed=950  rate=69/min  ETA=27min
[18:13:50]   [6300/6712] calls=25200 fetched=24244 skipped=0 failed=956  rate=69/min  ETA=24min
[18:17:38]   [6350/6712] calls=25400 fetched=24438 skipped=0 failed=962  rate=69/min  ETA=21min
[18:21:07]   [6400/6712] calls=25600 fetched=24622 skipped=0 failed=978  rate=69/min  ETA=18min
[18:24:27]   [6450/6712] calls=25800 fetched=24817 skipped=0 failed=983  rate=69/min  ETA=15min
[18:27:48]   [6500/6712] calls=26000 fetched=25009 skipped=0 failed=991  rate=69/min  ETA=12min
[18:31:07]   [6550/6712] calls=26200 fetched=25209 skipped=0 failed=991  rate=69/min  ETA=9min
[18:34:27]   [6600/6712] calls=26400 fetched=25397 skipped=0 failed=1003  rate=68/min  ETA=7min
[18:37:45]   [6650/6712] calls=26600 fetched=25590 skipped=0 failed=1010  rate=68/min  ETA=4min
[18:40:48]   [6700/6712] calls=26800 fetched=25789 skipped=0 failed=1011  rate=68/min  ETA=1min
[18:41:31] 
## DONE in 392.5min
[18:41:31]   Total API calls: 26848
[18:41:31]   Fetched: 25837, Skipped: 0, Failed: 1011
[18:41:44] # Step 4 — Build PIT universe membership
[18:41:44]   Calendar from AAPL: 4102 bars, 2010-01-04 00:00:00 -> 2026-04-24 00:00:00
[18:41:44]   Tickers with price files: 6706
[18:44:32]   Got intervals for 6706 tickers (skipped 0)
[18:44:34] 
## Universe size by year:
[18:44:34]   2010:  2221 avg active tickers
[18:44:34]   2011:  2313 avg active tickers
[18:44:34]   2012:  2408 avg active tickers
[18:44:34]   2013:  2544 avg active tickers
[18:44:34]   2014:  2744 avg active tickers
[18:44:34]   2015:  2937 avg active tickers
[18:44:34]   2016:  3045 avg active tickers
[18:44:34]   2017:  3100 avg active tickers
[18:44:34]   2018:  3198 avg active tickers
[18:44:34]   2019:  3311 avg active tickers
[18:44:34]   2020:  3592 avg active tickers
[18:44:34]   2021:  4739 avg active tickers
[18:44:34]   2022:  5243 avg active tickers
[18:44:34]   2023:  4393 avg active tickers
[18:44:34]   2024:  3641 avg active tickers
[18:44:35]   2025:  3075 avg active tickers
[18:44:35]   2026:  2638 avg active tickers
[18:44:37] 
## Saved membership matrix to C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\universes_pit\membership.parquet
[18:44:37]   shape: (4102, 6706)
[18:44:37]   Saved per-ticker intervals to C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\universes_pit\ticker_intervals.parquet
[18:44:49] # Step 5 — Re-run PIT matrix rebuild on extended (2010+) data
[18:44:50]   Augmented universe: 6706 tickers from membership
[18:48:45]   Trimmed to 2010+: T=4113 N=6706
[06:25:34] 
[save] Writing 89 parquets to C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[08:00:26]   done in 5691.8s
[08:00:26]   Manifest: C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2\manifest.json
[08:00:26] 
## DONE — extended PIT panel at C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[08:34:20] # Step 7 — Re-run proper_evaluation on extended (2010+) PIT-v2 panel
[08:34:20]   base.PIT_DIR -> C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[08:34:21]   PIT universe (>25% coverage): 4125 tickers
[08:34:24]   Synthetic PITV2 universe written: (4102, 4125)
[08:34:24]   OOS windows: 2013-04-01 → 2026-04-20
[08:34:24]   OUT_DIR: C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\aipt_results\proper_eval_extended
[08:34:24] 
  Launching proper_evaluation.main() ...
[09:35:51] # Step 7 — Re-run proper_evaluation on extended (2010+) PIT-v2 panel
[09:35:51]   base.PIT_DIR -> C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[09:35:51]   PIT universe (>25% coverage): 4125 tickers
[09:35:53]   Synthetic PITV2 universe written: (4102, 4125)
[09:35:53]   OOS windows: 2013-04-01 → 2026-04-20 (run from 2013-01-01)
[09:35:53]   OUT_DIR: C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\aipt_results\proper_eval_extended
[09:35:53] 
  Launching proper_evaluation.main() ...
[15:31:11] # Step 7 — Re-run proper_evaluation on extended (2010+) PIT-v2 panel
[15:31:11]   base.PIT_DIR -> C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[15:31:13]   PIT universe (>25% coverage): 4125 tickers
[15:31:13]   Synthetic PITV2 universe written: (4102, 4125)
[15:31:13]   OOS windows: 2013-04-01 → 2026-04-20 (run from 2013-01-01)
[15:31:13]   OUT_DIR: C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\aipt_results\proper_eval_extended
[15:31:13] 
  Launching proper_evaluation.main() ...
[15:38:24] # Step 7 — Re-run proper_evaluation on extended (2010+) PIT-v2 panel
[15:38:24]   base.PIT_DIR -> C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[15:38:24]   PIT universe (>25% coverage): 4125 tickers
[15:38:25]   Synthetic PITV2 universe written: (4102, 4125)
[15:38:25]   OOS windows: 2013-04-01 → 2026-04-20 (run from 2013-01-01)
[15:38:25]   OUT_DIR: C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\aipt_results\proper_eval_extended
[15:38:25] 
  Launching proper_evaluation.main() ...
[15:41:37] # Step 7 — Re-run proper_evaluation on extended (2010+) PIT-v2 panel
[15:41:37]   base.PIT_DIR -> C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\fmp_cache\matrices_pit_v2
[15:41:37]   PIT universe (>25% coverage): 4125 tickers
[15:41:38]   Synthetic PITV2 universe written: (4102, 4125)
[15:41:38]   OOS windows: 2013-04-01 → 2026-04-20 (run from 2013-01-01)
[15:41:38]   OUT_DIR: C:\Users\breth\PycharmProjects\AutomatedFactorResearcher\factor-alpha-platform\data\aipt_results\proper_eval_extended
[15:41:38] 
  Launching proper_evaluation.main() ...
