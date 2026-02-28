#!/usr/bin/env python3
"""
Enhanced Gemini Agent with comprehensive WorldQuant data fields and learning capabilities
"""

import asyncio
import logging
import re
import json
import random
from typing import Dict, List, Optional, Any
import google.generativeai as genai
from config import Config
import os
from datetime import datetime
try:
    from alpha_tactics import get_tactics_for_groups
except Exception:
    get_tactics_for_groups = None  # Tactics are optional

# Configure data-field groups here (comment out lines to disable groups)
# Valid group names (case-insensitive):
# - MARKET DATA
# - FSCORE METRICS
# - FINANCIAL STATEMENT DATA
# - ADDITIONAL FINANCIAL DATA
# - OPTIONS & VOLATILITY DATA
# - SENTIMENT DATA
# - GROUPING FIELDS
# - SYMBOL FIELDS
ALLOWED_DATA_GROUPS = [
    "ANALYST ESTIMATES DATA",
    #"OPTIONS & VOLATILITY DATA",
    #"MARKET DATA",
    # "FSCORE METRICS",
    "FINANCIAL STATEMENT DATA",
    "ADDITIONAL FINANCIAL DATA",
    "GROUPING FIELDS",
    #"SENTIMENT DATA",
    # "SYMBOL FIELDS",
]

# Default scope percentages (change here to adjust global defaults)
DEFAULT_DATA_FIELDS_PERCENTAGE: float = 33.0 #33 24
DEFAULT_FUNCTIONS_PERCENTAGE: float = 50.0   #33 40

# Default forbidden subexpressions to reduce correlation with past alphas
# Human-readable list and corresponding regex patterns used for enforcement
DEFAULT_FORBIDDEN_SUBEXPRESSIONS_HUMAN: List[str] = [
    "rank(ts_corr(close, vwap, N))",
]
DEFAULT_FORBIDDEN_SUBEXPRESSION_PATTERNS: List[str] = [
    r"rank\(\s*ts_corr\(\s*close\s*,\s*vwap\s*,\s*\d+\s*\)\s*\)",
]

# Full operator and data field reference to embed into prompts
OPERATORS_AND_FIELDS = """
<OPERATORS>
Unlock more complex operators at Expert, Master and Grandmaster Genius levels.
Arithmetic
abs(x) – absolute value of x
add(x, y, filter=false) – add inputs (filter NaNs)
divide(x, y) – x / y
inverse(x) – 1 / x
log(x) – natural log
max(x, y, ..) – maximum of inputs
min(x, y, ..) – minimum of inputs
multiply(x, y, ..., filter=false) – multiply inputs (filter NaNs)
power(x, y) – x ^ y
reverse(x) – -x
sign(x) – sign of x
signed_power(x, y) – sign-preserving power
sqrt(x) – square root
subtract(x, y, filter=false) – x - y

<Logical>
and(x, y) – logical AND
or(x, y) – logical OR
not(x) – logical NOT
if_else(cond, a, b) – ternary conditional
is_nan(x) – 1 if NaN else 0
comparisons: <, <=, ==, !=, >=, >

<Time-Series>
days_from_last_change(x)
ts_arg_max(x, d) / ts_arg_min(x, d)
ts_av_diff(x, d)
ts_backfill(x, d, k=1)
ts_corr(x, y, d)
ts_count_nans(x, d)
ts_covariance(y, x, d)
ts_decay_linear(x, d, dense=false)
ts_delay(x, d)
ts_delta(x, d)
ts_mean(x, d)
ts_product(x, d)
ts_quantile(x, d)
ts_rank(x, d, constant=0)
ts_scale(x, d, constant=0)
ts_std_dev(x, d)
ts_sum(x, d)
ts_zscore(x, d)
ts_regression(y, x, d, lag=0, rettype=0) – OLS regression on two variables over d days (rettype: 0=error, 1=y-int, 2=slope, 3=y-estimate, 4=SSE, 5=SST, 6=R², 7=MSE, 8=SEβ, 9=SEα)
hump(x, hump=0.01)
kth_element(x, d, k)
last_diff_value(x, d)
ts_step(1)

<Cross-Sectional>
normalize(x, useStd=false, limit=0.0)
quantile(x, driver=gaussian, sigma=1.0)
rank(x, rate=2)
scale(x, scale=1, longscale=1, shortscale=1)
winsorize(x, std=4)
zscore(x)

<Vector>
vec_avg(x) – mean of vector
vec_sum(x) – sum of vector

<Transformational>
bucket(rank(x), range="0,1,0.1") – bucketing
trade_when(cond, a, b) – conditional trade mask

<Group>
group_backfill(x, group, d)
group_mean(x, weight, group)
group_neutralize(x, group)
group_rank(x, group)
group_scale(x, group)
group_zscore(x, group)

<DATA FIELDS>
Market Data:
adv20 - Average daily volume in past 20 days
cap - Daily market capitalization (in millions)
close - Daily close price
country - Country grouping
currency - Currency
dividend - Dividend
exchange - Exchange grouping
high - Daily high price
industry - Industry grouping
low - Daily low price
market - Market grouping
open - Daily open price
returns - Daily returns
sector - Sector grouping
sedol - Sedol
sharesout - Daily outstanding shares (in millions)
split - Stock split ratio
subindustry - Subindustry grouping
ticker - Ticker
volume - Daily volume
vwap - Daily volume weighted average price

FScore Metrics:
fscore_growth - The purpose of this metric is to qualify the expected MT growth potential of the stock
fscore_momentum - The purpose of this metric is to identify stocks which are currently undergoing either upward or downward analyst revisions
fscore_profitability - The purpose of this metric is to rank stocks based on their ability to generate cash flows
fscore_quality - The purpose of this metric is to measure both the sustainability and certainty of earnings
fscore_surface - The static score. An index between 0 & 100 is applied for each stock and each composite factor. The first ranking is a pentagon surface-based score. The larger the surface, the higher the rank
fscore_surface_accel - The derivative score. In a second step, we calculate the derivative of this score (i.e., is the surface of the pentagon increasing or decreasing from the previous month?)
fscore_total - The final score, M-Score, is a weighted average of both the Pentagon surface score and the Pentagon acceleration score
fscore_value - The purpose of this metric is to see if the stock is under or overpriced given several well-known valuation standards

Financial Statement Data:
assets - Assets - Total
liabilities - Liabilities - Total
operating_income - Operating Income After Depreciation - Quarterly
sales - Sales/Turnover (Net)
enterprise_value - Enterprise Value
capex - Capital Expenditures
debt - Debt
equity - Common/Ordinary Equity - Total
ebit - Earnings Before Interest and Taxes
ebitda - Earnings Before Interest
eps - Earnings Per Share (Basic) - Including Extraordinary Items
debt_lt - Long-Term Debt - Total
assets_curr - Current Assets - Total
goodwill - Goodwill (net)
income - Net Income
cash - Cash
revenue - Revenue - Total
cashflow_op - Operating Activities - Net Cash Flow
cogs - Cost of Goods Sold
bookvalue_ps - Book Value Per Share
ppent - Property Plant and Equipment - Total (Net)
operating_expense - Operating Expense - Total
debt_st - Debt in Current Liabilities
cashflow - Cashflow (Annual)
inventory - Inventories - Total
liabilities_curr - Current Liabilities - Total
cash_st - Cash and Short-Term Investments
receivable - Receivables - Total
sga_expense - Selling, General and Administrative Expenses
fnd6_fopo - Funds from Operations - Other
return_equity - Return on Equity
retained_earnings - Retained Earnings
income_tax - Income Taxes - Total
pretax_income - Pretax Income
cashflow_fin - Financing Activities - Net Cash Flow
income_beforeextra - Income Before Extraordinary Items
sales_growth - Growth in Sales (Quarterly)
current_ratio - Current Ratio
return_assets - Return on Assets
fnd6_drlt - Deferred Revenue - Long-term
inventory_turnover - Inventory Turnover
sales_ps - Sales per Share (Quarterly)
invested_capital - Invested Capital - Total - Quarterly
cashflow_dividends - Cash Dividends (Cash Flow)
fnd6_drc - Deferred Revenue - Current
fnd6_ivaco - Investing Activities - Other
working_capital - Working Capital (Balance Sheet)
employee - Employees
cashflow_invst - Investing Activities - Net Cash Flow
depre_amort - Depreciation and Amortization - Total
fnd6_mrcta - Thereafter Portion of Leases
fnd6_newa2v1300_ppent - Property, Plant and Equipment - Total (Net)
fnd6_recd - Receivables - Estimated Doubtful
fnd6_fatl - Property, Plant, and Equipment - Leases at Cost
fnd6_rea - Retained Earnings - Restatement
fnd6_acdo - Current Assets of Discontinued Operations
fnd6_ciother - Comp. Inc. - Other Adj.
fnd6_acodo - Other Current Assets Excl Discontinued Operations
interest_expense - Interest and Related Expense - Total
rd_expense - Research And Development (Quarterly)
fnd6_newa2v1300_rdipeps - In Process R&D Expense Basic EPS Effect
fnd6_adesinda_curcd - ISO Currency Code - Company Annual Market
fnd6_xrent - Rental Expense
fnd6_acox - Current Assets - Other - Sundry
fnd6_ci - Comprehensive Income - Total
fnd6_lcox - Current Liabilities - Other - Sundry
fnd6_ein - Employer Identification Number code for the company
fnd6_ceql - Common Equity - Liquidation Value
fnd6_intc - Interest Capitalized
fnd6_newa2v1300_xsga - Selling, General and Administrative Expense
fnd6_zipcode - ZIP code related to the company
fnd6_newa2v1300_oibdp - Operating Income Before Depreciation
fnd6_capxv - Capital Expend Property, Plant and Equipment Schd V
fnd6_newa1v1300_gp - Gross Profit (Loss)
fnd6_newqv1300_ancq - Non-Current Assets - Total
fnd6_newqv1300_drltq - Deferred Revenue - Long-term
fnd6_txo - Income Taxes - Other
fnd6_itci - Investment Tax Credit (Income Account)
fnd6_lcoxdr - Current Liabilities - Other - Excluding Deferred Revenue
fnd6_exre - Exchange Rate Effect
fnd6_newqv1300_lltq - Long-Term Liabilities (Total)
fnd6_ch - Cash
fnd6_cik - nonimportant technical code
fnd6_cicurr - Comp Inc - Currency Trans Adj
fnd6_state - integer for identifying the state of the company
fnd6_am - Amortization of Intangibles
fnd6_newqv1300_seqoq - Other Stockholders' Equity Adjustments
fnd6_mrc2 - Rental Commitments - Minimum - 2nd Year
fnd6_weburl - WEB URL code for the company
fnd6_city - the city where a company's corporate headquarters or home office is located
fnd6_teq - Stockholders' Equity - Total
fnd6_pifo - Pretax Income - Foreign
fnd6_dlto - Debt - Long-Term - Other
fnd6_ds - Debt - Subordinated
fnd6_currencya_curcd - ISO Currency Code - Company Annual Market
fnd6_newqeventv110_drltq - Deferred Revenue - Long-term
fnd6_incorp - Incorporated
fnd6_newqv1300_drcq - Deferred Revenue - Current
fnd6_newa2v1300_prsho - Redeem Pfd Shares Outs (000)
fnd6_ppents - Property, Plant & Equipment
fnd6_dd1 - Long-Term Debt Due in 1 Year
fnd6_newa2v1300_rdipd - In Process R&D Expense Diluted EPS Effect
fnd6_dltr - Long-Term Debt - Reduction
fnd6_mrc5 - Rental Commitments - Minimum - 5th Year
fnd6_optca - Options - Cancelled (-)
fnd6_newa2v1300_ni - Net Income (Loss)
fnd6_incorp - Incorporated
fnd6_newqv1300_lcoq - Current Liabilities - Other - Total
fnd6_fopox - Funds from Operations - Other excluding Option Tax Benefit
fnd6_newqv1300_drcq - Deferred Revenue - Current
fnd6_mrc1 - Rental Commitments - Minimum - 1st Year
fnd6_newa2v1300_xidoc - Extraordinary Items and Discontinued Operations (Cash Flow)
fnd6_cld3 - Capitalized Leases - Due in 3rd Year
fnd6_dd3 - Debt Due in 3rd Year
fnd6_newa1v1300_act - Current Assets - Total
fnd6_newqeventv110_optrfrq - Risk Free Rate - Assumption (%)
fnd6_newa2v1300_optexd - Options - Exercised (-)
fnd6_cptmfmq_lctq - Current Liabilities - Total
fnd6_fyrc - Unimportant technical code, please ignore for research purposes
fnd6_txfo - Income Taxes - Foreign
fnd6_mfmq_cshprq - Common Shares Used to Calculate Earnings Per Share - Basic
fnd6_txbco - Excess Tax Benefit Stock Options - Cash Flow Operating
fnd6_newqv1300_acomincq - Accumulated Other Comprehensive Income (Loss)
fnd6_txdbclq - Current Deferred Tax Liability
fnd6_acqgdwl - Acquired Assets - Goodwill
fnd6_aox - Assets - Other - Sundry
fnd6_newqv1300_rdipq - In Process R&D
fnd6_aqc - Acquisitions
fnd6_cptmfmq_opepsq - Earnings Per Share from Operations
fnd6_cshtrq - Common Shares Traded - Quarter
fnd6_mfma1_at - Assets - Total
fnd6_lifr - LIFO Reserve
fnd6_xopr - Operating Expenses - Total
fnd6_cld2 - Capitalized Leases - Due in 2nd Year
fnd6_loc - string for locating the Headquarters of the company
fnd6_stype - Segment Type
fnd6_newqeventv110_tstkq - Treasury Stock - Total (All Capital)
fnd6_newa1v1300_lo - Liabilities - Other - Total
fnd6_aldo - Long-term Assets of Discontinued Operations
fnd6_cld5 - Capitalized Leases - Due in 5th Year
fnd6_naicss - NAICS Code
fnd6_cptmfmq_ceqq - Common/Ordinary Equity - Total
fnd6_newa1v1300_aco - Current Assets - Other - Total
fnd6_dc - Deferred Charges
fnd6_optex - Options Exercisable (000)
fnd6_newqv1300_intanoq - Other Intangibles
fnd6_mrc4 - Rental Commitments - Minimum - 4th Year
fnd6_cptnewqv1300_dlttq - Long-Term Debt - Total
fnd6_sstk - Sale of Common and Preferred Stock
fnd6_mrc3 - Rental Commitments - Minimum - 3rd Year
fnd6_cptmfmq_atq - Assets - Total
fnd6_cshtr - Common Shares Traded - Annual
fnd6_mrct - Rental Commitments - Minimum - 5-Year Total
fnd6_loxdr - Liabilities - Other - Excluding Deferred Revenue
fnd6_newa1v1300_bkvlps - Book Value Per Share
fnd6_fic - identifies the country in which the company is incorporated or legally registered
fnd6_xad - Advertising Expense
fnd6_eventv110_optlifeq - Life of Options - Assumption (# yrs)
fnd6_newa1v1300_dltt - Long-Term Debt - Total
fnd6_newqv1300_teqq - Stockholders' Equity - Total - Quarterly
fnd6_cptnewqv1300_apq - Accounts Payable/Creditors - Trade
fnd6_newa2v1300_oiadp - Operating Income After Depreciation
fnd6_cptmfmq_dlttq - Long-Term Debt - Total
fnd6_newa2v1300_seq - Stockholders Equity - Parent
fnd6_cptnewqv1300_rectq - Receivables - Total
fnd6_newqeventv110_drcq - Deferred Revenue - Current
fnd6_dcvt - Debt - Convertible
fnd6_cptnewqv1300_oiadpq - Operating Income After Depreciation - Quarterly
fnd6_newqv1300_xoprq - Operating Expense - Total
fnd6_xacc - Accrued Expenses
fnd6_mfma1_csho - Common Shares Outstanding
fnd6_recta - Retained Earnings - Cumulative Translation Adjustment
fnd6_intan - Intangible Assets - Total
fnd6_cstkcvq - Common Stock-Carrying Value
fnd6_mfma2_opeps - Earnings Per Share from Operations
fnd6_cptnewqv1300_oeps12 - Earnings Per Share from Operations - 12 Months Moving
fnd6_cptnewqv1300_opepsq - Earnings Per Share from Operations
fnd6_newqeventv110_prcraq - Repurchase Price - Average per share
fnd6_cibegni - Comp Inc - Beginning Net Income
fnd6_newqeventv110_pstknq - Preferred/Preference Stock - Nonredeemable
fnd6_cipen - Comprehensive Income - Minimum Pension Adjustment
fnd6_optosey - Options Outstanding - End of Year
fnd6_newqv1300_esoprq - Preferred ESOP Obligation - Redeemable
fnd6_cptnewqv1300_lctq - Current Liabilities - Total
fnd6_newa1v1300_dlc - Debt in Current Liabilities - Total
fnd6_cptnewqv1300_epsfxq - Earnings Per Share (Diluted) - Excluding Extraordinary items
fnd6_optdr - Dividend Rate - Assumption (%)
fnd6_cld4 - Capitalized Leases - Due in 4th Year
fnd6_dudd - Debt - Unamortized Debt Discount and Other
fnd6_newqv1300_oepf12 - Earnings Per Share - Diluted - from Operations - 12MM
fnd6_dcvsr - Debt - Senior Convertible
fnd6_newqv1300_recdq - Receivables - Estimated Doubtful
fnd6_aodo - Other Assets excluding Discontinued Operations
fnd6_cptnewqv1300_oibdpq - Operating Income Before Depreciation - Quarterly
fnd6_newa2v1300_rdipa - In-Process R&D Expense After-tax
fnd6_cptnewqv1300_atq - Assets - Total
fnd6_cptnewqv1300_nopiq - Non-Operating Income (Expense) - Total
fnd6_newqv1300_reunaq - Unadjusted Retained Earnings
fnd6_cptnewqv1300_ltq - Liabilities - Total
fnd6_mfma2_revt - Revenue - Total
fnd6_newqv1300_capsq - Capital Surplus/Share Premium Reserve
fnd6_oprepsx - Earnings Per Share - Diluted - from Operations
fnd6_newa1v1300_cstk - Common/Ordinary Stock (Capital)
fnd6_newa1v1300_at - Assets - Total
fnd6_txtubposdec - Decrease - Current Tax Positions
fnd6_cptnewqeventv110_lctq - Current Liabilities - Total
fnd6_cptmfmq_saleq - Sales/Turnover (Net)
fnd6_newqeventv110_esoprq - Preferred ESOP Obligation - Redeemable
fnd6_optosby - Options Outstanding - Beginning of Year
fnd6_optrfr - Risk-Free Rate - Assumption (%)
fnd6_ibmii - Income before Extraordinary Items and Noncontrolling Interests
fnd6_newa1v1300_lct - Current Liabilities - Total
fnd6_cptnewqv1300_ceqq - Common/Ordinary Equity - Total
fnd6_msa - Marketable Securities Adjustment
fnd6_optlife - Life of Options - Assumption (# yrs)
fnd6_dclo - Debt - Capitalized Lease Obligations
fnd6_newa2v1300_mib - Minority Interest (Balance Sheet)
fnd6_newa1v1300_acominc - Accumulated Other Comprehensive Income (Loss)
fnd6_newqv1300_esopnrq - Preferred ESOP Obligation - Non-Redeemable
fnd6_rectr - Receivables - Trade
fnd6_dd1q - Long-Term Debt Due in 1 Year
fnd6_currencyqv1300_curcd - ISO Currency Code - Company Annual Market
fnd6_txpd - Income Taxes Paid
fnd6_mfma1_aoloch - Assets and Liabilities - Other - Net Change
fnd6_newa1v1300_epsfi - Earnings Per Share (Diluted) - Including Extraordinary Items
fnd6_prcc - Price Close - Annual
fnd6_recco - Receivables - Current - Other
fnd6_dd - Debt - Debentures
fnd6_newa1v1300_ibc - Income Before Extraordinary Items (Cash Flow)
fnd6_txw - Excise Taxes
fnd6_newa1v1300_emp - Employees
fnd6_cptmfmq_oibdpq - Operating Income Before Depreciation - Quarterly
fnd6_mfmq_piq - Pretax Income
fnd6_optfvgr - Options - Fair Value of Options Granted
fnd6_newqv1300_dilavq - Dilution Available - Excluding Extraordinary Items
fnd6_eventv110_npq - Notes Payable
fnd6_citotal - Comprehensive Income - Parent
fnd6_ivaeq - Investment and Advances - Equity
fnd6_newa2v1300_reuna - Retained Earnings - Unadjusted
fnd6_fatb - Plant, Property and Equipment at Cost - Buildings
fnd6_newa1v1300_ceq - Common/Ordinary Equity - Total
fnd6_newa1v1300_cshfd - Common Shares Used to Calc Earnings Per Share - Fully Diluted
fnd6_beta - beta
fnd6_xaccq - Accrued Expenses
fnd6_optvol - Volatility - Assumption (%)
fnd6_ranks - Ranking
fnd6_newqv1300_icaptq - Invested Capital - Total - Quarterly
fnd6_newa1v1300_lco - Current Liabilities - Other - Total
fnd6_optprcby - Options Outstanding Beginning of Year - Price
fnd6_newa2v1300_ppegt - Property, Plant and Equipment - Total (Gross)
fnd6_cptnewqv1300_epsf12 - Earnings Per Share (Diluted) - Excluding Extraordinary Items - 12 Months Moving
fnd6_newa1v1300_capx - Capital Expenditures
fnd6_newqv1300_optfvgrq - Options - Fair Value of Options Granted
fnd6_newqv1300_xsgaq - Selling, General and Administrative Expenses
fnd6_newa1v1300_dpc - Depreciation and Amortization (Cash Flow)
fnd6_newqv1300_ibmiiq - Income before Extraordinary Items and Noncontrolling Interests
fnd6_xpp - Prepaid Expenses
fnd6_cstkcv - Common Stock-Carrying Value
fnd6_reajo - Retained Earnings - Other Adjustments
fnd6_prccq - Price Close - Quarter
fnd6_newqv1300_xrdq - Research and Development Expense
fnd6_newa1v1300_lt - Liabilities - Total
fnd6_dd2 - Debt Due in 2nd Year
fnd6_optgr - Options - Granted
fnd6_prclq - Price Low - Quarter
fnd6_acqintan - Acquired Assets - Intangibles
fnd6_newa1v1300_fincf - Financing Activities - Net Cash Flow
fnd6_newqv1300_seqq - Stockholders' Equity - Total - Quarterly
fnd6_newqv1300_aoq - Assets - Other - Total
fnd6_mfma1_capx - Capital Expenditures
fnd6_cptmfmq_actq - Current Assets - Total
fnd6_newqeventv110_cstkeq - Common Stock Equivalents - Dollar Savings
fnd6_newqv1300_cshfdq - Common Shares for Diluted EPS
fnd6_mfma2_oancf - Operating Activities - Net Cash Flow
fnd6_invwip - Inventories - Work In Process
fnd6_newa1v1300_epspi - Earnings Per Share (Basic) - Including Extraordinary Items
fnd6_newqv1300_aociotherq - Accumulated Other Comprehensive Income - Other Adjustments
fnd6_cisecgl - Comp Inc - Securities Gains/Losses
fnd6_mfma1_dpc - Depreciation and Amortization (Cash Flow)
fnd6_optprcex - Options Exercised - Price
fnd6_txtubend - Unrecog. Tax Benefits - End of Year
fnd6_pnrsho - Nonred Pfd Shares Outs (000)
fnd6_newa1v1300_apalch - Accounts Payable and Accrued Liabilities - Increase/(Decrease)
fnd6_dvpa - Preferred Dividends in Arrears
fnd6_newqv1300_ciotherq - Comp Inc - Other Adj
fnd6_donr - Nonrecurring Disc Operations
fnd6_mibn - Noncontrolling Interests - Nonredeemable - Balance Sheet
fnd6_fatc - Plant, Property and Equipment at Cost - Construction in Progress
fnd6_newqv1300_dcomq - Deferred Compensation
fnd6_optprcca - Options Cancelled - Price
fnd6_txdbca - Deferred Tax Asset - Current
fnd6_cshpri - Common Shares Used to Calculate Earnings Per Share - Basic
fnd6_newa2v1300_seqo - Other Stockholders' Equity Adjustments
fnd6_mfmq_cheq - Cash and Short-Term Investments
fnd6_optprcgr - Options Granted - Price
fnd6_newqv1300_txditcq - Deferred Taxes and Investment Tax Credit
fnd6_idesindq_curcd - ISO Currency Code - Company Annual Market
fnd6_newa1v1300_dvc - Dividends Common/Ordinary
fnd6_newqv1300_txwq - Excise Taxes
fnd6_newqv1300_ppentq - Property Plant and Equipment - Total (Net)

Analyst Estimates Data:
- anl4_adjusted_netincome_ft - Adjusted net income - forecast type (revision/new/...) — Matrix — Coverage 87%
- anl4_ptp_flag - Pretax income - forecast type (revision/new/...) — Matrix — Coverage 83%
- anl4_ebit_value - Earnings before interest and taxes - announced financial value — Matrix — Coverage 49%
- anl4_ebitda_value - Earnings before interest, taxes, depreciation and amortization - announced financial value — Matrix — Coverage 45%
- anl4_ptpr_number - Reported Pretax Income - number of estimations — Matrix — Coverage 41%
- anl4_epsr_flag - GAAP Earnings - estimation type (revision/new/...), per share — Matrix — Coverage 87%
- anl4_flag_erbfintax - Earnings before interest and taxes - forecast type (revision/new/...) — Matrix — Coverage 82%
- anl4_capex_high - Capital Expenditures - The highest estimation — Matrix — Coverage 62%
- anl4_netprofit_flag - Net profit - forecast type (revision/new/...) — Matrix — Coverage 88%
- anl4_bvps_flag - Book value per share - forecast type (revision/new/...) — Matrix — Coverage 82%
- anl4_capex_low - Capital Expenditures - The lowest estimation — Matrix — Coverage 62%
- anl4_gric_flag - Gross income - forecast type (revision/new/...) — Matrix — Coverage 90%
- anl4_cfo_low - Cash Flow From Operations - The lowest estimation — Matrix — Coverage 62%
- anl4_ebitda_low - Earnings before interest, taxes, depreciation and amortization - The lowest estimation — Matrix — Coverage 58%
- anl4_cfo_value - Cash Flow From Operations - announced financial value — Matrix — Coverage 32%
- anl4_capex_value - Capital Expenditures - announced financial value — Matrix — Coverage 37%
- est_capex - Capital Expenditures - mean of estimations — Matrix — Coverage 64%
- anl4_ptp_number - Pretax Income - number of estimations — Matrix — Coverage 73%
- anl4_eaz2lrec_ratingvalue - Score on the given instrument — Vector — Coverage 53%
- anl4_qfd1_az_eps_number - Earnings per share - number of estimations — Matrix — Coverage 78%
- anl4_ady_pu - The number of upper estimations — Vector — Coverage 72%
- anl4_epsr_low - GAAP Earnings per share - The lowest estimation — Matrix — Coverage 72%
- est_rd_expense - Research and Development Expense - mean of estimations — Matrix — Coverage 31%
- anl4_ebitda_high - Earnings before interest, taxes, depreciation, and amortization - the highest estimation — Matrix — Coverage 58%
- anl4_cff_low - Cash Flow From Financing - The lowest estimation — Matrix — Coverage 59%
- anl4_fsdetailrecv4v104_item - Financial item — Vector — Coverage 80%
- anl4_fcfps_number - Free Cash Flow per Share - number of estimations — Matrix — Coverage 44%
- anl4_totassets_number - Total Assets - number of estimations — Matrix — Coverage 72%
- anl4_netdebt_flag - Net debt - forecast type (revision/new/...) — Matrix — Coverage 86%
- anl4_qf_az_eps_number - Earnings per share - number of estimations — Matrix — Coverage 78%
- anl4_ptp_low - Pretax income - the lowest estimation — Matrix — Coverage 73%
- anl4_adxqfv110_pu - The number of upper estimations — Vector — Coverage 70%
- anl4_fsdtlestmtafv4_item - Financial item — Vector — Coverage 79%
- anl4_tbve_ft - Tangible Book Value per Share - forecast type (revision/new/...) — Matrix — Coverage 93%
- anl4_totassets_flag - Total Assets - forecast type (revision/new/...) — Matrix — Coverage 81%
- anl4_netprofit_number - Net profit - number of estimations — Matrix — Coverage 77%
- anl4_fsguidancebasicqfv4_item - Financial item — Vector — Coverage 32%
- anl4_bac1detaillt_item - Financial item — Vector — Coverage 68%
- anl4_dei3lltv110_item - Financial item — Vector — Coverage 47%
- anl4_cuo1detailafv110_item - Financial item — Vector — Coverage 69%
- est_eps - Earnings per share - mean of estimations — Matrix — Coverage 78%
- anl4_netprofita_low - Adjusted net income - the lowest estimation — Matrix — Coverage 75%
- anl4_median_epsreported - GAAP Earnings per share - median of estimations — Matrix — Coverage 72%
- anl4_ebit_low - Earnings before interest and taxes - The lowest estimation — Matrix — Coverage 71%
- anl4_fsdetailltv4v104_item - Financial item — Vector — Coverage 80%
- anl4_fcf_flag - Free cash flow - forecast type (revision/new/...) — Matrix — Coverage 86%
- anl4_epsr_number - GAAP Earnings per share - number of estimations — Matrix — Coverage 72%
- anl4_ebit_high - Earnings before interest and taxes - The highest estimation — Matrix — Coverage 71%
- anl4_rd_exp_flag - Research and Development Expense - forecast type (revision/new/...) — Matrix — Coverage 98%
- anl4_basicconqfv110_pu - The number of upper estimations — Vector — Coverage 73%
- anl4_ebitda_flag - Earnings before interest, taxes, depreciation and amortization - forecast type (revision/new/...) — Matrix — Coverage 86%
- anl4_fcf_low - Free Cash Flow - The lowest estimation — Matrix — Coverage 55%
- anl4_rd_exp_mean - Research and Development Expense - mean of estimations — Matrix — Coverage 29%
- anl4_capex_std - Capital Expenditures - standard deviation of estimations — Matrix — Coverage 26%
- est_epsr - GAAP Earnings per share - mean of estimations — Matrix — Coverage 74%
- anl4_cfi_low - Cash Flow From Investing - The lowest estimation — Matrix — Coverage 58%
- anl4_dei2lqfv110_item - Financial item — Vector — Coverage 43%
- anl4_fsguidanceqfv4_maxguidance - Max guidance value — Vector — Coverage 31%
- est_sales - Sales - mean of estimations — Matrix — Coverage 75%
- anl4_capex_flag - Capital Expenditures - forecast type (revision/new/...) — Matrix — Coverage 85%
- anl4_ptpr_low - Reported Pretax Income - The Lowest Estimation — Matrix — Coverage 41%
- anl4_rd_exp_low - Research and Development Expense - the lowest estimation — Matrix — Coverage 29%
- anl4_ptp_high - Pretax income - the highest estimation — Matrix — Coverage 73%
- anl4_fsactualafv4_item - Financial item — Vector — Coverage 79%
- anl4_cuo1detailqfv110_item - Financial item — Vector — Coverage 63%
- anl4_guibasicqfv4_est - Estimation value — Vector — Coverage 32%
- anl4_gric_std - Gross income - std of estimations — Matrix — Coverage 41%
- anl4_epsr_high - GAAP Earnings per share - The highest estimation — Matrix — Coverage 72%
- anl4_total_rec - The total number of recommendations — Vector — Coverage 62%
- anl4_bac1detailrec_item - Financial item — Vector — Coverage 71%
- anl4_dts_rspe - Reported Earnings per share - standard deviation of estimations — Matrix — Coverage 48%
- anl4_ptpr_flag - Reported Pretax income - forecast type (revision/new/...) — Matrix — Coverage 90%
- anl4_cfo_median - Cash Flow From Operations - median of estimations — Matrix — Coverage 62%
- anl4_cfo_high - Cash Flow From Operations - The highest value among forecasts — Matrix — Coverage 62%
- anl4_fsguidanceqfv4_minguidance - Min guidance value — Vector — Coverage 31%
- anl4_cff_flag - Cash Flow From Financing Activities - forecast type (revision/new/...) — Matrix — Coverage 82%
- anl4_cfi_value - Cash Flow From Investing - announced financial value — Matrix — Coverage 26%
- anl4_gric_value - Gross Income - announced financial value — Matrix — Coverage 35%
- anl4_ebit_number - Earnings before interest and taxes - number of estimations — Matrix — Coverage 71%
- est_ebit - Earnings before interest and taxes - mean of estimations — Matrix — Coverage 73%
- est_ebitda - Earnings before interest, taxes, depreciation, and amortization - mean of estimations — Matrix — Coverage 60%
- anl4_totassets_low - Total Assets - The lowest estimation — Matrix — Coverage 72%
- anl4_netprofita_number - Adjusted net income - number of estimations — Matrix — Coverage 75%
- est_netprofit - Net profit - mean of estimations — Matrix — Coverage 79%
- anl4_cff_high - Cash Flow From Financing - The highest of forecasted values — Matrix — Coverage 59%
- anl4_ebit_median - Earnings before interest and taxes - median of estimations — Matrix — Coverage 71%
- anl4_fsguidanceqfv4_item - Financial item — Vector — Coverage 31%
- anl4_gric_low - Gross income - The lowest estimation — Matrix — Coverage 60%
- anl4_fcf_value - Free cash flow - announced financial value — Matrix — Coverage 28%
- anl4_epsr_value - GAAP Earnings per share - announced financial value — Matrix — Coverage 57%
- anl4_guiqfv4_est - Estimation value — Vector — Coverage 31%
- anl4_cfo_flag - Cash Flow From Operations - forecast type (revision/new/...) — Matrix — Coverage 83%
- anl4_basicconltv110_numest - The number of forecasts counted in aggregation — Vector — Coverage 70%
- est_cashflow_op - Cash Flow From Operations - mean of estimations — Matrix — Coverage 65%
- anl4_netprofit_value - Net profit - announced financial value — Matrix — Coverage 58%
- anl4_basicconafv110_pu - The number of upper estimations — Vector — Coverage 73%
- anl4_tbvps_number - Tangible Book Value per Share - number of estimations — Matrix — Coverage 33%
- anl4_cfi_high - Cash Flow From Investing - The highest estimation — Matrix — Coverage 58%
- anl4_eaz2lrec_person - Broker Id — Vector — Coverage 55%
- anl4_bvps_low - Book value - the lowest estimation, per share — Matrix — Coverage 65%

Sentiment Data:
scl12_alltype_buzzvec - sentiment volume - Vector — Matrix — Coverage 95%
scl12_alltype_sentvec - sentiment - Vector — Matrix — Coverage 95%
scl12_alltype_typevec - instrument type index - Vector — Matrix — Coverage 95%
scl12_buzz - relative sentiment volume - Matrix — Coverage 95%
scl12_sentiment - sentiment - Matrix — Coverage 94%
snt_buzz - negative relative sentiment volume, fill nan with 0 - Matrix — Coverage 95%
snt_buzz_bfl - negative relative sentiment volume, fill nan with 1 - Matrix — Coverage 100%
snt_buzz_ret - negative return of relative sentiment volume - Matrix — Coverage 95%
snt_value - negative sentiment, fill nan with 0 - Matrix — Coverage 94%
snt_social_value - Z score of sentiment - Matrix — Coverage 86%
snt_social_volume - Normalized tweet volume - Matrix — Coverage 86%

Options & Volatility Data:
historical_volatility_10 - Close-to-close historical volatility over 10 days
historical_volatility_20 - Close-to-close historical volatility over 20 days
historical_volatility_30 - Close-to-close historical volatility over 30 days
historical_volatility_60 - Close-to-close historical volatility over 60 days
historical_volatility_90 - Close-to-close historical volatility over 90 days
historical_volatility_120 - Close-to-close historical volatility over 120 days
historical_volatility_150 - Close-to-close historical volatility over 150 days
historical_volatility_180 - Close-to-close historical volatility over 180 days
parkinson_volatility_10 - Parkinson model's historical volatility over 2 weeks
parkinson_volatility_20 - Parkinson model's historical volatility over 20 days
parkinson_volatility_30 - Parkinson model's historical volatility over 30 days
parkinson_volatility_60 - Parkinson model's historical volatility over 60 days
parkinson_volatility_90 - Parkinson model's historical volatility over 90 days
parkinson_volatility_120 - Parkinson model's historical volatility over 120 days
parkinson_volatility_150 - Parkinson model's historical volatility over 150 days
parkinson_volatility_180 - Parkinson model's historical volatility over 180 days
implied_volatility_call_10 - At-the-money option-implied volatility for call option for 10 days
implied_volatility_call_20 - At-the-money option-implied volatility for call option for 20 days
implied_volatility_call_30 - At-the-money option-implied volatility for call option for 30 days
implied_volatility_call_60 - At-the-money option-implied volatility for call option for 60 days
implied_volatility_call_90 - At-the-money option-implied volatility for call option for 90 days
implied_volatility_call_120 - At-the-money option-implied volatility for call option for 120 days
implied_volatility_call_150 - At-the-money option-implied volatility for call option for 150 days
implied_volatility_call_180 - At-the-money option-implied volatility for call option for 180 days
implied_volatility_call_270 - At-the-money option-implied volatility for call option for 270 days
implied_volatility_call_360 - At-the-money option-implied volatility for call option for 360 days
implied_volatility_call_720 - At-the-money option-implied volatility for call option for 720 days
implied_volatility_call_1080 - At-the-money option-implied volatility for call option for 1080 days
implied_volatility_put_10 - At-the-money option-implied volatility for put option for 10 days
implied_volatility_put_20 - At-the-money option-implied volatility for put option for 20 days
implied_volatility_put_30 - At-the-money option-implied volatility for put option for 30 days
implied_volatility_put_60 - At-the-money option-implied volatility for put option for 60 days
implied_volatility_put_90 - At-the-money option-implied volatility for put option for 90 days
implied_volatility_put_120 - At-the-money option-implied volatility for put option for 120 days
implied_volatility_put_150 - At-the-money option-implied volatility for put option for 150 days
implied_volatility_put_180 - At-the-money option-implied volatility for put option for 180 days
implied_volatility_put_270 - At-the-money option-implied volatility for put option for 270 days
implied_volatility_put_360 - At-the-money option-implied volatility for put option for 360 days
implied_volatility_put_720 - At-the-money option-implied volatility for put option for 720 days
implied_volatility_put_1080 - At-the-money option-implied volatility for put option for 3 years
implied_volatility_mean_10 - At-the-money option-implied volatility mean for 10 days
implied_volatility_mean_20 - At-the-money option-implied volatility mean for 20 days
implied_volatility_mean_30 - At-the-money option-implied volatility mean for 30 days
implied_volatility_mean_60 - At-the-money option-implied volatility mean for 60 days
implied_volatility_mean_90 - At-the-money option-implied volatility mean for 90 days
implied_volatility_mean_120 - At-the-money option-implied volatility mean for 120 days
implied_volatility_mean_150 - At-the-money option-implied volatility mean for 150 days
implied_volatility_mean_180 - At-the-money option-implied volatility mean for 180 days
implied_volatility_mean_270 - At-the-money option-implied volatility mean for 270 days
implied_volatility_mean_360 - At-the-money option-implied volatility mean for 360 days
implied_volatility_mean_720 - At-the-money option-implied volatility mean for 720 days
implied_volatility_mean_1080 - At-the-money option-implied volatility mean for 3 years
implied_volatility_mean_skew_10 - At-the-money option-implied volatility mean skew for 10 days
implied_volatility_mean_skew_20 - At-the-money option-implied volatility mean skew for 20 days
implied_volatility_mean_skew_30 - At-the-money option-implied volatility mean skew for 30 days
implied_volatility_mean_skew_60 - At-the-money option-implied volatility mean skew for 60 days
implied_volatility_mean_skew_90 - At-the-money option-implied volatility mean skew for 90 days
implied_volatility_mean_skew_120 - At-the-money option-implied volatility mean skew for 120 days
implied_volatility_mean_skew_150 - At-the-money option-implied volatility mean skew for 150 days
implied_volatility_mean_skew_180 - At-the-money option-implied volatility mean skew for 180 days
implied_volatility_mean_skew_270 - At-the-money option-implied volatility mean skew for 270 days
implied_volatility_mean_skew_360 - At-the-money option-implied volatility mean skew for 360 days
implied_volatility_mean_skew_720 - At-the-money option-implied volatility mean skew for 720 days
implied_volatility_mean_skew_1080 - At-the-money option-implied volatility mean skew for 3 years
"""

class EnhancedGeminiAgent:
    """Enhanced Gemini agent for WorldQuant alpha generation with learning capabilities"""
    
    def __init__(self, history_lookback: int = 30, data_fields_percentage: float = DEFAULT_DATA_FIELDS_PERCENTAGE, functions_percentage: float = DEFAULT_FUNCTIONS_PERCENTAGE, allowed_data_groups: Optional[List[str]] = None):
        self.logger = logging.getLogger(__name__)
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        # History lookback window for prompts/diversity checks
        self.history_lookback = max(1, int(history_lookback))
        
        # Scope percentages
        self.data_fields_percentage = max(10.0, min(100.0, data_fields_percentage))
        self.functions_percentage = max(10.0, min(100.0, functions_percentage))

        # Cached scoped operators/fields for this agent run (sampled once on first use)
        self._scoped_operators_and_fields_text: Optional[str] = None

        # Build whitelist of known function names from operator catalog for hallucination checks
        self._allowed_function_names = self._build_allowed_function_names()

        # Configure which data field groups are eligible for selection
        # If constructor param is provided, use it; otherwise use ALLOWED_DATA_GROUPS constant
        if allowed_data_groups:
            self.allowed_data_groups = set(s.strip().upper().rstrip(':') for s in allowed_data_groups)
        else:
            # Use module-level constant for code-based configuration
            self.allowed_data_groups = set(s.strip().upper().rstrip(':') for s in ALLOWED_DATA_GROUPS) if ALLOWED_DATA_GROUPS else None
        
        # Connect to DB for learning patterns
        try:
            from mysql_database import MySQLAlphaDatabase
            self.db = MySQLAlphaDatabase()
        except Exception:
            self.db = None

        # Per-run thought history used for in-context learning within a single run
        # Each entry: { trial_index, alpha_code, model_reasoning, parameters, outcome, ts }
        self.thought_history: List[Dict[str, Any]] = []

        # Forbidden patterns (regex) seeded from defaults; can be extended at runtime
        self.forbidden_subexpression_patterns: List[str] = list(DEFAULT_FORBIDDEN_SUBEXPRESSION_PATTERNS)

    def _clean_alpha_code(self, alpha_code: str) -> str:
        """Clean alpha code by removing markdown artifacts and ensuring valid FASTEXPR syntax"""
        # Remove markdown formatting
        cleaned = re.sub(r'[*`]+', '', alpha_code)  # Remove bold/italic markers and backticks
        cleaned = re.sub(r'^\s*[-*]\s*', '', cleaned)  # Remove list bullets
        cleaned = re.sub(r'\n+', ' ', cleaned)  # Replace newlines with spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()

        # New: Truncate any leading narrative before first operator occurrence
        first_op_idx = -1
        for marker in ['rank(', 'ts_', 'group_', 'divide(', 'multiply(', 'add(', 'subtract(']:
            idx = cleaned.find(marker)
            if idx != -1:
                first_op_idx = idx if first_op_idx == -1 else min(first_op_idx, idx)
        if first_op_idx > 0:
            cleaned = cleaned[first_op_idx:]
        
        # Remove any trailing punctuation that might cause syntax errors
        cleaned = re.sub(r'[.,;]+$', '', cleaned)
        
        # CRITICAL FIX: Add missing operators between function calls
        # Pattern: function(...) function(...) -> function(...) * function(...)
        cleaned = re.sub(r'\)\s+rank\(', ') * rank(', cleaned)
        cleaned = re.sub(r'\)\s+ts_', ') * ts_', cleaned)
        cleaned = re.sub(r'\)\s+group_', ') * group_', cleaned)
        cleaned = re.sub(r'\)\s+reverse\(', ') * reverse(', cleaned)
        cleaned = re.sub(r'\)\s+multiply\(', ') * multiply(', cleaned)
        cleaned = re.sub(r'\)\s+divide\(', ') * divide(', cleaned)
        cleaned = re.sub(r'\)\s+add\(', ') * add(', cleaned)
        cleaned = re.sub(r'\)\s+subtract\(', ') * subtract(', cleaned)
        cleaned = re.sub(r'\)\s+([a-zA-Z_]\w*\()', r') * \1', cleaned)
        
        # Fix subtract(0, x) unit mismatch by swapping arguments
        cleaned = re.sub(r'subtract\(\s*0\s*,\s*([^\)]+)\)', r'multiply(-1, \1)', cleaned)
        
        # CRITICAL UNIT COMPATIBILITY FIXES:
        # Fix min(0, returns) unit mismatch - min with constant and price data
        cleaned = re.sub(r'min\(\s*0\s*,\s*returns\s*\)', r'multiply(-1, rank(returns))', cleaned)
        cleaned = re.sub(r'min\(\s*0\s*,\s*([^\)]+)\s*\)', r'multiply(-1, rank(\1))', cleaned)
        
        # Fix max(0, returns) unit mismatch - max with constant and price data
        cleaned = re.sub(r'max\(\s*0\s*,\s*returns\s*\)', r'rank(returns)', cleaned)
        cleaned = re.sub(r'max\(\s*0\s*,\s*([^\)]+)\s*\)', r'rank(\1)', cleaned)
        
        # Fix min/max with incompatible units - replace with rank-based alternatives
        cleaned = re.sub(r'min\(\s*([^,]+),\s*([^,]+)\s*\)', r'rank(\1)', cleaned)  # Use first argument
        cleaned = re.sub(r'max\(\s*([^,]+),\s*([^,]+)\s*\)', r'rank(\2)', cleaned)  # Use second argument
        
        # Fix missing operator between numeric literal and function (e.g., "-1 ts_zscore(")
        cleaned = re.sub(r'(-?\d+)\s+([a-zA-Z_]\w*\()', r'\1 * \2', cleaned)

        # Fix missing operator between numeric literal and parenthesis (e.g., "-1 (x)")
        cleaned = re.sub(r'(-?\d+(?:\.\d+)?)\s*\(', r'\1 * (', cleaned)

        # Fix ts_corr unit mismatches - ensure compatible units
        cleaned = re.sub(r'ts_corr\(\s*([^,]+),\s*volume\s*,\s*([^,]+)\)', r'ts_corr(\1, close, \2)', cleaned)
        
        # Fix common syntax issues
        cleaned = re.sub(r'ts_std\(', 'ts_std_dev(', cleaned)  # Fix function name
        cleaned = re.sub(r'ts_mean\(', 'ts_mean(', cleaned)  # Ensure correct function names
        
        # FINAL SAFETY CLEANUP:
        # WorldQuant FASTEXPR does not use backslash escape sequences, and any stray
        # backslashes (e.g. "\1") cause errors like:
        # "Unexpected character '\' near 'ts_zscore(\1, 60), 1'".
        # Strip all backslashes so the engine sees plain numeric/text tokens instead
        # of regex-style backreferences.
        cleaned = cleaned.replace('\\', '')
        
        return cleaned

    def _fix_operator_names(self, alpha_code: str) -> str:
        """Fix common operator name errors in alpha code (from simple_agent.py)"""
        # Only fix the most critical operator name issues
        corrections = {
            r'\bdecay_linear\b': 'ts_decay_linear',
            r'\bdelay\b': 'ts_delay',
            r'\bdelta\b': 'ts_delta',
            r'\bmean\b': 'ts_mean',
            r'\bsum\b': 'ts_sum',
            r'\bcorr\b': 'ts_corr',
            r'\bstd_dev\b': 'ts_std_dev',
            r'\bzscore\b': 'ts_zscore',
            r'\bts_arg_rank\b': 'ts_rank',
        }
        
        for wrong_pattern, correct_name in corrections.items():
            alpha_code = re.sub(wrong_pattern, correct_name, alpha_code)
        
        return alpha_code

    def _build_allowed_function_names(self) -> set:
        """Parse the catalog to extract allowed function names for validation."""
        names = set()
        try:
            # Extract lines from the <OPERATORS> section
            text = OPERATORS_AND_FIELDS
            start = text.find('<OPERATORS>')
            end = text.find('<DATA FIELDS>')
            if start != -1:
                section = text[start:end] if end != -1 else text[start:]
                for line in section.split('\n'):
                    s = line.strip()
                    if not s or s.startswith('<') or s in ('Arithmetic',):
                        continue
                    # capture token before '('
                    m = re.match(r'([a-zA-Z_]\w*)\s*\(', s)
                    if m:
                        names.add(m.group(1))
            # Add logical operators often used as functions
            names.update({'if_else', 'is_nan', 'and', 'or', 'not'})
        except Exception:
            pass
        return names

    def _unknown_function_names(self, alpha_code: str) -> set:
        """Return any function identifiers not in the allowed set."""
        found = set(re.findall(r'([a-zA-Z_]\w*)\s*\(', alpha_code))
        # Ignore numeric-like or obvious safe names just in case
        return {fn for fn in found if fn not in self._allowed_function_names}

    def _get_forbidden_field_patterns(self) -> List[str]:
        """Patterns for fields that must not appear based on excluded groups."""
        patterns: List[str] = []
        # If options/vol group not allowed, block their common prefixes
        if self.allowed_data_groups is not None and 'OPTIONS & VOLATILITY DATA' not in self.allowed_data_groups:
            patterns.extend([
                r'\bimplied_volatility_[a-z_0-9]+',
                r'\bhistorical_volatility_[0-9]+',
                r'\bparkinson_volatility_[0-9]+'
            ])
        return patterns

    def _get_allowed_field_examples(self, max_count: int = 12) -> List[str]:
        """Return example field identifiers from the currently allowed groups for prompt guidance."""
        examples: List[str] = []
        # Reuse the parsing in _get_scoped_operators_and_fields by reading the allowed section directly
        full_text = OPERATORS_AND_FIELDS
        data_fields_start = full_text.find('<DATA FIELDS>')
        if data_fields_start == -1:
            return examples
        section = full_text[data_fields_start:]
        # Track group
        current_group = None
        def detect_group(header_line: str) -> Optional[str]:
            s = header_line.strip().upper().rstrip(':')
            if s.startswith('MARKET DATA'):
                return 'MARKET DATA'
            if s.startswith('FSCORE METRICS'):
                return 'FSCORE METRICS'
            if s.startswith('FINANCIAL STATEMENT DATA'):
                return 'FINANCIAL STATEMENT DATA'
            if s.startswith('ADDITIONAL FINANCIAL DATA'):
                return 'ADDITIONAL FINANCIAL DATA'
            if s.startswith('ANALYST ESTIMATES DATA'):
                return 'ANALYST ESTIMATES DATA'
            if s.startswith('SENTIMENT DATA'):
                return 'SENTIMENT DATA'
            if s.startswith('OPTIONS & VOLATILITY DATA'):
                return 'OPTIONS & VOLATILITY DATA'
            if s.startswith('GROUPING FIELDS'):
                return 'GROUPING FIELDS'
            if s.startswith('SYMBOL FIELDS'):
                return 'SYMBOL FIELDS'
            return None
        for raw_line in section.split('\n'):
            line = raw_line.strip()
            grp = detect_group(line)
            if grp is not None:
                current_group = grp
                continue
            if not line or ' - ' not in line:
                continue
            if self.allowed_data_groups is None or (current_group in self.allowed_data_groups):
                # take field name before ' - '
                field_name = line.split(' - ', 1)[0].strip('- ').strip()
                if field_name:
                    examples.append(field_name)
                    if len(examples) >= max_count:
                        break
        return examples
    
    def _validate_alpha_syntax(self, alpha_code: str) -> bool:
        """Validate that alpha code has basic syntax validity before testing"""
        try:
            # Check for basic syntax issues
            if not alpha_code or len(alpha_code.strip()) < 5:
                return False
            
            # Check for balanced parentheses
            if alpha_code.count('(') != alpha_code.count(')'):
                return False
            
            # Check for proper function structure
            if not re.search(r'\w+\(', alpha_code):
                return False
            
            # Check for common syntax errors
            if re.search(r'[,\s]+\)', alpha_code):  # Empty function arguments
                return False
            
            if re.search(r'\(\s*\)', alpha_code):  # Empty parentheses
                return False
            
            # Check for trailing commas
            if re.search(r',\s*\)', alpha_code):
                return False
            
            # Check for double commas
            if re.search(r',\s*,\s*', alpha_code):
                return False
            
            # Check for malformed function calls (like "se, 120)) reverse(ra")
            if re.search(r'[a-zA-Z_]\w*[,\s]+[0-9]+\s*\)\s*\)\s*[a-zA-Z_]\w*', alpha_code):
                return False
            
            # Check for proper operator usage
            if re.search(r'[+\-*/]\s*[+\-*/]', alpha_code):  # Double operators
                return False

            # Disallow numeric literal immediately followed by '(' (missing operator)
            if re.search(r'(^|[^a-zA-Z0-9_])\-?\d+(?:\.\d+)?\s*\(', alpha_code):
                # Allow cases like power(2, x) which are proper, the pattern above
                # specifically targets a bare number followed by '(' not as an argument list
                # But to be safe, flag and let the cleaner fix it upstream
                return False
            
            return True
            
        except Exception:
            return False

    def _extract_alpha_from_response(self, text: str) -> str:
        """Extract alpha code from Gemini's response, handling various formats"""
        # Look for "ALPHA:" marker (like simple_agent.py)
        for line in text.split('\n'):
            if line.startswith('ALPHA:'):
                alpha_code = line.split(':', 1)[1].strip()
                return self._clean_alpha_code(alpha_code)
        
        # Prefer regex extraction of explicit rank(...) components, join them with '*'
        ranks = re.findall(r'rank\([^\n]+?\)', text)
        if ranks:
            expr = ' * '.join(ranks[:3])  # limit to 3 components
            return self._clean_alpha_code(expr)

        # Fallback: look for any line that contains FASTEXPR operators
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if any(op in line for op in ['rank(', 'ts_', 'group_', 'divide(', 'multiply(']):
                # Strip leading narrative before operator
                first_op_idx = -1
                for marker in ['rank(', 'ts_', 'group_', 'divide(', 'multiply(', 'add(', 'subtract(']:
                    idx = line.find(marker)
                    if idx != -1:
                        first_op_idx = idx if first_op_idx == -1 else min(first_op_idx, idx)
                if first_op_idx > 0:
                    line = line[first_op_idx:]
                return self._clean_alpha_code(line)
        
        # Last resort: return the first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                return self._clean_alpha_code(line)
        
        return ""

    def _violates_forbidden_subexpressions(self, alpha_code: str) -> Optional[str]:
        """Return the first forbidden pattern that matches the alpha, if any."""
        for pat in getattr(self, 'forbidden_subexpression_patterns', []):
            try:
                if re.search(pat, alpha_code):
                    return pat
            except Exception:
                # Ignore bad regex entries
                continue
        return None

    def _has_ts_regression(self, alpha_code: str) -> bool:
        """Check if alpha_code contains AT LEAST TWO ts_regression function calls with both lag and rettype keywords."""
        # Find all ts_regression function calls
        matches = re.finditer(r'\bts_regression\s*\([^)]+\)', alpha_code, re.IGNORECASE)
        valid_count = 0
        for match in matches:
            func_call = match.group(0)
            # Check that both lag= and rettype= keywords are present
            has_lag = bool(re.search(r'\blag\s*=', func_call, re.IGNORECASE))
            has_rettype = bool(re.search(r'\brettype\s*=', func_call, re.IGNORECASE))
            if has_lag and has_rettype:
                valid_count += 1
        return valid_count >= 2

    def _extract_reasoning_from_response(self, text: str) -> str:
        """Extract the REASONING section from model response if present."""
        try:
            lines = text.split('\n')
            reasoning_lines: List[str] = []
            capturing = False
            for raw in lines:
                line = raw.strip()
                if not capturing and line.startswith('REASONING:'):
                    reasoning_lines.append(line.split(':', 1)[1].strip())
                    capturing = True
                    continue
                if capturing:
                    if line.startswith('ALPHA:') or line.startswith('PARAMETERS_JSON:'):
                        break
                    if line == '':
                        break
                    reasoning_lines.append(line)
            return ' '.join(reasoning_lines).strip()
        except Exception:
            return ""

    def _extract_parameters_from_response(self, text: str) -> Dict[str, Any]:
        """Extract testing parameters from the model response.
        Expects a line starting with 'PARAMETERS_JSON:' followed by a JSON object.
        Falls back to sensible defaults if missing or invalid.
        """
        # Allowed options (expandable)
        allowed_universes = ["TOP3000"]
        allowed_regions = ["USA"]
        allowed_neuts = ["SUBINDUSTRY"]

        def random_defaults() -> Dict[str, Any]:
            return {
                'universe': 'TOP3000',  # fixed universe for this agent
                'region': 'USA',
                'delay': 1,
                'decay': 0,
                'neutralization': 'SUBINDUSTRY',
                'truncation': 0.1,
                'pasteurization': 'ON',
                'nanHandling': 'ON',
            }

        try:
            # Robustly extract PARAMETERS_JSON allowing inline or multi-line JSON
            params = None
            if 'PARAMETERS_JSON:' in text:
                after = text.split('PARAMETERS_JSON:', 1)[1].strip()
                # If inline JSON on same line
                if after.startswith('{'):
                    # Try to capture a balanced JSON object
                    brace_count = 0
                    collected = []
                    for ch in after:
                        collected.append(ch)
                        if ch == '{':
                            brace_count += 1
                        elif ch == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                break
                    candidate = ''.join(collected).strip('` ').strip()
                    try:
                        params = json.loads(candidate)
                    except Exception:
                        params = None
                # Else, try to find JSON on next lines
                if params is None:
                    lines = after.splitlines()
                    buf = []
                    seen_open = False
                    brace_count = 0
                    for ln in lines:
                        for ch in ln:
                            if ch == '{':
                                seen_open = True
                                brace_count += 1
                            if seen_open:
                                buf.append(ch)
                            if ch == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    break
                        if seen_open and brace_count == 0:
                            break
                    if buf:
                        candidate = ''.join(buf).strip('` ').strip()
                        try:
                            params = json.loads(candidate)
                        except Exception:
                            params = None
            if params is not None:
                    out = random_defaults()
                    allowed_param_keys = {
                        'universe', 'region', 'delay', 'decay',
                        'neutralization', 'truncation', 'pasteurization', 'nanHandling'
                    }
                    out.update({k: params[k] for k in params if k in allowed_param_keys})
                    # Force our desired hard settings regardless of input
                    out['neutralization'] = 'SUBINDUSTRY'
                    out['pasteurization'] = 'ON'
                    out['nanHandling'] = 'ON'
                    # Force delay to valid choice (1)
                    out['delay'] = 1
                    # Clamp decay to [0,10], default 0
                    try:
                        out['decay'] = int(out.get('decay', 0))
                    except Exception:
                        # Handle ranges like "0-10" or invalid strings
                        out['decay'] = 0
                    out['decay'] = max(0, min(10, out['decay']))
                    out['truncation'] = float(out.get('truncation', 0.08))
                    # Validate against allowed sets
                    if out.get('universe') not in allowed_universes:
                        out['universe'] = 'TOP3000'
                    if out.get('region') not in allowed_regions:
                        out['region'] = 'USA'
                    out['neutralization'] = 'SUBINDUSTRY'
                    return out
        except Exception:
            pass
        # If PARAMETERS_JSON missing or invalid, return randomized defaults
        return random_defaults()

    async def _generate_with_retries(self, prompt: str, max_retries: int = 5) -> str:
        """Call Gemini with retries and safe text extraction to handle transient/empty responses"""
        backoff_seconds = 2
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.model.generate_content_async(prompt)
                # Prefer direct text if available
                try:
                    if hasattr(response, 'text') and response.text:
                        text = response.text
                        if text and text.strip():
                            # Console log full Gemini output
                            try:
                                self.logger.info(f"GEMINI OUTPUT (attempt {attempt}/{max_retries}) - length {len(text)}:\n{text}")
                                print(f"\n--- GEMINI OUTPUT (attempt {attempt}/{max_retries}) ---\n{text}\n--- END GEMINI OUTPUT ---\n")
                            except Exception:
                                pass
                            return text
                except Exception:
                    pass

                # Try candidates/parts if SDK shape differs
                try:
                    candidates = getattr(response, 'candidates', None)
                    if candidates:
                        for cand in candidates:
                            # cand.content.parts may be a list of dicts with 'text'
                            content = getattr(cand, 'content', None)
                            parts = None
                            if content is not None:
                                parts = getattr(content, 'parts', None)
                                if parts is None and isinstance(content, dict):
                                    parts = content.get('parts')
                            if parts:
                                texts = []
                                for p in parts:
                                    # part may be dict with 'text' or object with 'text'
                                    t = None
                                    if isinstance(p, dict):
                                        t = p.get('text')
                                    else:
                                        t = getattr(p, 'text', None)
                                    if t:
                                        texts.append(t)
                                combined = "\n".join(texts).strip()
                                if combined:
                                    # Console log full Gemini output (combined from parts)
                                    try:
                                        self.logger.info(f"GEMINI OUTPUT (attempt {attempt}/{max_retries}) [parts] - length {len(combined)}:\n{combined}")
                                        print(f"\n--- GEMINI OUTPUT (attempt {attempt}/{max_retries}) [parts] ---\n{combined}\n--- END GEMINI OUTPUT ---\n")
                                    except Exception:
                                        pass
                                    return combined
                except Exception:
                    pass

                # If we got here, treat as empty and retry
                self.logger.warning(f"Gemini returned empty content on attempt {attempt}/{max_retries}")
            except Exception as e:
                self.logger.warning(f"Gemini call failed on attempt {attempt}/{max_retries}: {e}")

            # Backoff before retrying
            await asyncio.sleep(backoff_seconds)
            # Cap backoff
            backoff_seconds = min(backoff_seconds * 2, 8)

        raise RuntimeError("Gemini generation failed after retries with empty/invalid response")

    def _detect_no_progress(self, history: list, min_attempts: int = 5) -> bool:
        """
        Detect if no progress is being made in alpha generation.
        
        Args:
            history: List of recent alpha results
            min_attempts: Minimum attempts before checking for no progress
            
        Returns:
            True if no progress detected, False otherwise
        """
        if len(history) < min_attempts:
            return False
        
        # Get recent results
        recent_results = history[-min_attempts:]
        fitness_values = [result.get('fitness', 0) for result in recent_results]
        sharpe_values = [result.get('sharpe', 0) for result in recent_results]
        
        # Extremely lenient - only flag if ALL values are identical AND we have significant history
        if len(fitness_values) >= 8:  # Increased from 5 to 8
            # Only flag if ALL fitness values are exactly the same
            if len(set(fitness_values)) == 1:
                return True
        
        if len(sharpe_values) >= 8:  # Increased from 5 to 8
            # Only flag if ALL sharpe values are exactly the same
            if len(set(sharpe_values)) == 1:
                return True
        
        # Skip pattern checking entirely - too restrictive
        return False

    def _tactics_block(self) -> str:
        """Return a concise tactics block filtered by allowed data groups."""
        try:
            if get_tactics_for_groups is None:
                return ""
            allowed = sorted(list(self.allowed_data_groups)) if self.allowed_data_groups else None
            tactics = get_tactics_for_groups(allowed)
            if not tactics:
                return ""
            # Keep the block concise to avoid prompt bloat
            lines: List[str] = []
            for t in tactics[:6]:
                lines.append(f"- {t['title']}: {t['rationale']}")
                lines.append(f"  Skeleton: {t['skeleton']}")
                lines.append(f"  Components: {t['components']}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _check_duplicate_alpha(self, alpha_code: str, history: list) -> bool:
        """
        Check if this alpha code is too similar to recent alphas in history.
        
        Args:
            alpha_code: The alpha code to check
            history: List of recent alpha results
            
        Returns:
            True if this is a duplicate/similar alpha, False otherwise
        """
        if not history or len(history) < 3:
            return False
        
        # ONLY check for exact code match - remove all component-based checking
        for h in history[-3:]:  # Only check last 3
            if h['code'].strip() == alpha_code.strip():
                return True
        
        # Skip all component-based similarity checking - too restrictive
        return False

    def _build_prompt(self, strategy: str, complexity: str, history: list) -> str:
        """Create a rich prompt for Gemini including recent results for learning and full operator / data-field reference"""
        hist_lines = []
        for h in history[-self.history_lookback:]:  # Last N results
            hist_lines.append(f"ID {h['alpha_id']} | fitness {h['fitness']:.2f} | sharpe {h['sharpe']:.2f} | code: {h['code']}")
        hist_block = "\n".join(hist_lines) if hist_lines else "None yet"
        
        # Analyze history to prevent loops and encourage diversity
        diversity_guidance = ""
        if len(history) >= min(12, self.history_lookback):
            # Check for repeated patterns in recent alphas
            window = min(12, self.history_lookback)
            recent_codes = [h['code'] for h in history[-window:]]
            recent_fitness = [h['fitness'] for h in history[-window:]]
            recent_sharpe = [h['sharpe'] for h in history[-window:]]
            
            # Extremely lenient - only warn if ALL values are identical
            if len(set(recent_fitness)) == 1:
                diversity_guidance += "\n💡 SUGGESTION: All recent fitness values are identical. Consider trying different approaches.\n"
            if len(set(recent_sharpe)) == 1:
                diversity_guidance += "\n💡 SUGGESTION: All recent sharpe values are identical. Consider different risk management.\n"
            

        # Build thought history preview for in-context learning (entire run)
        thought_lines = []
        recent_thoughts = self.thought_history
        for t in recent_thoughts:
            trial_idx = t.get('trial_index', '?')
            alpha_preview = (t.get('alpha_code') or '')
            reasoning_preview = (t.get('model_reasoning') or '')
            outcome = t.get('outcome')
            outcome_str = 'pending'
            if isinstance(outcome, dict):
                if outcome.get('success'):
                    fitness = outcome.get('fitness')
                    sharpe = outcome.get('sharpe')
                    if isinstance(fitness, (int, float)) and isinstance(sharpe, (int, float)):
                        outcome_str = f"success F {fitness:.2f} S {sharpe:.2f}"
                    else:
                        outcome_str = "success"
                else:
                    err = (outcome.get('error') or '').strip()
                    if len(err) > 48:
                        err = err[:45] + '...'
                    outcome_str = f"fail {err}" if err else "fail"
            thought_lines.append(f"T{trial_idx}: {outcome_str} | code: {alpha_preview} | why: {reasoning_preview}")
        thought_block = "\n".join(thought_lines) if thought_lines else "None yet"

        tactics_block = self._tactics_block()

        prompt = f"""
You are an expert quantitative researcher creating WorldQuant alpha factors.

TASK: Generate a NOVEL WorldQuant alpha factor. {strategy}

PRIMARY GOAL: Pass WorldQuant checks on the TOP3000 universe. Aim for passed_checks ≥ 7. Additionally target Sharpe ≥ 1.25 and Fitness > 1.0, prioritizing consistent, risk-adjusted returns.

SECONDARY GOAL: After meeting pass criteria (passed_checks ≥ 7, Sharpe ≥ 1.25, Fitness > 1.0), maximize Fitness further without sacrificing robustness.

FITNESS DEFINITION:
Fitness is defined as: Fitness = Sharpe * Sqrt( Abs( Returns ) / Max( Turnover, 0.125 ) )
This is a hybrid metric that indicates overall performance. Higher values indicate better performance.

KEY TRADING STRATEGY:
You can often sacrifice Sharpe to increase Fitness by decreasing the turnover. This can typically be done by:
- Creating alphas with longer-term, more stable signals that trade less frequently
- Using longer lookback periods (60-120 days vs 20-40 days) for time series operators
- Focusing on fundamental data that changes less frequently than price data
- Using decay parameters to reduce excessive trading
- Start with decay = 0; increase only if it clearly reduces turnover without harming Sharpe or pass-rate

        FORMAT: Generate a single-line alpha using FIELDS ONLY from the allowed data groups for this run.
        - Prefer a simple structure such as rank(expression) or ts_rank(expression, n) wrapped in rank(...)
        - Examples of allowed fields for this run: {', '.join(self._get_allowed_field_examples(6))}
        REQUIREMENTS:
        - Use ONLY fields from the allowed groups listed below (do not use any disallowed groups)
        - Keep the expression concise and unit-consistent

RECENT ALPHAS & PERFORMANCE (avoid duplicate ideas):
{hist_block}

THOUGHT HISTORY (all {len(recent_thoughts)} from this run):
{thought_block}

HISTORY-DRIVEN SYNTHESIS:
- From RECENT ALPHAS and THOUGHT HISTORY above, select 2–4 distinct components/sub-expressions that historically improved Sharpe/Fitness in this run and combine them into a composite (e.g., rank(F1) * rank(F2) + rank(F3)).
- Prefer orthogonal components and longer lookbacks when they reduced turnover in prior trials.
- Universe is fixed to TOP3000; do not change.

TACTICS TO CONSIDER (filtered by allowed data groups):
{tactics_block}

{diversity_guidance}

        CRITICAL REQUIREMENTS:
- MANDATORY: All alphas MUST include AT LEAST TWO ts_regression(y, x, d, lag=value, rettype=value) calls with BOTH keyword arguments explicitly specified. Example: ts_regression(fnd6_newa1v1300_gp, fnd6_newa1v1300_ceq, 20, lag=126, rettype=2). OLS regression function parameters: lag=lookback offset (default 0), rettype=return type (0=error, 1=y-int α, 2=slope β, 3=y-estimate, 4=SSE, 5=SST, 6=R², 7=MSE, 8=SEβ, 9=SEα)
- Generate a COMPLETELY NEW alpha that is different from the previous ones
- Use different data fields and combinations than what you see in the history
- Leverage the full range of available operators and data fields
- Create sophisticated but effective alpha expressions
- Use appropriate time series lookback periods (10-252 days)
- PRIORITIZE PASS CRITERIA: Design alphas that should achieve passed_checks ≥ 7, Sharpe ≥ 1.25, and Fitness > 1.0
- PRIORITIZE SHARPE RATIO: Focus on factors that should produce consistent, low-volatility returns
- AVOID REPEATED PATTERNS: Don't use the same data fields and combinations repeatedly
- TRY DIFFERENT APPROACHES: If recent alphas have similar performance, try completely different strategies
        - USE SIMPLE MULTIPLICATIVE PATTERN: Use 2-3 rank(...) components multiplied or added together

AVAILABLE DATA FIELDS & OPERATORS (SCOPED TO {self.data_fields_percentage}% DATA FIELDS, {self.functions_percentage}% FUNCTIONS):
{self._get_scoped_operators_and_fields()}

IMPORTANT GROUP CONSTRAINTS FOR THIS RUN:
- Only these data groups are allowed: {', '.join(sorted(self.allowed_data_groups)) if self.allowed_data_groups else 'ALL'}
- Do NOT use any fields from groups that are not allowed.
- Prefer using fields like: {', '.join(self._get_allowed_field_examples(12))}

IMPORTANT: Use only a subset of the above operators and data fields. Focus on the most relevant ones for your strategy rather than using everything available.

        DO NOT USE THESE COMMON PATTERNS (to avoid correlation):
        - {', '.join(DEFAULT_FORBIDDEN_SUBEXPRESSIONS_HUMAN)}

        EXAMPLE STRUCTURE: Use 1-2 components drawn from the allowed groups for this run. These are illustrative, not mandatory.

        RECOMMENDED SIMPLE PATTERN:
        - rank(ts_rank(close, 60))
        - rank(ts_corr(close, vwap, 60))
        - rank(ts_rank(volume, 30))
        - rank(ts_rank(returns, 90))
        - rank(group_rank(ts_rank(close, 60), subindustry))
        - rank(ts_regression(y_var, x_var, 20, lag=0, rettype=2))  # MANDATORY: must include at least 2 ts_regression calls with both lag= and rettype= keywords
        - rank({self._get_allowed_field_examples(1)[0] if self._get_allowed_field_examples(1) else 'close'})

        CREATIVE COMBINATIONS (allowed groups only):
        - Try momentum and correlation using allowed market data fields
        - Try cross-sectional operators: rank, zscore, normalize, quantile, winsorize, scale, group_rank, group_neutralize
        - Try different time horizons (short 20-30 days vs long 60-180 days)
        - Try sector-relative effects using group_rank or group_neutralize with subindustry only

        

DESIGN GUIDELINES (generalized):
- Do NOT enforce a fixed two-part structure. Examples are illustrative only.
- Most alphas should combine at least 2 distinct factors; 2–3 is typical. Use 4 only if components are simple and stable.
- Diversify operators and lookbacks based on what worked recently; prefer stable, longer horizons when they improved turnover.
- Combine 2–4 historically strong components into composites; maintain unit compatibility.
- Avoid copying examples literally; treat example metrics as placeholders. Pick different fields and lookbacks.
- Always specify lookback periods for time series operators and maintain unit compatibility.
- BUILD COMPLEXITY GRADUALLY and prioritize robustness, Sharpe, and turnover control.

        IMPORTANT RULES:
        - MANDATORY ts_regression FORMAT: When using ts_regression, you MUST include AT LEAST TWO ts_regression calls, each with both keyword arguments: lag= and rettype=. Example format: ts_regression(y_var, x_var, days, lag=0, rettype=2). Do NOT omit the keyword names - both "lag=" and "rettype=" must be explicitly written in each function call.
        - PRICE/VOLUME COMPONENTS: Use price/volume metrics (ts_rank(close), ts_rank(volume), ts_corr(close,vwap), etc.)
        - START SIMPLE: Begin with basic ratios and 2-3 operators maximum
        - PRIORITIZE SHARPE RATIO: Focus on factors that should produce consistent, low-volatility returns
        - BACKFILL OPTIONS DATA (MANDATORY for options-heavy signals): Wrap the final multiplicative rank(...) expression in rank(...) and use ts_backfill(..., 30-90) to handle sparse options coverage
        - Always specify lookback periods for time series operators (e.g., ts_rank(close, 20))
        - Use subindustry grouping for neutralization
        - Create meaningful combinations that capture alpha-generating relationships
        - Use logical operators for conditional expressions when appropriate

        CRITICAL UNIT COMPATIBILITY RULES:
- ts_corr(x, y, d) requires both x and y to have compatible units (both prices OR both volumes)
- group_neutralize() works with any units but should be applied to the final expression
- ts_decay_linear() works with any units but requires a lookback period
- Avoid mixing price and volume units in the same operator unless explicitly allowed
- min(x, y) and max(x, y) cause unit compatibility errors - use rank() instead
- log(x): Can ONLY be used on price and volume data (close, open, high, low, volume), NOT on sharesout, cap, ebitda, adv20, etc.
- log(x): NEVER use log() on adv20, sharesout, cap, or any share-based metrics
- log(x): ONLY use log() on price data: close, open, high, low, vwap
- log(x): NEVER use log() inside ts_corr() - causes unit compatibility errors
- log(x): NEVER use log() on fundamental ratios like divide(cashflow_op, sales) - use rank() instead
- log(x): NEVER use log() on financial ratios like divide(ebitda, debt) - use rank() instead
- log(x): ONLY use log() on pure price data: close, open, high, low, vwap
- log(x): PREFER rank() or ts_rank() over log() for better unit compatibility
- ts_corr(x, y, n): When using ts_corr, ensure both x and y are price data OR both are volume data
- ts_corr(x, y, n): NEVER mix price and volume in ts_corr - use price with price, volume with volume

        SIMPLE PATTERN GUIDELINES:
        - ✅ Use 2–3 rank(...) components multiplied
        - ✅ Prefer orthogonal components (e.g., momentum + stability) rather than redundant signals
        - ✅ Keep units compatible inside each component
        - ✅ Prefer ts_rank/lookbacks for price data and direct rank() for stable fields
        - ✅ If a 2-factor construction scores well but not passing, augment with a third rank(...) component sourced from other strong-performing factors (from recent good trials) to form 3+ multiplied rank(...) terms
        - ✅ When several single factors demonstrate strong performance individually, build a chained composite: rank(F1) * rank(F2) * rank(F3) * rank(F4) (and optionally more), provided unit compatibility and stability are maintained
        - ✅ Also try additive composites when appropriate: rank(F1) + rank(F2) + rank(F3) (+ ...). Ensure terms are scaled compatibly and avoid overfitting
        - ✅ Use symbolic operators '+' and '*' instead of add(...) and multiply(...). Example: rank(F1) * rank(F2) + rank(F3)

UNIT COMPATIBILITY EXAMPLES:
CORRECT:
- ts_corr(close, vwap, 30) - both price data
- ts_corr(volume, adv20, 30) - both volume data
- ts_corr(close, open, 30) - both price data
- ts_rank(close, 30) - price data
- ts_rank(vwap, 40) - price data
- rank(close) - price data
- log(close) - pure price data
- log(vwap) - pure price data
- rank(log(close)) - log of pure price data
- rank(returns) - instead of min(0, returns)
- rank(close) - instead of max(0, close)
- rank(historical_volatility_30) - volatility data
- rank(implied_volatility_mean_60) - implied volatility data
- rank(implied_volatility_mean_skew_30) - volatility skew data
- rank(implied_volatility_call_30) - call option volatility data
- rank(implied_volatility_put_30) - put option volatility data
- rank(parkinson_volatility_60) - Parkinson volatility data
- rank(implied_volatility_mean_30) - implied volatility mean data
- rank(implied_volatility_mean_skew_60) - volatility skew data
- rank(historical_volatility_90) - longer-term historical volatility data
- rank(implied_volatility_call_60) - call option volatility data
- rank(implied_volatility_put_60) - put option volatility data
- rank(parkinson_volatility_90) - Parkinson volatility data

INCORRECT:
- log(adv20) - adv20 is share data, NOT price data
- log(sharesout) - sharesout is share data, NOT price data
- log(cap) - cap is share*price data, NOT pure price data
- min(0, returns) - unit compatibility error
- max(0, close) - unit compatibility error
- ts_corr(close, volume, 30) - mixing price and volume
         RESPONSE FORMAT:
ALPHA: [single line of simple WorldQuant alpha code]
REASONING: [brief explanation of why this alpha should work]
        
NOTE: Start with decay=0 by default. Increase only if it clearly improves turnover without harming Sharpe or pass-rate.
PARAMETERS_JSON: {{"universe": "TOP3000", "region": "USA", "delay": 1, "decay": 0-10, "neutralization": "SUBINDUSTRY", "truncation": 0.02-0.20, "pasteurization": "ON|OFF", "nanHandling": "OFF|ON"}}
IMPORTANT: Universe is fixed to TOP3000. Use the run’s history to combine the strongest components into a composite.
""".strip()
        return prompt

    async def generate_enhanced_alpha(self, strategy_description: str, complexity_level: str = "Expert", history: Optional[list] = None, trial_index: Optional[int] = None) -> Dict[str, Any]:
        """Generate an enhanced alpha using Gemini with learning from history"""
        try:
            if history is None:
                history = []
            
            # Check for no progress in recent attempts - extremely lenient
            if self._detect_no_progress(history) and len(history) >= 15:  # Increased from 10 to 15
                # Modify strategy to force different approach - very gentle
                strategy_description += " [OPTIONAL: Consider trying different data field combinations.]"
            
            # Build prompt with learning context
            prompt = self._build_prompt(strategy_description, complexity_level, history)
            
            # Persist Gemini prompt for debugging
            try:
                os.makedirs('logs', exist_ok=True)
                with open(os.path.join('logs', 'gemini_prompts.txt'), 'a', encoding='utf-8') as f:
                    f.write(datetime.now().isoformat() + '\n')
                    f.write(prompt + '\n\n-----\n\n')
            except Exception as prompt_log_err:
                self.logger.warning(f"Failed logging Gemini prompt: {prompt_log_err}")
            
            # Generate response from Gemini with retries and safe extraction
            response_text = await self._generate_with_retries(prompt)
            
            # Extract and clean alpha code
            alpha_code = self._extract_alpha_from_response(response_text)
            model_reasoning = self._extract_reasoning_from_response(response_text)
            
            # Normalize/fix operator names BEFORE any validation checks to avoid false rejections
            # (e.g., map ts_arg_rank -> ts_rank)
            alpha_code = self._fix_operator_names(self._clean_alpha_code(alpha_code))

            # If extraction/normalization failed, bail out early
            if not alpha_code:
                return {
                    'success': False,
                    'error': 'Failed to extract valid alpha code from response',
                    'raw_response': response_text
                }

            # Enforce data-field group inclusion based on configuration (on normalized code)
            forbidden = []
            for pat in self._get_forbidden_field_patterns():
                if re.search(pat, alpha_code):
                    forbidden.append(pat)
            if forbidden:
                return {
                    'success': False,
                    'error': 'Alpha uses disallowed data fields for this run',
                    'raw_response': response_text
                }
            # Enforce forbidden subexpressions to reduce correlation
            violated = self._violates_forbidden_subexpressions(alpha_code)
            if violated:
                return {
                    'success': False,
                    'error': f"Alpha contains forbidden subexpression pattern: {violated}",
                    'raw_response': response_text
                }
            
            # Enforce mandatory ts_regression in all alphas
            if not self._has_ts_regression(alpha_code):
                return {
                    'success': False,
                    'error': 'Alpha must include AT LEAST TWO ts_regression functions with both lag= and rettype= keywords',
                    'raw_response': response_text
                }
            
            # Quick hallucination filter: reject if unknown functions present (after normalization)
            unknown_funcs = self._unknown_function_names(alpha_code)
            if unknown_funcs:
                return {
                    'success': False,
                    'error': f"Unknown functions detected: {', '.join(sorted(unknown_funcs))}",
                    'raw_response': response_text
                }
            
            # Extract testing parameters or use defaults
            parameters = self._extract_parameters_from_response(response_text)
            
            # Check for duplicate alpha after cleaning - extremely lenient
            if self._check_duplicate_alpha(alpha_code, history):
                return {
                    'success': False,
                    'error': 'Generated alpha is too similar to recent alphas. Please try a different approach.',
                    'raw_response': response_text
                }
            
            # Append a running per-run learning log entry
            try:
                os.makedirs('logs', exist_ok=True)
                with open(os.path.join('logs', 'run_learning_log.txt'), 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps({
                        'ts': datetime.now().isoformat(),
                        'strategy': strategy_description,
                        'trial_index': trial_index,
                        'alpha_code': alpha_code,
                        'parameters': parameters,
                        'model_reasoning': model_reasoning,
                        'history_size': len(history),
                        'scope': {
                            'data_fields_percentage': self.data_fields_percentage,
                            'functions_percentage': self.functions_percentage
                        },
                        'phase': 'generated'
                    }) + "\n")
            except Exception as log_err:
                self.logger.warning(f"Failed to append run learning log: {log_err}")

            # Add to in-run thought history; outcome to be updated later
            self.thought_history.append({
                'ts': datetime.now().isoformat(),
                'trial_index': trial_index,
                'alpha_code': alpha_code,
                'parameters': parameters,
                'model_reasoning': model_reasoning,
                'outcome': None
            })

            return {
                'success': True,
                'alpha_code': alpha_code,
                'reasoning': f"Generated {complexity_level} alpha using Gemini with learning from {len(history)} previous results",
                'model_reasoning': model_reasoning,
                'complexity_used': complexity_level,
                'raw_response': response_text,
                'data_fields_percentage': self.data_fields_percentage,
                'functions_percentage': self.functions_percentage,
                'parameters': parameters,
                'trial_index': trial_index
            }
            
        except Exception as e:
            self.logger.error(f"Error generating alpha: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_scope(self, data_fields_percentage: float = None, functions_percentage: float = None):
        """Update the scope percentages for data fields and functions"""
        if data_fields_percentage is not None:
            self.data_fields_percentage = max(10.0, min(100.0, data_fields_percentage))
        else:
            self.data_fields_percentage = 20.0  # Default to 20%
        if functions_percentage is not None:
            self.functions_percentage = max(10.0, min(100.0, functions_percentage))
        else:
            self.functions_percentage = 20.0  # Default to 20%
        # Reset cached scoped set so a new one is sampled next time
        self._scoped_operators_and_fields_text = None
        self.logger.info(f"Scope updated: {self.data_fields_percentage}% data fields, {self.functions_percentage}% functions")

    def record_trial_outcome(self, trial_index: Optional[int], outcome: Dict[str, Any], alpha_code: Optional[str] = None, model_reasoning: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Record outcome for a trial into the thought history and logs.
        If an entry for this trial does not exist yet, create one.
        """
        try:
            entry = None
            for t in reversed(self.thought_history):
                if t.get('trial_index') == trial_index:
                    entry = t
                    break
            if entry is None:
                entry = {
                    'ts': datetime.now().isoformat(),
                    'trial_index': trial_index,
                    'alpha_code': alpha_code,
                    'parameters': parameters or {},
                    'model_reasoning': model_reasoning or '',
                    'outcome': outcome
                }
                self.thought_history.append(entry)
            else:
                entry['outcome'] = outcome
                if alpha_code is not None:
                    entry['alpha_code'] = alpha_code
                if model_reasoning is not None:
                    entry['model_reasoning'] = model_reasoning
                if parameters is not None:
                    entry['parameters'] = parameters

            try:
                os.makedirs('logs', exist_ok=True)
                with open(os.path.join('logs', 'run_learning_log.txt'), 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps({
                        'ts': datetime.now().isoformat(),
                        'trial_index': trial_index,
                        'alpha_code': entry.get('alpha_code'),
                        'parameters': entry.get('parameters'),
                        'model_reasoning': entry.get('model_reasoning'),
                        'outcome': outcome,
                        'scope': {
                            'data_fields_percentage': self.data_fields_percentage,
                            'functions_percentage': self.functions_percentage
                        },
                        'phase': 'outcome'
                    }) + "\n")
            except Exception as log_err:
                self.logger.warning(f"Failed to append outcome to learning log: {log_err}")
        except Exception as e:
            self.logger.warning(f"Failed to record trial outcome: {e}")

    # Runtime update method is kept for completeness but not required if using ALLOWED_DATA_GROUPS
    def update_allowed_data_groups(self, groups: Optional[List[str]]):
        if not groups:
            self.allowed_data_groups = None
        else:
            self.allowed_data_groups = set(s.strip().upper().rstrip(':') for s in groups)
        self._scoped_operators_and_fields_text = None
    
    def get_current_scope(self) -> Dict[str, Any]:
        """Get current scope information"""
        return {
            'data_fields_percentage': self.data_fields_percentage,
            'functions_percentage': self.functions_percentage,
            'allowed_data_groups': sorted(list(self.allowed_data_groups)) if self.allowed_data_groups else 'ALL'
        }
    
    def _get_scoped_operators_and_fields(self) -> str:
        """Get a scoped subset of operators and fields based on current scope percentages"""
        # Reuse cached subset for entire agent run
        if getattr(self, '_scoped_operators_and_fields_text', None):
            return self._scoped_operators_and_fields_text

        # Parse the full OPERATORS_AND_FIELDS to extract sections
        full_text = OPERATORS_AND_FIELDS
        
        # Find the start of each section
        operators_start = full_text.find('<OPERATORS>')
        data_fields_start = full_text.find('<DATA FIELDS>')
        
        if operators_start == -1 or data_fields_start == -1:
            # Fallback to full text if sections not found
            return full_text
        
        # Extract operators section (from <OPERATORS> to <DATA FIELDS>)
        operators_section = full_text[operators_start:data_fields_start]
        # Extract data fields section (from <DATA FIELDS> to end)
        data_fields_section = full_text[data_fields_start:]
        
        # Parse operators section
        operator_lines = []
        for line in operators_section.split('\n'):
            line = line.strip()
            if line and not line.startswith('<OPERATORS>') and not line.startswith('Arithmetic') and not line.startswith('<Logical') and not line.startswith('<Time-Series') and not line.startswith('<Cross-Sectional') and not line.startswith('<Vector') and not line.startswith('<Transformational') and not line.startswith('<Group'):
                if '–' in line or '(' in line:  # Only include actual function definitions
                    operator_lines.append(line)
        
        # Parse data fields section with group filtering
        def detect_group(header_line: str) -> Optional[str]:
            s = header_line.strip().upper().rstrip(':')
            if s.startswith('MARKET DATA'):
                return 'MARKET DATA'
            if s.startswith('FSCORE METRICS'):
                return 'FSCORE METRICS'
            if s.startswith('FINANCIAL STATEMENT DATA'):
                return 'FINANCIAL STATEMENT DATA'
            if s.startswith('ADDITIONAL FINANCIAL DATA'):
                return 'ADDITIONAL FINANCIAL DATA'
            if s.startswith('ANALYST ESTIMATES DATA'):
                return 'ANALYST ESTIMATES DATA'
            if s.startswith('SENTIMENT DATA'):
                return 'SENTIMENT DATA'
            if s.startswith('OPTIONS & VOLATILITY DATA'):
                return 'OPTIONS & VOLATILITY DATA'
            if s.startswith('GROUPING FIELDS'):
                return 'GROUPING FIELDS'
            if s.startswith('SYMBOL FIELDS'):
                return 'SYMBOL FIELDS'
            return None

        data_field_lines = []
        current_group = None
        for raw_line in data_fields_section.split('\n'):
            line = raw_line.strip()
            # Track current group based on headers
            grp = detect_group(line)
            if grp is not None:
                current_group = grp
                continue
            # Skip non-definition and meta lines
            if not line or line.startswith('<DATA FIELDS>') or line.startswith('TIME SERIES OPERATORS:') or line.startswith('CROSS-SECTIONAL OPERATORS:') or line.startswith('MATH OPERATORS:') or line.startswith('NEUTRALIZATION TECHNIQUES:') or line.startswith('UNIT COMPATIBILITY EXAMPLES:') or line.startswith('CORRECT:') or line.startswith('INCORRECT:') or line.startswith('SOPHISTICATED PATTERN EXAMPLES:') or line.startswith('SIMPLE PATTERN GUIDELINES:') or line.startswith('CREATIVE COMBINATIONS:') or line.startswith('DESIGN GUIDELINES:') or line.startswith('IMPORTANT RULES:') or line.startswith('CRITICAL UNIT COMPATIBILITY RULES:') or line.startswith('RESPONSE FORMAT:'):
                continue
            # Only include actual field definitions (contain ' - ')
            if ' - ' in line:
                # Apply group filter if configured
                if self.allowed_data_groups is None or (current_group in self.allowed_data_groups):
                    data_field_lines.append(line)
        
        # Calculate how many items to include from each section based on percentages
        num_operators = len(operator_lines)
        num_data_fields = len(data_field_lines)
        
        operators_to_include = max(1, int(num_operators * self.functions_percentage / 100.0))
        data_fields_to_include = max(1, int(num_data_fields * self.data_fields_percentage / 100.0))
        
        # Randomly sample a different subset on each call
        sampled_operators = random.sample(operator_lines, k=min(operators_to_include, num_operators))
        sampled_data_fields = random.sample(data_field_lines, k=min(data_fields_to_include, num_data_fields))
        
        # Build scoped version and cache it for reuse
        scoped_text = f"<OPERATORS> (SCOPED TO {self.functions_percentage}% - {len(sampled_operators)}/{num_operators} functions)\n"
        scoped_text += "\n".join(sampled_operators)
        
        scoped_text += f"\n\n<DATA FIELDS> (SCOPED TO {self.data_fields_percentage}% - {len(sampled_data_fields)}/{num_data_fields} fields)\n"
        scoped_text += "\n".join(sampled_data_fields)

        self._scoped_operators_and_fields_text = scoped_text
        return self._scoped_operators_and_fields_text

# VALID DATA FIELDS SECTION - Comprehensive list for Gemini reference
"""
VALID DATA FIELDS FOR WORLDQUANT FASTEXPR:

MARKET DATA (Price & Volume):
- close, open, high, low: Current day's OHLC prices
- volume: Trading volume
- vwap: Volume-weighted average price
- adv20: 20-day average daily volume
- returns: Daily returns
- market_cap: Market capitalization
- enterprise_value: Enterprise value

GROUPING FIELDS (for neutralization):
- subindustry, industry, sector: Industry classification
- country, region: Geographic classification
- market_cap_group: Market cap size groups

SYMBOL FIELDS:
- symbol: Stock ticker symbol
- name: Company name

FSCORE METRICS:
- fscore_quality: Quality score (0-9)
- fscore_value: Value score (0-9)
- fscore_growth: Growth score (0-9)
- fscore_financial: Financial strength score (0-9)

FINANCIAL STATEMENT DATA:
- assets: Total assets
- sales: Revenue/sales
- cashflow_op: Operating cash flow
- ebitda: Earnings before interest, taxes, depreciation, amortization
- return_equity: Return on equity
- return_assets: Return on assets
- debt_equity: Debt-to-equity ratio
- current_ratio: Current ratio
- quick_ratio: Quick ratio
- gross_margin: Gross profit margin
- operating_margin: Operating profit margin
- net_margin: Net profit margin

 - Extended fields:
 - assets - Assets - Total
 - liabilities - Liabilities - Total
 - operating_income - Operating Income After Depreciation - Quarterly
 - sales - Sales/Turnover (Net)
 - enterprise_value - Enterprise Value
 - capex - Capital Expenditures
 - debt - Debt
 - equity - Common/Ordinary Equity - Total
 - ebit - Earnings Before Interest and Taxes
 - ebitda - Earnings Before Interest
 - eps - Earnings Per Share (Basic) - Including Extraordinary Items
 - debt_lt - Long-Term Debt - Total
 - assets_curr - Current Assets - Total
 - goodwill - Goodwill (net)
 - income - Net Income
 - cash - Cash
 - revenue - Revenue - Total
 - cashflow_op - Operating Activities - Net Cash Flow
 - cogs - Cost of Goods Sold
 - bookvalue_ps - Book Value Per Share
 - ppent - Property Plant and Equipment - Total (Net)
 - operating_expense - Operating Expense - Total
 - debt_st - Debt in Current Liabilities
 - cashflow - Cashflow (Annual)
 - inventory - Inventories - Total
 - liabilities_curr - Current Liabilities - Total
 - cash_st - Cash and Short-Term Investments
 - receivable - Receivables - Total
 - sga_expense - Selling, General and Administrative Expenses
 - fnd6_fopo - Funds from Operations - Other
 - return_equity - Return on Equity
 - retained_earnings - Retained Earnings
 - income_tax - Income Taxes - Total
 - pretax_income - Pretax Income
 - cashflow_fin - Financing Activities - Net Cash Flow
 - income_beforeextra - Income Before Extraordinary Items
 - sales_growth - Growth in Sales (Quarterly)
 - current_ratio - Current Ratio
 - return_assets - Return on Assets
 - fnd6_drlt - Deferred Revenue - Long-term
 - inventory_turnover - Inventory Turnover
 - sales_ps - Sales per Share (Quarterly)
 - invested_capital - Invested Capital - Total - Quarterly
 - cashflow_dividends - Cash Dividends (Cash Flow)
 - fnd6_drc - Deferred Revenue - Current
 - fnd6_ivaco - Investing Activities - Other
 - working_capital - Working Capital (Balance Sheet)
 - employee - Employees
 - cashflow_invst - Investing Activities - Net Cash Flow
 - depre_amort - Depreciation and Amortization - Total
 - fnd6_mrcta - Thereafter Portion of Leases
 - fnd6_newa2v1300_ppent - Property, Plant and Equipment - Total (Net)
 - fnd6_recd - Receivables - Estimated Doubtful
 - fnd6_fatl - Property, Plant, and Equipment - Leases at Cost
 - fnd6_rea - Retained Earnings - Restatement
 - fnd6_acdo - Current Assets of Discontinued Operations
 - fnd6_ciother - Comp. Inc. - Other Adj.
 - fnd6_acodo - Other Current Assets Excl Discontinued Operations
 - interest_expense - Interest and Related Expense - Total
 - rd_expense - Research And Development (Quarterly)
 - fnd6_newa2v1300_rdipeps - In Process R&D Expense Basic EPS Effect
 - fnd6_adesinda_curcd - ISO Currency Code - Company Annual Market
 - fnd6_xrent - Rental Expense
 - fnd6_acox - Current Assets - Other - Sundry
 - fnd6_ci - Comprehensive Income - Total
 - fnd6_lcox - Current Liabilities - Other - Sundry
 - fnd6_ein - Employer Identification Number code for the company
 - fnd6_ceql - Common Equity - Liquidation Value
 - fnd6_intc - Interest Capitalized
 - fnd6_newa2v1300_xsga - Selling, General and Administrative Expense
 - fnd6_zipcode - ZIP code related to the company
 - fnd6_newa2v1300_oibdp - Operating Income Before Depreciation
 - fnd6_capxv - Capital Expend Property, Plant and Equipment Schd V
 - fnd6_newa1v1300_gp - Gross Profit (Loss)
 - fnd6_newqv1300_ancq - Non-Current Assets - Total
 - fnd6_newqv1300_drltq - Deferred Revenue - Long-term
 - fnd6_txo - Income Taxes - Other
 - fnd6_itci - Investment Tax Credit (Income Account)
 - fnd6_lcoxdr - Current Liabilities - Other - Excluding Deferred Revenue
 - fnd6_exre - Exchange Rate Effect
 - fnd6_newqv1300_lltq - Long-Term Liabilities (Total)
 - fnd6_ch - Cash
 - fnd6_cik - nonimportant technical code
 - fnd6_cicurr - Comp Inc - Currency Trans Adj
 - fnd6_state - integer for identifying the state of the company
 - fnd6_am - Amortization of Intangibles
 - fnd6_newqv1300_seqoq - Other Stockholders' Equity Adjustments
 - fnd6_mrc2 - Rental Commitments - Minimum - 2nd Year
 - fnd6_weburl - WEB URL code for the company
 - fnd6_city - the city where a company's corporate headquarters or home office is located
 - fnd6_teq - Stockholders' Equity - Total
 - fnd6_pifo - Pretax Income - Foreign
 - fnd6_dlto - Debt - Long-Term - Other
 - fnd6_ds - Debt - Subordinated
 - fnd6_currencya_curcd - ISO Currency Code - Company Annual Market
 - fnd6_newqeventv110_drltq - Deferred Revenue - Long-term
 - fnd6_incorp - Incorporated
 - fnd6_newqv1300_drcq - Deferred Revenue - Current
 - fnd6_newa2v1300_prsho - Redeem Pfd Shares Outs (000)
 - fnd6_ppents - Property, Plant & Equipment

ADDITIONAL FINANCIAL DATA:
- book_value: Book value per share
- earnings_per_share: EPS
- price_book: Price-to-book ratio
- price_earnings: P/E ratio
- price_sales: Price-to-sales ratio
- dividend_yield: Dividend yield
- beta: Stock beta relative to market

ANALYST ESTIMATES DATA:
- anl4_adjusted_netincome_ft - Adjusted net income - forecast type (revision/new/...) — Matrix — Coverage 87%
- anl4_ptp_flag - Pretax income - forecast type (revision/new/...) — Matrix — Coverage 83%
- anl4_ebit_value - Earnings before interest and taxes - announced financial value — Matrix — Coverage 49%
- anl4_ebitda_value - Earnings before interest, taxes, depreciation and amortization - announced financial value — Matrix — Coverage 45%
- anl4_ptpr_number - Reported Pretax Income - number of estimations — Matrix — Coverage 41%
- anl4_epsr_flag - GAAP Earnings - estimation type (revision/new/...), per share — Matrix — Coverage 87%
- anl4_flag_erbfintax - Earnings before interest and taxes - forecast type (revision/new/...) — Matrix — Coverage 82%
- anl4_capex_high - Capital Expenditures - The highest estimation — Matrix — Coverage 62%
- anl4_netprofit_flag - Net profit - forecast type (revision/new/...) — Matrix — Coverage 88%
- anl4_bvps_flag - Book value per share - forecast type (revision/new/...) — Matrix — Coverage 82%
- anl4_capex_low - Capital Expenditures - The lowest estimation — Matrix — Coverage 62%
- anl4_gric_flag - Gross income - forecast type (revision/new/...) — Matrix — Coverage 90%
- anl4_cfo_low - Cash Flow From Operations - The lowest estimation — Matrix — Coverage 62%
- anl4_ebitda_low - Earnings before interest, taxes, depreciation and amortization - The lowest estimation — Matrix — Coverage 58%
- anl4_cfo_value - Cash Flow From Operations - announced financial value — Matrix — Coverage 32%
- anl4_capex_value - Capital Expenditures - announced financial value — Matrix — Coverage 37%
- est_capex - Capital Expenditures - mean of estimations — Matrix — Coverage 64%
- anl4_ptp_number - Pretax Income - number of estimations — Matrix — Coverage 73%
- anl4_eaz2lrec_ratingvalue - Score on the given instrument — Vector — Coverage 53%
- anl4_qfd1_az_eps_number - Earnings per share - number of estimations — Matrix — Coverage 78%
- anl4_ady_pu - The number of upper estimations — Vector — Coverage 72%
- anl4_epsr_low - GAAP Earnings per share - The lowest estimation — Matrix — Coverage 72%
- est_rd_expense - Research and Development Expense - mean of estimations — Matrix — Coverage 31%
- anl4_ebitda_high - Earnings before interest, taxes, depreciation, and amortization - the highest estimation — Matrix — Coverage 58%
- anl4_cff_low - Cash Flow From Financing - The lowest estimation — Matrix — Coverage 59%
- anl4_fsdetailrecv4v104_item - Financial item — Vector — Coverage 80%
- anl4_fcfps_number - Free Cash Flow per Share - number of estimations — Matrix — Coverage 44%
- anl4_totassets_number - Total Assets - number of estimations — Matrix — Coverage 72%
- anl4_netdebt_flag - Net debt - forecast type (revision/new/...) — Matrix — Coverage 86%
- anl4_qf_az_eps_number - Earnings per share - number of estimations — Matrix — Coverage 78%
- anl4_ptp_low - Pretax income - the lowest estimation — Matrix — Coverage 73%
- anl4_adxqfv110_pu - The number of upper estimations — Vector — Coverage 70%
- anl4_fsdtlestmtafv4_item - Financial item — Vector — Coverage 79%
- anl4_tbve_ft - Tangible Book Value per Share - forecast type (revision/new/...) — Matrix — Coverage 93%
- anl4_totassets_flag - Total Assets - forecast type (revision/new/...) — Matrix — Coverage 81%
- anl4_netprofit_number - Net profit - number of estimations — Matrix — Coverage 77%
- anl4_fsguidancebasicqfv4_item - Financial item — Vector — Coverage 32%
- anl4_bac1detaillt_item - Financial item — Vector — Coverage 68%
- anl4_dei3lltv110_item - Financial item — Vector — Coverage 47%
- anl4_cuo1detailafv110_item - Financial item — Vector — Coverage 69%
- est_eps - Earnings per share - mean of estimations — Matrix — Coverage 78%
- anl4_netprofita_low - Adjusted net income - the lowest estimation — Matrix — Coverage 75%
- anl4_median_epsreported - GAAP Earnings per share - median of estimations — Matrix — Coverage 72%
- anl4_ebit_low - Earnings before interest and taxes - The lowest estimation — Matrix — Coverage 71%
- anl4_fsdetailltv4v104_item - Financial item — Vector — Coverage 80%
- anl4_fcf_flag - Free cash flow - forecast type (revision/new/...) — Matrix — Coverage 86%
- anl4_epsr_number - GAAP Earnings per share - number of estimations — Matrix — Coverage 72%
- anl4_ebit_high - Earnings before interest and taxes - The highest estimation — Matrix — Coverage 71%
- anl4_rd_exp_flag - Research and Development Expense - forecast type (revision/new/...) — Matrix — Coverage 98%
- anl4_basicconqfv110_pu - The number of upper estimations — Vector — Coverage 73%
- anl4_ebitda_flag - Earnings before interest, taxes, depreciation and amortization - forecast type (revision/new/...) — Matrix — Coverage 86%
- anl4_fcf_low - Free Cash Flow - The lowest estimation — Matrix — Coverage 55%
- anl4_rd_exp_mean - Research and Development Expense - mean of estimations — Matrix — Coverage 29%
- anl4_capex_std - Capital Expenditures - standard deviation of estimations — Matrix — Coverage 26%
- est_epsr - GAAP Earnings per share - mean of estimations — Matrix — Coverage 74%
- anl4_cfi_low - Cash Flow From Investing - The lowest estimation — Matrix — Coverage 58%
- anl4_dei2lqfv110_item - Financial item — Vector — Coverage 43%
- anl4_fsguidanceqfv4_maxguidance - Max guidance value — Vector — Coverage 31%
- est_sales - Sales - mean of estimations — Matrix — Coverage 75%
- anl4_capex_flag - Capital Expenditures - forecast type (revision/new/...) — Matrix — Coverage 85%
- anl4_ptpr_low - Reported Pretax Income - The Lowest Estimation — Matrix — Coverage 41%
- anl4_rd_exp_low - Research and Development Expense - the lowest estimation — Matrix — Coverage 29%
- anl4_ptp_high - Pretax income - the highest estimation — Matrix — Coverage 73%
- anl4_fsactualafv4_item - Financial item — Vector — Coverage 79%
- anl4_cuo1detailqfv110_item - Financial item — Vector — Coverage 63%
- anl4_guibasicqfv4_est - Estimation value — Vector — Coverage 32%
- anl4_gric_std - Gross income - std of estimations — Matrix — Coverage 41%
- anl4_epsr_high - GAAP Earnings per share - The highest estimation — Matrix — Coverage 72%
- anl4_total_rec - The total number of recommendations — Vector — Coverage 62%
- anl4_bac1detailrec_item - Financial item — Vector — Coverage 71%
- anl4_dts_rspe - Reported Earnings per share - standard deviation of estimations — Matrix — Coverage 48%
- anl4_ptpr_flag - Reported Pretax income - forecast type (revision/new/...) — Matrix — Coverage 90%
- anl4_cfo_median - Cash Flow From Operations - median of estimations — Matrix — Coverage 62%
- anl4_cfo_high - Cash Flow From Operations - The highest value among forecasts — Matrix — Coverage 62%
- anl4_fsguidanceqfv4_minguidance - Min guidance value — Vector — Coverage 31%
- anl4_cff_flag - Cash Flow From Financing Activities - forecast type (revision/new/...) — Matrix — Coverage 82%
- anl4_cfi_value - Cash Flow From Investing - announced financial value — Matrix — Coverage 26%
- anl4_gric_value - Gross Income - announced financial value — Matrix — Coverage 35%
- anl4_ebit_number - Earnings before interest and taxes - number of estimations — Matrix — Coverage 71%
- est_ebit - Earnings before interest and taxes - mean of estimations — Matrix — Coverage 73%
- est_ebitda - Earnings before interest, taxes, depreciation, and amortization - mean of estimations — Matrix — Coverage 60%
- anl4_totassets_low - Total Assets - The lowest estimation — Matrix — Coverage 72%
- anl4_netprofita_number - Adjusted net income - number of estimations — Matrix — Coverage 75%
- est_netprofit - Net profit - mean of estimations — Matrix — Coverage 79%
- anl4_cff_high - Cash Flow From Financing - The highest of forecasted values — Matrix — Coverage 59%
- anl4_ebit_median - Earnings before interest and taxes - median of estimations — Matrix — Coverage 71%
- anl4_fsguidanceqfv4_item - Financial item — Vector — Coverage 31%
- anl4_gric_low - Gross income - The lowest estimation — Matrix — Coverage 60%
- anl4_fcf_value - Free cash flow - announced financial value — Matrix — Coverage 28%
- anl4_epsr_value - GAAP Earnings per share - announced financial value — Matrix — Coverage 57%
- anl4_guiqfv4_est - Estimation value — Vector — Coverage 31%
- anl4_cfo_flag - Cash Flow From Operations - forecast type (revision/new/...) — Matrix — Coverage 83%
- anl4_basicconltv110_numest - The number of forecasts counted in aggregation — Vector — Coverage 70%
- est_cashflow_op - Cash Flow From Operations - mean of estimations — Matrix — Coverage 65%
- anl4_netprofit_value - Net profit - announced financial value — Matrix — Coverage 58%
- anl4_basicconafv110_pu - The number of upper estimations — Vector — Coverage 73%
- anl4_tbvps_number - Tangible Book Value per Share - number of estimations — Matrix — Coverage 33%
- anl4_cfi_high - Cash Flow From Investing - The highest estimation — Matrix — Coverage 58%
- anl4_eaz2lrec_person - Broker Id — Vector — Coverage 55%
- anl4_bvps_low - Book value - the lowest estimation, per share — Matrix — Coverage 65%

SENTIMENT DATA:
- scl12_alltype_buzzvec - sentiment volume — Vector — Matrix — Coverage 95%
- scl12_alltype_sentvec - sentiment — Vector — Matrix — Coverage 95%
- scl12_alltype_typevec - instrument type index — Vector — Matrix — Coverage 95%
- scl12_buzz - relative sentiment volume — Matrix — Coverage 95%
- scl12_sentiment - sentiment — Matrix — Coverage 94%
- snt_buzz - negative relative sentiment volume, fill nan with 0 — Matrix — Coverage 95%
- snt_buzz_bfl - negative relative sentiment volume, fill nan with 1 — Matrix — Coverage 100%
- snt_buzz_ret - negative return of relative sentiment volume — Matrix — Coverage 95%
- snt_value - negative sentiment, fill nan with 0 — Matrix — Coverage 94%
 - snt_social_value - Z score of sentiment — Matrix — Coverage 86%
 - snt_social_volume - Normalized tweet volume — Matrix — Coverage 86%

OPTIONS & VOLATILITY DATA:
- historical_volatility_10, historical_volatility_20, historical_volatility_30, historical_volatility_60, historical_volatility_90, historical_volatility_120, historical_volatility_150, historical_volatility_180: Close-to-close historical volatility over various time periods
- parkinson_volatility_10, parkinson_volatility_20, parkinson_volatility_30, parkinson_volatility_60, parkinson_volatility_90, parkinson_volatility_120, parkinson_volatility_150, parkinson_volatility_180: Parkinson model's historical volatility over various time periods
- implied_volatility_call_10, implied_volatility_call_20, implied_volatility_call_30, implied_volatility_call_60, implied_volatility_call_90, implied_volatility_call_120, implied_volatility_call_150, implied_volatility_call_180, implied_volatility_call_270, implied_volatility_call_360, implied_volatility_call_720, implied_volatility_call_1080: At-the-money option-implied volatility for call options over various time periods
- implied_volatility_put_10, implied_volatility_put_20, implied_volatility_put_30, implied_volatility_put_60, implied_volatility_put_90, implied_volatility_put_120, implied_volatility_put_150, implied_volatility_put_180, implied_volatility_put_270, implied_volatility_put_360, implied_volatility_put_720, implied_volatility_put_1080: At-the-money option-implied volatility for put options over various time periods
- implied_volatility_mean_10, implied_volatility_mean_20, implied_volatility_mean_30, implied_volatility_mean_60, implied_volatility_mean_90, implied_volatility_mean_120, implied_volatility_mean_150, implied_volatility_mean_180, implied_volatility_mean_270, implied_volatility_mean_360, implied_volatility_mean_720, implied_volatility_mean_1080: At-the-money option-implied volatility mean over various time periods
- implied_volatility_mean_skew_10, implied_volatility_mean_skew_20, implied_volatility_mean_skew_30, implied_volatility_mean_skew_60, implied_volatility_mean_skew_90, implied_volatility_mean_skew_120, implied_volatility_mean_skew_150, implied_volatility_mean_skew_180, implied_volatility_mean_skew_270, implied_volatility_mean_skew_360, implied_volatility_mean_skew_720, implied_volatility_mean_skew_1080: At-the-money option-implied volatility mean skew over various time periods

TIME SERIES OPERATORS:
- ts_rank(expr, n): Rank over n periods
- ts_delta(expr, n): Change over n periods
- ts_mean(expr, n): Mean over n periods
- ts_sum(expr, n): Sum over n periods
- ts_corr(expr1, expr2, n): Correlation over n periods
- ts_std_dev(expr, n): Standard deviation over n periods
- ts_zscore(expr, n): Z-score over n periods
- ts_decay_linear(expr, n): Linear decay over n periods

CROSS-SECTIONAL OPERATORS:
- rank(expr): Cross-sectional rank
- zscore(expr): Cross-sectional z-score
- normalize(expr): Normalize to [-1, 1]
- quantile(expr, q): Quantile function
- scale(expr): Scale to unit variance
- winsorize(expr, p): Winsorize outliers
- group_rank(expr, group): Rank within groups
- group_neutralize(expr, group): Neutralize within groups

MATH OPERATORS:
- add(expr1, expr2): Addition
- subtract(expr1, expr2): Subtraction
- multiply(expr1, expr2): Multiplication
- divide(expr1, expr2): Division
- power(expr, n): Exponentiation
- log(expr): Natural logarithm
- abs(expr): Absolute value
- sign(expr): Sign function
- max(expr1, expr2): Maximum
- min(expr1, expr2): Minimum

NEUTRALIZATION TECHNIQUES:
- group_rank(expr, subindustry): Rank within subindustry groups
- group_rank(expr, industry): Rank within industry groups
- group_rank(expr, sector): Rank within sector groups
- group_neutralize(expr, subindustry): Remove subindustry effects
- group_neutralize(expr, industry): Remove industry effects
- group_neutralize(expr, sector): Remove sector effects

COVERAGE NOTES:
- Market data: Available for all US equities
- Financial data: Available for most US equities (coverage varies by field)
- FScore: Available for US equities with sufficient data
- Grouping fields: Available for all US equities
"""

async def test_enhanced_agent(trials: int = 60):
    """Test the enhanced agent with a configurable number of trials
    
    This test follows the same error handling pattern as existing agents:
    - On unit compatibility errors or other failures, continue with next trial
    - Don't try to fix complex errors, just generate another alpha
    - Maintain learning history for pattern diversity
    - Actually test alphas with WorldQuant and insert into database
    """
    agent = EnhancedGeminiAgent(history_lookback=30)
    
    # Initialize WQAlphaTools for actual testing
    try:
        from mcp_server import WQAlphaTools
        wq_tools = WQAlphaTools()
        print("✅ WQAlphaTools initialized for actual alpha testing")
    except Exception as e:
        print(f"❌ Failed to initialize WQAlphaTools: {e}")
        wq_tools = None
    
    # Test with learning history
    history = [
    ]
    
    print(f"🚀 Enhanced Gemini Agent Test - {trials} Trials")
    print("=" * 60)
    print()
    
    successful_trials = 0
    failed_trials = 0
    
    # Let the LLM propose its own strategy each time (no preset strategies)
    for trial in range(1, trials + 1):
        strategy = (
        )
        
        print(f"Trial {trial:2d}/{trials}")
        
        try:
            result = await agent.generate_enhanced_alpha(
                strategy,
                "Expert",
                history,
                trial_index=trial
            )
            
            if result['success']:
                print(f"✅ Generated: {result['alpha_code']}")
                
                # Validate alpha code syntax before testing
                alpha_code = result['alpha_code']
                # Reject if hallucinated functions are present even post-generation
                bad_funcs = agent._unknown_function_names(alpha_code)
                if bad_funcs:
                    failed_trials += 1
                    print(f"❌ Invalid functions detected: {', '.join(sorted(bad_funcs))} - skipping trial")
                    continue
                if not agent._validate_alpha_syntax(alpha_code):
                    failed_trials += 1
                    print(f"❌ Invalid alpha syntax - skipping trial")
                    continue
                
                # Actually test the alpha with WorldQuant if tools available
                if wq_tools:
                    try:
                        print(f"🧪 Testing alpha with WorldQuant...")
                        
                        # Use parameters suggested by Gemini (with safe defaults)
                        parameters = result.get('parameters', {
                            'neutralization': 'SUBINDUSTRY',
                            'decay': 8,
                            'truncation': 0.08,
                            'delay': 1,
                            'universe': 'TOP3000',
                            'region': 'USA'
                        })
                        # Force universe to TOP3000 regardless of model output
                        parameters['universe'] = 'TOP3000'
                        
                        # Test the alpha
                        test_result = wq_tools.test_alpha(result['alpha_code'], parameters)
                        
                        if test_result.get('success'):
                            successful_trials += 1
                            fitness = test_result.get('fitness', 0)
                            sharpe = test_result.get('sharpe', 0)
                            alpha_id = test_result.get('alpha_id', f'trial_{trial}')
                            
                            print(f"✅ WQ Test Success - Fitness: {fitness:.3f}, Sharpe: {sharpe:.3f}, ID: {alpha_id}")
                            agent.record_trial_outcome(trial_index=trial, outcome={
                                'success': True,
                                'fitness': fitness,
                                'sharpe': sharpe,
                                'alpha_id': alpha_id
                            }, alpha_code=result['alpha_code'], model_reasoning=result.get('model_reasoning'), parameters=parameters)
                            
                            # Add to history with actual results
                            history.append({
                                'alpha_id': alpha_id,
                                'fitness': fitness,
                                'sharpe': sharpe,
                                'code': result['alpha_code']
                            })
                            
                        else:
                            failed_trials += 1
                            print(f"❌ WQ Test Failed: {test_result.get('error', 'Unknown error')}")
                            agent.record_trial_outcome(trial_index=trial, outcome={
                                'success': False,
                                'error': test_result.get('error', 'Unknown error')
                            }, alpha_code=result['alpha_code'], model_reasoning=result.get('model_reasoning'), parameters=parameters)
                            # Auto-retry once on WQ failure: regenerate and retest quickly
                            try:
                                retry = await agent.generate_enhanced_alpha(
                                    strategy,
                                    "Expert",
                                    history
                                )
                                if retry.get('success'):
                                    print(f"🔁 Retrying with new alpha: {retry['alpha_code']}")
                                    
                                    # Validate retry alpha code syntax
                                    retry_alpha_code = retry['alpha_code']
                                    if not agent._validate_alpha_syntax(retry_alpha_code):
                                        print(f"❌ Retry alpha has invalid syntax - skipping retry")
                                        continue
                                    
                                    # Use the same parameters for retry to keep evaluation consistent
                                    test_retry = wq_tools.test_alpha(retry_alpha_code, parameters)
                                    if test_retry.get('success'):
                                        successful_trials += 1
                                        fitness = test_retry.get('fitness', 0)
                                        sharpe = test_retry.get('sharpe', 0)
                                        alpha_id = test_retry.get('alpha_id', f'trial_{trial}_retry')
                                        print(f"✅ Retry WQ Test Success - Fitness: {fitness:.3f}, Sharpe: {sharpe:.3f}, ID: {alpha_id}")
                                        history.append({'alpha_id': alpha_id, 'fitness': fitness, 'sharpe': sharpe, 'code': retry['alpha_code']})
                                        agent.record_trial_outcome(trial_index=f"{trial}_retry", outcome={
                                            'success': True,
                                            'fitness': fitness,
                                            'sharpe': sharpe,
                                            'alpha_id': alpha_id
                                        }, alpha_code=retry['alpha_code'], model_reasoning=retry.get('model_reasoning'), parameters=parameters)
                                        if len(history) > 20:
                                            history = history[-20:]
                                    else:
                                        print(f"❌ Retry WQ Test Failed: {test_retry.get('error', 'Unknown error')}")
                                        agent.record_trial_outcome(trial_index=f"{trial}_retry", outcome={
                                            'success': False,
                                            'error': test_retry.get('error', 'Unknown error')
                                        }, alpha_code=retry.get('alpha_code'), model_reasoning=retry.get('model_reasoning'), parameters=parameters)
                                else:
                                    print(f"❌ Retry generation failed: {retry.get('error', 'Unknown error')}")
                                    agent.record_trial_outcome(trial_index=f"{trial}_retry", outcome={
                                        'success': False,
                                        'error': retry.get('error', 'Unknown error')
                                    })
                            except Exception as re:
                                print(f"❌ Retry error: {str(re)}")
                                agent.record_trial_outcome(trial_index=f"{trial}_retry", outcome={
                                    'success': False,
                                    'error': str(re)
                                })
                            
                    except Exception as e:
                        failed_trials += 1
                        print(f"❌ WQ Test Error: {str(e)}")
                        agent.record_trial_outcome(trial_index=trial, outcome={
                            'success': False,
                            'error': str(e)
                        }, alpha_code=result['alpha_code'], model_reasoning=result.get('model_reasoning'), parameters=parameters)
                else:
                    # WQ tools required - fail the trial
                    failed_trials += 1
                    print(f"❌ WQ tools required - alpha cannot be tested")
                    agent.record_trial_outcome(trial_index=trial, outcome={
                        'success': False,
                        'error': 'WQ tools required - alpha cannot be tested'
                    }, alpha_code=result['alpha_code'], model_reasoning=result.get('model_reasoning'))
                
                # Keep history manageable
                if len(history) > 20:
                    history = history[-20:]
                    
            else:
                failed_trials += 1
                print(f"❌ Failed: {result['error']}")
                agent.record_trial_outcome(trial_index=trial, outcome={
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
                # Continue with next trial - don't stop on errors
                
        except Exception as e:
            failed_trials += 1
            print(f"❌ Error: {str(e)}")
            agent.record_trial_outcome(trial_index=trial, outcome={
                'success': False,
                'error': str(e)
            })
            # Continue with next trial - don't stop on exceptions
        
        # Small delay between trials
        await asyncio.sleep(0.5)
        
        # Progress update every 10 trials
        if trial % 10 == 0:
            print(f"\n📊 Progress: {trial}/{trials} trials completed")
            print(f"✅ Successful: {successful_trials}, ❌ Failed: {failed_trials}")
            print("-" * 40)
    
    print("\n🎯 Final Results:")
    print(f"Total Trials: {trials}")
    print(f"Successful: {successful_trials}")
    print(f"Failed: {failed_trials}")
    print(f"Success Rate: {(successful_trials/max(trials,1))*100:.1f}%")
    
    # Show tested alphas with actual results
    tested_alphas = [h for h in history if h.get('fitness', 0) > 0 or h.get('sharpe', 0) > 0]
    if tested_alphas:
        print(f"\n🧪 Successfully Tested Alphas ({len(tested_alphas)}):")
        for i, h in enumerate(tested_alphas[-5:], 1):
            print(f"{i}. ID: {h['alpha_id']} | Fitness: {h['fitness']:.3f} | Sharpe: {h['sharpe']:.3f}")
            print(f"   Code: {h['code']}")
            print()
    else:
        print("\n⚠️ No alphas were successfully tested with WorldQuant")
    

    
    print("\n✨ Test completed! Check WorldQuant platform for tested alphas.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EnhancedGeminiAgent trials")
    parser.add_argument("--trials", type=int, default=60, help="Number of trials to run")
    args = parser.parse_args()
    asyncio.run(test_enhanced_agent(trials=args.trials))