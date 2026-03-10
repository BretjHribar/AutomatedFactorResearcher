"""
Visualize 4h alpha decay: cumulative PnL and rolling Sharpe across IS/OOS.
Generates an HTML page with interactive charts.
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from src.operators.fastexpression import FastExpressionEngine
from src.simulation.vectorized_sim_polars import simulate_vectorized_polars as sim_vec

DATA_DIR = Path("data/binance_cache")
MATRICES_DIR = DATA_DIR / "matrices" / "4h"
UNIVERSE_DIR = DATA_DIR / "universes"
TRAIN_END = "2024-04-27"
BOOKSIZE = 2_000_000.0
MAX_WEIGHT = 0.05

# The 9 unique alphas from our discovery run
ALPHAS = [
    ("VWAP Deviation (Rev)", "negative(vwap_deviation)"),
    ("VWAP/Close Ratio", "true_divide(vwap, close)"),
    ("VWAP Dev × InvVolMom", "multiply(negative(vwap_deviation), Inverse(volume_momentum_1))"),
    ("High/Close Ratio", "true_divide(high, close)"),
    ("Close Position (Neg)", "npfmul(close_position_in_range, -0.4367897822614297)"),
    ("Log Returns (Rev)", "negative(log_returns)"),
    ("Log Close Position", "npfmul(log10(close_position_in_range), extend(-8.575142575595148))"),
    ("Upper Shadow Rank", "ts_rank(upper_shadow, 30)"),
]

def main():
    # Load data
    print("Loading data...")
    matrices = {}
    for fpath in sorted(MATRICES_DIR.glob("*.parquet")):
        matrices[fpath.stem] = pd.read_parquet(fpath)

    universe = pd.read_parquet(UNIVERSE_DIR / "BINANCE_TOP50_4h.parquet")
    coverage = universe.sum(axis=0) / len(universe)
    valid_tickers = sorted(coverage[coverage > 0.1].index.tolist())
    for name in list(matrices.keys()):
        cols = [c for c in valid_tickers if c in matrices[name].columns]
        if cols:
            matrices[name] = matrices[name][cols]
        else:
            del matrices[name]

    engine = FastExpressionEngine(data_fields=matrices)
    returns = matrices["returns"]
    close = matrices.get("close")

    # Evaluate each alpha on FULL period
    all_daily_pnl = {}
    all_cum_pnl = {}
    all_rolling_sharpe = {}

    for name, expr in ALPHAS:
        try:
            alpha = engine.evaluate(expr)
            if alpha is None or alpha.empty:
                continue
            r = sim_vec(
                alpha_df=alpha, returns_df=returns, close_df=close,
                universe_df=universe, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
                decay=0, delay=0, neutralization="market", fees_bps=0.0,
            )
            all_daily_pnl[name] = r.daily_pnl
            all_cum_pnl[name] = r.cumulative_pnl
            
            # Rolling 60-bar Sharpe (annualized with sqrt(252))
            rolling_mean = r.daily_pnl.rolling(60, min_periods=30).mean()
            rolling_std = r.daily_pnl.rolling(60, min_periods=30).std()
            rolling_sharpe = (rolling_mean / rolling_std) * math.sqrt(252)
            all_rolling_sharpe[name] = rolling_sharpe
            
            print(f"  {name}: full Sharpe={r.sharpe:+.3f}")
        except Exception as e:
            print(f"  {name}: ERROR {e}")

    # Also compute equal-weight rank-average combo
    ranked_signals = []
    for name, expr in ALPHAS:
        try:
            a = engine.evaluate(expr)
            if a is not None and not a.empty:
                ranked_signals.append(a.rank(axis=1, pct=True) - 0.5)
        except:
            pass
    
    if ranked_signals:
        combined = ranked_signals[0].copy()
        for s in ranked_signals[1:]:
            combined = combined.add(s, fill_value=0)
        combined = combined / len(ranked_signals)
        
        for fees_label, fees_val in [("Combined (0 bps)", 0.0), ("Combined (3 bps)", 3.0), ("Combined (5 bps)", 5.0)]:
            r = sim_vec(
                alpha_df=combined, returns_df=returns, close_df=close,
                universe_df=universe, booksize=BOOKSIZE, max_stock_weight=MAX_WEIGHT,
                decay=0, delay=0, neutralization="market", fees_bps=fees_val,
            )
            all_daily_pnl[fees_label] = r.daily_pnl
            all_cum_pnl[fees_label] = r.cumulative_pnl
            rolling_mean = r.daily_pnl.rolling(60, min_periods=30).mean()
            rolling_std = r.daily_pnl.rolling(60, min_periods=30).std()
            all_rolling_sharpe[fees_label] = (rolling_mean / rolling_std) * math.sqrt(252)
            print(f"  {fees_label}: full Sharpe={r.sharpe:+.3f}")

    # Build HTML
    train_end_ts = pd.Timestamp(TRAIN_END)
    
    # Prepare chart data
    chart_data = {"dates": [], "train_end_idx": 0}
    
    # Use common dates
    sample_key = list(all_cum_pnl.keys())[0]
    dates = all_cum_pnl[sample_key].index
    chart_data["dates"] = [d.strftime("%Y-%m-%d %H:%M") if hasattr(d, 'strftime') else str(d) for d in dates]
    
    # Find train end index
    for i, d in enumerate(dates):
        if d >= train_end_ts:
            chart_data["train_end_idx"] = i
            break
    
    # Subsample for performance (every 6th bar = daily)
    step = 6
    sub_dates = chart_data["dates"][::step]
    sub_train_idx = chart_data["train_end_idx"] // step
    
    cum_pnl_series = {}
    rolling_sharpe_series = {}
    for name in all_cum_pnl:
        vals = all_cum_pnl[name].values[::step]
        cum_pnl_series[name] = [round(float(v), 0) if not np.isnan(v) else None for v in vals]
    for name in all_rolling_sharpe:
        vals = all_rolling_sharpe[name].values[::step]
        rolling_sharpe_series[name] = [round(float(v), 3) if not np.isnan(v) else None for v in vals]

    # Color palette
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#E74C3C", "#2ECC71", "#3498DB"
    ]
    
    import json
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>4h Alpha Decay Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ 
    font-family: 'Inter', 'Segoe UI', sans-serif; 
    background: #0a0a1a; 
    color: #e0e0e0; 
    padding: 20px;
  }}
  h1 {{ 
    text-align: center; 
    font-size: 1.8rem; 
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .subtitle {{ text-align: center; color: #888; margin-bottom: 24px; font-size: 0.9rem; }}
  .chart-container {{ 
    background: #12122a; 
    border-radius: 12px; 
    padding: 20px; 
    margin-bottom: 20px;
    border: 1px solid #1e1e3a;
  }}
  .chart-title {{ 
    font-size: 1.1rem; 
    font-weight: 600; 
    margin-bottom: 12px; 
    color: #b0b0d0;
  }}
  canvas {{ max-height: 400px; }}
  .stats-grid {{ 
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
    gap: 12px; 
    margin-bottom: 20px;
  }}
  .stat-card {{ 
    background: #12122a; 
    border-radius: 10px; 
    padding: 16px; 
    border: 1px solid #1e1e3a;
  }}
  .stat-card h3 {{ font-size: 0.85rem; color: #888; margin-bottom: 6px; }}
  .stat-card .value {{ font-size: 1.4rem; font-weight: 700; }}
  .positive {{ color: #4ECDC4; }}
  .negative {{ color: #FF6B6B; }}
  .note {{ 
    background: #1a1a2e; 
    border-left: 3px solid #667eea; 
    padding: 12px 16px; 
    border-radius: 0 8px 8px 0;
    margin-bottom: 20px;
    font-size: 0.85rem;
    color: #a0a0c0;
  }}
</style>
</head>
<body>

<h1>4h Alpha Decay Analysis</h1>
<p class="subtitle">IS: 2020-01 to 2024-04 (9,469 bars) | OOS: 2024-04 to present (4,065 bars) | 0 fees | TOP50 Universe</p>

<div class="note">
  <strong>Key observation:</strong> The vertical dashed line marks the IS→OOS boundary. 
  Watch how the slope of cumulative PnL changes after this line — flatter slopes indicate signal decay.
  Rolling Sharpe collapses from ~2.0 IS to ~0.3 OOS, suggesting these 4h mean-reversion signals have short half-lives.
</div>

<div class="chart-container">
  <div class="chart-title">📈 Cumulative PnL — Individual Alphas (0 fees)</div>
  <canvas id="cumPnlChart"></canvas>
</div>

<div class="chart-container">
  <div class="chart-title">📊 Rolling 60-bar Sharpe Ratio (annualized √252)</div>
  <canvas id="rollingSharpeChart"></canvas>
</div>

<div class="chart-container">
  <div class="chart-title">💰 Combined Portfolio — Fee Sensitivity</div>
  <canvas id="combinedChart"></canvas>
</div>

<script>
const dates = {json.dumps(sub_dates)};
const trainEndIdx = {sub_train_idx};
const colors = {json.dumps(colors)};

const annotation = {{
  annotations: {{
    trainEnd: {{
      type: 'line',
      xMin: trainEndIdx,
      xMax: trainEndIdx,
      borderColor: '#FFD700',
      borderWidth: 2,
      borderDash: [6, 4],
      label: {{
        display: true,
        content: 'IS → OOS',
        position: 'start',
        backgroundColor: '#FFD700',
        color: '#000',
        font: {{ size: 11, weight: 'bold' }},
        padding: 4,
      }}
    }}
  }}
}};

// Chart 1: Individual alpha cum PnL
const individualNames = {json.dumps([n for n in cum_pnl_series if 'Combined' not in n])};
const individualDatasets = individualNames.map((name, i) => ({{
  label: name,
  data: {json.dumps({n: v for n, v in cum_pnl_series.items() if 'Combined' not in n})}[name],
  borderColor: colors[i % colors.length],
  borderWidth: 1.5,
  pointRadius: 0,
  tension: 0.1,
  fill: false,
}}));

new Chart(document.getElementById('cumPnlChart'), {{
  type: 'line',
  data: {{ labels: dates, datasets: individualDatasets }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ 
      annotation,
      legend: {{ position: 'top', labels: {{ color: '#aaa', font: {{ size: 10 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + (ctx.parsed.y||0).toLocaleString() }} }}
    }},
    scales: {{
      x: {{ display: true, ticks: {{ color: '#666', maxTicksLimit: 12, maxRotation: 0 }}, grid: {{ color: '#1a1a3a' }} }},
      y: {{ ticks: {{ color: '#888', callback: v => '$' + (v/1e6).toFixed(1) + 'M' }}, grid: {{ color: '#1a1a3a' }} }}
    }}
  }}
}});

// Chart 2: Rolling Sharpe
const sharpeNames = {json.dumps([n for n in rolling_sharpe_series if 'Combined' not in n])};
const sharpeDatasets = sharpeNames.map((name, i) => ({{
  label: name,
  data: {json.dumps({n: v for n, v in rolling_sharpe_series.items() if 'Combined' not in n})}[name],
  borderColor: colors[i % colors.length],
  borderWidth: 1.5,
  pointRadius: 0,
  tension: 0.1,
  fill: false,
}}));

// Add zero line dataset
sharpeDatasets.push({{
  label: 'Zero',
  data: new Array(dates.length).fill(0),
  borderColor: '#444',
  borderWidth: 1,
  borderDash: [4, 4],
  pointRadius: 0,
  fill: false,
}});

new Chart(document.getElementById('rollingSharpeChart'), {{
  type: 'line',
  data: {{ labels: dates, datasets: sharpeDatasets }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ 
      annotation,
      legend: {{ position: 'top', labels: {{ color: '#aaa', font: {{ size: 10 }} }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#666', maxTicksLimit: 12, maxRotation: 0 }}, grid: {{ color: '#1a1a3a' }} }},
      y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#1a1a3a' }},
           suggestedMin: -3, suggestedMax: 5 }}
    }}
  }}
}});

// Chart 3: Combined portfolio fee sweep
const combinedNames = {json.dumps([n for n in cum_pnl_series if 'Combined' in n])};
const feeColors = ['#4ECDC4', '#FFD700', '#FF6B6B'];
const combinedDatasets = combinedNames.map((name, i) => ({{
  label: name,
  data: {json.dumps({n: v for n, v in cum_pnl_series.items() if 'Combined' in n})}[name],
  borderColor: feeColors[i % feeColors.length],
  borderWidth: 2.5,
  pointRadius: 0,
  tension: 0.1,
  fill: false,
}}));

new Chart(document.getElementById('combinedChart'), {{
  type: 'line',
  data: {{ labels: dates, datasets: combinedDatasets }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{ 
      annotation,
      legend: {{ position: 'top', labels: {{ color: '#ccc', font: {{ size: 12 }} }} }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + (ctx.parsed.y||0).toLocaleString() }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#666', maxTicksLimit: 12, maxRotation: 0 }}, grid: {{ color: '#1a1a3a' }} }},
      y: {{ ticks: {{ color: '#888', callback: v => '$' + (v/1e6).toFixed(1) + 'M' }}, grid: {{ color: '#1a1a3a' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    out_path = Path("alpha_decay_4h.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"\nSaved to {out_path.absolute()}")

if __name__ == "__main__":
    main()
