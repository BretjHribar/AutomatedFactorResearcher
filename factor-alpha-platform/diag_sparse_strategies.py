"""
Test 5 sparse trading strategies at 3 bps and 7 bps.
Lean version with progress logging.
"""
import sys, time, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import eval_alpha_5m as ea
import eval_portfolio_5m as ep

BOOKSIZE = ep.BOOKSIZE
MAX_WT = ea.MAX_WEIGHT
BARS_PER_DAY = ep.BARS_PER_DAY

def sharpe(pnl, bpd=BARS_PER_DAY):
    daily = [np.sum(pnl[d:d+bpd]) for d in range(0, len(pnl), bpd)]
    d = np.array(daily)
    return np.mean(d)/np.std(d)*np.sqrt(365) if len(d)>1 and np.std(d)>0 else 0

def continuous_baseline(sig, ret, ewma_span=1, fees_bps=3):
    n, m = sig.shape
    s = sig.copy()
    if ewma_span > 1:
        a = 2.0/(ewma_span+1)
        for t in range(1, n): s[t] = a*s[t]+(1-a)*s[t-1]
    fee = fees_bps/10000.0
    pos = np.zeros(m); pnl = np.zeros(n); to_total = 0
    for t in range(1, n):
        pnl[t] = np.sum(pos*ret[t])*BOOKSIZE
        asum = np.abs(s[t]).sum()
        if asum > 1e-10:
            new = np.clip(s[t]/asum, -MAX_WT, MAX_WT)
            asum2 = np.abs(new).sum()
            if asum2 > 1e-10: new /= asum2
        else: new = pos.copy()
        to = np.abs(new-pos).sum(); to_total += to
        pnl[t] -= to*fee*BOOKSIZE; pos = new
    return pnl, to_total/(n-1), np.cumsum(pnl)


def s1_zscore_gate(sig, ret, lb=288, entry_z=1.0, exit_z=0.3, min_hold=12, top_k=10, fees_bps=3):
    """TS Z-Score Gating: only trade when signal is abnormally strong vs own history"""
    n, m = sig.shape; fee = fees_bps/10000.0
    tsz = np.zeros_like(sig)
    for t in range(lb, n):
        w = sig[t-lb:t]; mu = np.mean(w,0); sd = np.std(w,0); sd[sd<1e-8]=1e-8
        tsz[t] = (sig[t]-mu)/sd
    pos = np.zeros(m); hold = np.zeros(m,dtype=int); pnl = np.zeros(n); to_t = 0
    for t in range(lb+1, n):
        pnl[t] = np.sum(pos*ret[t])*BOOKSIZE; hold[pos!=0] += 1
        for j in range(m):
            if pos[j]!=0 and hold[j]>=min_hold:
                if abs(tsz[t,j])<exit_z or np.sign(tsz[t,j])!=np.sign(pos[j]):
                    to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE; pos[j]=0; hold[j]=0
        longs = np.argsort(tsz[t])[-top_k:]; shorts = np.argsort(tsz[t])[:top_k]
        for j in longs:
            if tsz[t,j]>entry_z and pos[j]==0:
                pos[j]=1.0/(2*top_k); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
        for j in shorts:
            if tsz[t,j]<-entry_z and pos[j]==0:
                pos[j]=-1.0/(2*top_k); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
    return pnl, to_t/(n-lb), np.cumsum(pnl)


def s2_rank_hysteresis(sig, ret, enter_rank=5, exit_rank=20, min_hold=24, fees_bps=3):
    """Rank with hysteresis: enter top-K, exit only when drops below wider band"""
    n, m = sig.shape; fee = fees_bps/10000.0
    pos = np.zeros(m); hold = np.zeros(m,dtype=int); pnl = np.zeros(n); to_t = 0
    for t in range(1, n):
        pnl[t] = np.sum(pos*ret[t])*BOOKSIZE; hold[pos!=0] += 1
        ranks = np.argsort(np.argsort(-sig[t]))
        na = (np.abs(sig[t])>1e-10).sum()
        if na < 10: continue
        for j in range(m):
            if pos[j]>0 and (ranks[j]>=exit_rank or sig[t,j]<=0) and hold[j]>=min_hold:
                to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE; pos[j]=0; hold[j]=0
            elif pos[j]<0 and (ranks[j]<na-exit_rank or sig[t,j]>=0) and hold[j]>=min_hold:
                to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE; pos[j]=0; hold[j]=0
            elif pos[j]==0:
                if ranks[j]<enter_rank and sig[t,j]>0:
                    pos[j]=1.0/(2*enter_rank); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
                elif ranks[j]>=na-enter_rank and sig[t,j]<0:
                    pos[j]=-1.0/(2*enter_rank); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
    return pnl, to_t/(n-1), np.cumsum(pnl)


def s3_disp_gated(sig, ret, disp_pct=75, top_k=10, min_hold=12, fees_bps=3):
    """Only trade when cross-section dispersion is high (clear winners/losers)"""
    n, m = sig.shape; fee = fees_bps/10000.0; lb = 288
    cs_d = np.std(sig, axis=1); thr = np.zeros(n)
    for t in range(lb, n): thr[t] = np.percentile(cs_d[t-lb:t], disp_pct)
    pos = np.zeros(m); hold = np.zeros(m,dtype=int); pnl = np.zeros(n); to_t = 0; active = False
    for t in range(lb+1, n):
        pnl[t] = np.sum(pos*ret[t])*BOOKSIZE; hold[pos!=0] += 1
        hd = cs_d[t] > thr[t]
        if hd and not active:
            si = np.argsort(sig[t])
            for j in si[-top_k:]:
                if sig[t,j]>0 and pos[j]==0:
                    pos[j]=1.0/(2*top_k); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
            for j in si[:top_k]:
                if sig[t,j]<0 and pos[j]==0:
                    pos[j]=-1.0/(2*top_k); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
            active = True
        elif not hd and active:
            if all(hold[j]>=min_hold for j in range(m) if pos[j]!=0):
                for j in range(m):
                    if pos[j]!=0: to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE; pos[j]=0; hold[j]=0
                active = False
    return pnl, to_t/(n-lb), np.cumsum(pnl)


def s4_momentum_cross(sig, ret, slow=72, fast=6, top_k=10, min_hold=24, fees_bps=3):
    """Enter when fast EMA of signal crosses above slow EMA (signal accelerating)"""
    n, m = sig.shape; fee = fees_bps/10000.0
    f_ema = np.zeros_like(sig); s_ema = np.zeros_like(sig)
    f_ema[0]=sig[0]; s_ema[0]=sig[0]
    fa=2.0/(fast+1); sa=2.0/(slow+1)
    for t in range(1,n): f_ema[t]=fa*sig[t]+(1-fa)*f_ema[t-1]; s_ema[t]=sa*sig[t]+(1-sa)*s_ema[t-1]
    mom = f_ema - s_ema
    pos = np.zeros(m); hold = np.zeros(m,dtype=int); pnl = np.zeros(n); to_t = 0; pmom = np.zeros(m)
    for t in range(slow+1, n):
        pnl[t] = np.sum(pos*ret[t])*BOOKSIZE; hold[pos!=0]+=1
        for j in range(m):
            if pos[j]!=0 and hold[j]>=min_hold:
                if (pos[j]>0 and mom[t,j]<0) or (pos[j]<0 and mom[t,j]>0):
                    to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE; pos[j]=0; hold[j]=0
        tl = np.argsort(mom[t])[-top_k:]; ts = np.argsort(mom[t])[:top_k]
        for j in tl:
            if mom[t,j]>0 and pmom[j]<=0 and pos[j]==0 and sig[t,j]>0:
                pos[j]=1.0/(2*top_k); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
        for j in ts:
            if mom[t,j]<0 and pmom[j]>=0 and pos[j]==0 and sig[t,j]<0:
                pos[j]=-1.0/(2*top_k); hold[j]=0; to_t+=abs(pos[j]); pnl[t]-=abs(pos[j])*fee*BOOKSIZE
        pmom = mom[t].copy()
    return pnl, to_t/(n-slow), np.cumsum(pnl)


def s5_bucket_hold(sig, ret, n_bkts=5, rebal=36, fees_bps=3):
    """Rank into quintiles, long top short bottom, rebalance every N bars with EMA"""
    n, m = sig.shape; fee = fees_bps/10000.0
    es = np.zeros_like(sig); es[0]=sig[0]; a=2.0/(rebal+1)
    for t in range(1,n): es[t]=a*sig[t]+(1-a)*es[t-1]
    pos = np.zeros(m); pnl = np.zeros(n); to_t = 0
    for t in range(1, n):
        pnl[t] = np.sum(pos*ret[t])*BOOKSIZE
        if t % rebal == 0:
            s = es[t]; na = (np.abs(s)>1e-10).sum()
            if na < 2*n_bkts: continue
            bs = na//n_bkts; si = np.argsort(s); ai = si[np.abs(s[si])>1e-10]
            new = np.zeros(m)
            for j in ai[-bs:]:
                if s[j]>0: new[j]=1.0/(2*bs)
            for j in ai[:bs]:
                if s[j]<0: new[j]=-1.0/(2*bs)
            to = np.abs(new-pos).sum(); to_t+=to; pnl[t]-=to*fee*BOOKSIZE; pos=new
    return pnl, to_t/(n-1), np.cumsum(pnl)


# ========== LOAD DATA ==========
print("Loading data...", flush=True); t0 = time.time()
matrices, universe = ea.load_data("train")
close = matrices["close"]; returns = close.pct_change()
alphas = ep.load_alphas(universe="TOP100")
print(f"Data loaded in {time.time()-t0:.0f}s. Building composite...", flush=True)

raw_sum = None; n = 0
for i, (aid, expr, ic, sr) in enumerate(alphas):
    print(f"  Alpha {i+1}/{len(alphas)} (#{aid})...", end='', flush=True)
    try:
        raw = ep.evaluate_expression(expr, matrices)
        if raw is None: print(" SKIP"); continue
        r = raw.fillna(0)
        if raw_sum is None: raw_sum = r.copy()
        else: raw_sum = raw_sum.add(r, fill_value=0)
        n += 1; print(f" OK", flush=True)
    except Exception as e: print(f" ERR: {e}")

composite = ea.process_signal(raw_sum / n, universe_df=universe, max_wt=MAX_WT)
common = composite.columns.intersection(returns.columns).tolist()
idx = composite.index.intersection(returns.index)
sig = np.nan_to_num(composite.loc[idx, common].values.astype(np.float64), nan=0)
ret = np.nan_to_num(returns.loc[idx, common].values.astype(np.float64), nan=0)
print(f"Composite ready: {sig.shape[0]} bars, {sig.shape[1]} tickers", flush=True)

# ======== RUN ALL STRATEGIES ========
print(f"\n{'='*80}", flush=True)
print("Running strategy sweep...", flush=True)

results = {}

# Baselines
for ema in [1, 48, 96]:
    for fee in [3, 7]:
        pnl, to, cum = continuous_baseline(sig, ret, ewma_span=ema, fees_bps=fee)
        sr = sharpe(pnl); tr = cum[-1]/BOOKSIZE
        results[f"Cont EMA={ema} {fee}bp"] = (sr, to, tr, cum)
        print(f"  Baseline EMA={ema:>3} {fee}bp: SR={sr:+.2f} TO={to:.4f} Ret={tr:+.1%}", flush=True)

# S1: TS Z-Score Gating
print("\nS1: Z-Score Gating...", flush=True)
for ez in [0.5, 1.0, 1.5, 2.0]:
    for xz in [0.0, 0.3]:
        for mh in [6, 12, 24]:
            for k in [5, 10]:
                for fee in [3, 7]:
                    pnl, to, cum = s1_zscore_gate(sig, ret, entry_z=ez, exit_z=xz, min_hold=mh, top_k=k, fees_bps=fee)
                    sr = sharpe(pnl); tr = cum[-1]/BOOKSIZE
                    key = f"S1 z={ez} xz={xz} h={mh} k={k} {fee}bp"
                    results[key] = (sr, to, tr, cum)
                    if sr > 0: print(f"  + {key}: SR={sr:+.2f} TO={to:.4f} Ret={tr:+.1%}", flush=True)

# S2: Rank Hysteresis
print("\nS2: Rank Hysteresis...", flush=True)
for er in [3, 5, 10]:
    for xr in [15, 20, 30]:
        if xr <= er: continue
        for mh in [12, 24, 36]:
            for fee in [3, 7]:
                pnl, to, cum = s2_rank_hysteresis(sig, ret, enter_rank=er, exit_rank=xr, min_hold=mh, fees_bps=fee)
                sr = sharpe(pnl); tr = cum[-1]/BOOKSIZE
                key = f"S2 er={er} xr={xr} h={mh} {fee}bp"
                results[key] = (sr, to, tr, cum)
                if sr > 0: print(f"  + {key}: SR={sr:+.2f} TO={to:.4f} Ret={tr:+.1%}", flush=True)

# S3: Dispersion Gated
print("\nS3: Dispersion Gated...", flush=True)
for dp in [60, 75, 90]:
    for k in [5, 10]:
        for mh in [6, 12, 24]:
            for fee in [3, 7]:
                pnl, to, cum = s3_disp_gated(sig, ret, disp_pct=dp, top_k=k, min_hold=mh, fees_bps=fee)
                sr = sharpe(pnl); tr = cum[-1]/BOOKSIZE
                key = f"S3 dp={dp} k={k} h={mh} {fee}bp"
                results[key] = (sr, to, tr, cum)
                if sr > 0: print(f"  + {key}: SR={sr:+.2f} TO={to:.4f} Ret={tr:+.1%}", flush=True)

# S4: Signal Momentum
print("\nS4: Signal Momentum...", flush=True)
for sl in [36, 72, 144]:
    for fa in [3, 6, 12]:
        for k in [5, 10]:
            for mh in [12, 24]:
                for fee in [3, 7]:
                    pnl, to, cum = s4_momentum_cross(sig, ret, slow=sl, fast=fa, top_k=k, min_hold=mh, fees_bps=fee)
                    sr = sharpe(pnl); tr = cum[-1]/BOOKSIZE
                    key = f"S4 s={sl} f={fa} k={k} h={mh} {fee}bp"
                    results[key] = (sr, to, tr, cum)
                    if sr > 0: print(f"  + {key}: SR={sr:+.2f} TO={to:.4f} Ret={tr:+.1%}", flush=True)

# S5: Bucket Hold
print("\nS5: Bucket Hold...", flush=True)
for nb in [3, 5, 10]:
    for rb in [12, 24, 36, 72, 144]:
        for fee in [3, 7]:
            pnl, to, cum = s5_bucket_hold(sig, ret, n_bkts=nb, rebal=rb, fees_bps=fee)
            sr = sharpe(pnl); tr = cum[-1]/BOOKSIZE
            key = f"S5 b={nb} r={rb} {fee}bp"
            results[key] = (sr, to, tr, cum)
            if sr > 0: print(f"  + {key}: SR={sr:+.2f} TO={to:.4f} Ret={tr:+.1%}", flush=True)

# ======== SUMMARY ========
print(f"\n{'='*80}", flush=True)
print("TOP 10 STRATEGIES BY SHARPE (3 bps):", flush=True)
three_bps = [(k,v) for k,v in results.items() if "3bp" in k]
three_bps.sort(key=lambda x: -x[1][0])
for k, (sr, to, tr, _) in three_bps[:10]:
    print(f"  {sr:+6.2f} SR  {to:.4f} TO  {tr:+8.1%} Ret  | {k}", flush=True)

print(f"\nTOP 10 STRATEGIES BY SHARPE (7 bps):", flush=True)
seven_bps = [(k,v) for k,v in results.items() if "7bp" in k]
seven_bps.sort(key=lambda x: -x[1][0])
for k, (sr, to, tr, _) in seven_bps[:10]:
    print(f"  {sr:+6.2f} SR  {to:.4f} TO  {tr:+8.1%} Ret  | {k}", flush=True)

# ======== PLOT ========
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top strategies at 3bps
ax = axes[0, 0]
for k, (sr, to, tr, cum) in three_bps[:6]:
    ax.plot(cum, label=f"{k[:25]} (SR={sr:+.1f})", linewidth=1.5)
ax.set_title('Top Strategies at 3 bps'); ax.set_xlabel('Bar'); ax.set_ylabel('Cum PnL ($)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Top strategies at 7bps
ax = axes[0, 1]
for k, (sr, to, tr, cum) in seven_bps[:6]:
    ax.plot(cum, label=f"{k[:25]} (SR={sr:+.1f})", linewidth=1.5)
ax.set_title('Top Strategies at 7 bps'); ax.set_xlabel('Bar'); ax.set_ylabel('Cum PnL ($)')
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# SR by strategy type at 3bps
ax = axes[1, 0]
by_type = {}
for k, (sr, to, tr, _) in three_bps:
    stype = k.split()[0]
    if stype not in by_type or sr > by_type[stype][0]:
        by_type[stype] = (sr, to, k)
types = sorted(by_type.keys())
srs = [by_type[t][0] for t in types]
colors = ['green' if s>0 else 'red' for s in srs]
ax.barh(range(len(types)), srs, color=colors, alpha=0.7)
ax.set_yticks(range(len(types))); ax.set_yticklabels(types)
ax.set_xlabel('Best Sharpe (3 bps)'); ax.set_title('Best SR by Strategy Type (3 bps)')
ax.axvline(x=0, color='gray', linestyle='--'); ax.grid(True, alpha=0.3)

# Same for 7bps
ax = axes[1, 1]
by_type7 = {}
for k, (sr, to, tr, _) in seven_bps:
    stype = k.split()[0]
    if stype not in by_type7 or sr > by_type7[stype][0]:
        by_type7[stype] = (sr, to, k)
types7 = sorted(by_type7.keys())
srs7 = [by_type7[t][0] for t in types7]
colors7 = ['green' if s>0 else 'red' for s in srs7]
ax.barh(range(len(types7)), srs7, color=colors7, alpha=0.7)
ax.set_yticks(range(len(types7))); ax.set_yticklabels(types7)
ax.set_xlabel('Best Sharpe (7 bps)'); ax.set_title('Best SR by Strategy Type (7 bps)')
ax.axvline(x=0, color='gray', linestyle='--'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diag_sparse_strategies.png', dpi=150)
print(f"\nChart saved: diag_sparse_strategies.png", flush=True)
plt.close()
print(f"Total time: {time.time()-t0:.0f}s", flush=True)
