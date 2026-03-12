"""
V9c: V9 base + ensemble top-3 configs (not top-1).
Combine the signal diversity of V9 with mild ensembling.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time

DATA_DIR = 'data/binance_futures_15m'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
FEE_BPS = 3; FEE_FRAC = FEE_BPS / 10000.0

def sma(s,w): return s.rolling(w,min_periods=max(w//2,2)).mean()
def ema(s,w): return s.ewm(halflife=w,min_periods=1).mean()
def stddev(s,w): return s.rolling(w,min_periods=2).std()
def ts_zscore(s,w):
    m=s.rolling(w,min_periods=max(w//2,2)).mean()
    sd=s.rolling(w,min_periods=max(w//2,2)).std()
    return (s-m)/sd.replace(0,np.nan)
def delta(s,p): return s-s.shift(p)
def ts_sum(s,w): return s.rolling(w,min_periods=1).sum()
def ts_min(s,w): return s.rolling(w,min_periods=1).min()
def ts_max(s,w): return s.rolling(w,min_periods=1).max()
def safe_div(a,b): r=a/b; return r.replace([np.inf,-np.inf],0).fillna(0)
def correlation(x,y,w): return x.rolling(w,min_periods=2).corr(y)
def decay_exp(s,hl):
    if 0<hl<1: ah=-np.log(2)/np.log(hl)
    else: ah=hl
    return s.ewm(halflife=max(ah,0.5),min_periods=1).mean()

def build_htf_signals(df_htf, df_1h, prefix, shift_n=1):
    c = df_htf['close'].shift(shift_n);h = df_htf['high'].shift(shift_n);l = df_htf['low'].shift(shift_n)
    v = df_htf['volume'].shift(shift_n);lr = np.log(c / c.shift(1));cp = safe_div(c - l, h - l)
    alphas = {}
    for w in [3,5,8,12,20,30]:
        alphas[f'{prefix}_mr_{w}'] = (-ts_zscore(c, w)).reindex(df_1h.index, method='ffill')
    for w in [3,5,8,12,20]:
        alphas[f'{prefix}_mom_{w}'] = ts_sum(lr, w).reindex(df_1h.index, method='ffill')
    for w in [12,24,48,72,120]:
        h_ = ts_max(h, w); l_ = ts_min(l, w); rng = h_ - l_
        alphas[f'{prefix}_brk_{w}'] = (safe_div(c - l_, rng) * 2 - 1).reindex(df_1h.index, method='ffill')
    for mw in [3,6,12]:
        for dc in [0.9,0.95,0.98]:
            alphas[f'{prefix}_dec{dc}_d{mw}'] = decay_exp((c - c.shift(mw)) * cp, dc).reindex(df_1h.index, method='ffill')
    vz = ts_zscore(v, 20)
    if vz is not None: alphas[f'{prefix}_vol_z20'] = vz.reindex(df_1h.index, method='ffill')
    return alphas

def build_1h_alphas(df_1h):
    close=df_1h['close'];volume=df_1h['volume'];high=df_1h['high'];low=df_1h['low'];opn=df_1h['open']
    qv=df_1h['quote_volume'];tbv=df_1h['taker_buy_volume']
    ret=close.pct_change();log_ret=np.log(close/close.shift(1))
    vwap=safe_div(qv,volume);taker_ratio=safe_div(tbv,volume)
    rvol_short=ret.rolling(6,min_periods=2).std();rvol_long=ret.rolling(72,min_periods=2).std()
    close_pos=safe_div(close-low,high-low);body=close-opn;candle_range=high-low
    alphas={}
    for w in [3,4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,30,36,42,48,60,72,96]:
        alphas[f'mr_{w}']=-ts_zscore(close,w)
    for w in [3,5,8,10,12,15,20,24,30]:
        alphas[f'logrev_{w}']=-ts_sum(log_ret,w)
    for w in [3,5,8,10,12,15,20]:
        alphas[f'dstd_{w}']=-safe_div(delta(close,w),stddev(close,w))
    for w in [5,10,15,20,30,48]:
        alphas[f'vwap_mr_{w}']=-ts_zscore(vwap,w)
    for w in [5,10,15,20,30]:
        alphas[f'ema_mr_{w}']=-(close-ema(close,w))/stddev(close,w*2)
    for w in [5,10,20,30]:
        alphas[f'ema_ret_{w}']=-ts_zscore(ema(ret,w),w*2)
    for w in [3,5,8,12,18,24,36,48,72]:
        alphas[f'mom_{w}']=ts_sum(log_ret,w)
    for fast,slow in [(3,12),(5,15),(5,20),(8,24),(12,36),(12,48),(24,72),(36,120)]:
        alphas[f'emax_{fast}_{slow}']=(ema(close,fast)-ema(close,slow))/stddev(close,slow)
    for w in [12,24,36,48,72,96,120,168,240,360]:
        h=ts_max(high,w);l=ts_min(low,w);rng=h-l
        alphas[f'breakout_{w}']=safe_div(close-l,rng)*2-1
    for w in [6,12,24,48]:
        up=high-high.shift(1);dn=low.shift(1)-low
        plus_dm=up.where((up>dn)&(up>0),0);minus_dm=dn.where((dn>up)&(dn>0),0)
        atr=(high-low).rolling(w,min_periods=1).mean()
        alphas[f'trend_{w}']=safe_div(plus_dm.rolling(w).mean(),atr)-safe_div(minus_dm.rolling(w).mean(),atr)
    for mw in [3,6,12,24,48]:
        for dc in [0.8,0.9,0.95,0.98,0.99]:
            alphas[f'dec{dc}_d{mw}_cp']=decay_exp((close-close.shift(mw))*close_pos,dc)
            alphas[f'dec{dc}_d{mw}_tbr']=decay_exp((close-close.shift(mw))*taker_ratio,dc)
    vol_ratio=safe_div(rvol_short,rvol_long)
    for w in [8,10,15,20,30]:
        alphas[f'lovol_mr_{w}']=-ts_zscore(close,w)*(vol_ratio<1.0).astype(float)
    for w in [5,8,12,24]:
        alphas[f'hivol_mom_{w}']=ts_sum(log_ret,w)*(vol_ratio>1.0).astype(float)
    for w in [10,20,30]:
        alphas[f'vs_mr_{w}']=-ts_zscore(close,w)*safe_div(rvol_long,rvol_short).clip(0.2,5.0)
    obv=(np.sign(ret)*volume).cumsum()
    for w in [10,20,30,48]:alphas[f'obv_{w}']=-ts_zscore(obv,w)
    for w in [5,10,20,30]:alphas[f'tbr_{w}']=ts_zscore(taker_ratio,w)
    timb=safe_div(tbv-(volume-tbv),volume)
    for w in [5,10,20,30]:alphas[f'timb_{w}']=ts_zscore(timb,w)
    for w in [5,10,20]:
        vw=(ret*volume).rolling(w).sum();vs=volume.rolling(w).sum()
        alphas[f'vwret_{w}']=safe_div(vw,vs)
    for w in [5,10,20]:alphas[f'body_{w}']=ts_zscore(safe_div(body,candle_range),w)
    uwick=high-np.maximum(close,opn);lwick=np.minimum(close,opn)-low
    rejection=safe_div(uwick-lwick,candle_range)
    for w in [5,10]:alphas[f'reject_{w}']=-ts_zscore(rejection,w)
    atr_v=(high-low).rolling(14,min_periods=2).mean()
    for w in [5,10,20]:alphas[f'atr_mr_{w}']=-safe_div(delta(close,w),atr_v)
    for w in [12,24,48,72]:
        ret_lag=ret.shift(1);ac=correlation(ret,ret_lag,w)
        alphas[f'regime_mom_{w}']=ts_sum(log_ret,5)*ac.clip(lower=0)
        alphas[f'regime_mr_{w}']=-ts_zscore(close,10)*(-ac).clip(lower=0)
    m5=ts_sum(log_ret,5);m20=ts_sum(log_ret,20);m60=ts_sum(log_ret,60)
    alphas['mtf_agree']=np.sign(m5)*np.sign(m20)*np.sign(m60)*abs(m5)
    alphas['trend_pullback']=-ts_zscore(close,5)*np.sign(m60)
    alphas['trend_pullback_20']=-ts_zscore(close,10)*np.sign(m20)
    for w in [8,10,15,20,30]:
        s=close.rolling(w).mean();sd=close.rolling(w).std()
        alphas[f'bb_{w}']=-(close-s)/sd.replace(0,np.nan)
    for w in [7,10,14,21]:
        d=close.diff();gain=d.clip(lower=0).rolling(w).mean();loss=(-d.clip(upper=0)).rolling(w).mean()
        rs=safe_div(gain,loss);rsi=100-100/(1+rs);alphas[f'rsi_{w}']=-(rsi-50)/50
    for w in [14,21]:
        lowest=ts_min(low,w);highest=ts_max(high,w)
        alphas[f'stoch_{w}']=-(safe_div(close-lowest,highest-lowest)*100-50)/50
    for w in [14,20]:
        tp=(high+low+close)/3;tp_sma=sma(tp,w)
        md=tp.rolling(w).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
        alphas[f'cci_{w}']=-safe_div(tp-tp_sma,0.015*md)
    for w in [3,5,8,12]:
        mom_w = ts_sum(log_ret, w);alphas[f'accel_{w}'] = mom_w - mom_w.shift(w)
    vol_ma5 = volume.rolling(5).mean();vol_ma20 = volume.rolling(20).mean()
    alphas['vol_accel'] = safe_div(vol_ma5, vol_ma20) - 1
    ib_vol = safe_div(high - low, close.shift(1))
    for w in [10, 20]:alphas[f'ibvol_z_{w}'] = -ts_zscore(ib_vol, w)
    direction = np.sign(ret)
    for w in [3, 5, 8]:alphas[f'consec_{w}'] = -direction.rolling(w).sum() / w
    return alphas

def build_cross_asset_all(all_1h, sym, df_1h):
    alphas = {}
    for factor_sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        if factor_sym == sym or factor_sym not in all_1h: continue
        fac = all_1h[factor_sym];common = df_1h.index.intersection(fac.index)
        if len(common) < 500: continue
        fac_lr = np.log(fac['close'] / fac['close'].shift(1)).loc[common]
        sym_ret = df_1h['close'].pct_change().loc[common]
        pfx = factor_sym[:3].lower()
        for w in [3,5,8,12,24]:alphas[f'{pfx}_mom_{w}'] = ts_sum(fac_lr, w).reindex(df_1h.index, method='ffill')
        for w in [5,10,20]:alphas[f'{pfx}_mr_{w}'] = (-ts_zscore(fac['close'].loc[common], w)).reindex(df_1h.index, method='ffill')
        fac_ret = fac['close'].pct_change().loc[common]
        for w in [5,12,24]:
            rel = ts_sum(sym_ret, w) - ts_sum(fac_ret, w)
            alphas[f'{pfx}_relstr_{w}'] = (-rel).reindex(df_1h.index, method='ffill')
    return alphas

def eval_alpha(signal,returns,min_bars=200):
    common=signal.dropna().index.intersection(returns.dropna().index)
    if len(common)<min_bars: return None
    d=np.sign(signal.loc[common]);raw=d*returns.loc[common]
    daily=raw.resample('1D').sum();daily=daily[daily!=0]
    if len(daily)<10 or daily.std()==0: return None
    return {'nofee_sharpe':daily.mean()/daily.std()*np.sqrt(365)}

def select_orth(results,alpha_matrix,max_n=15,corr_cutoff=0.70):
    selected=[]
    for r in results:
        if r['name'] not in alpha_matrix.columns: continue
        sig=alpha_matrix[r['name']]
        too_corr=False
        for sel in selected:
            if abs(sig.corr(alpha_matrix[sel['name']]))>corr_cutoff: too_corr=True;break
        if not too_corr: selected.append(r)
        if len(selected)>=max_n: break
    return selected

def get_combined_signal(alpha_matrix,returns,selected,lookback,phl):
    cols=[s['name'] for s in selected if s['name'] in alpha_matrix.columns]
    if len(cols)<2: return None
    X=alpha_matrix[cols].copy();ret=returns.copy()
    valid=X.dropna().index.intersection(ret.dropna().index)
    X,ret=X.loc[valid],ret.loc[valid]
    if len(valid)<lookback+24: return None
    factor_returns=pd.DataFrame(index=valid,columns=cols,dtype=float)
    for col in cols:factor_returns[col]=np.sign(X[col].values)*ret.values
    rolling_er=factor_returns.rolling(lookback,min_periods=min(100,lookback)).mean()
    weights=rolling_er.clip(lower=0)
    wsum=weights.sum(axis=1).replace(0,np.nan)
    weights_norm=weights.div(wsum,axis=0).fillna(0)
    if phl>1:
        weights_smooth=weights_norm.ewm(halflife=phl,min_periods=1).mean()
        wsum2=weights_smooth.sum(axis=1).replace(0,np.nan)
        weights_smooth=weights_smooth.div(wsum2,axis=0).fillna(0)
    else: weights_smooth=weights_norm
    return (X*weights_smooth).sum(axis=1)

def strategy_from_signal(combined,returns):
    valid = combined.dropna().index.intersection(returns.dropna().index)
    direction=np.sign(combined.loc[valid].values)
    ret_vals=returns.loc[valid].values
    pos_changes=np.abs(np.diff(np.concatenate([[0],direction])))
    pnl=direction*ret_vals-FEE_FRAC*pos_changes
    pnl_s=pd.Series(pnl,index=valid)
    daily=pnl_s.resample('1D').sum();daily=daily[daily!=0]
    if len(daily)<5 or daily.std()==0: return None
    sh=daily.mean()/daily.std()*np.sqrt(365)
    return {'sharpe':sh,'net_pnl_bps':pnl.sum()*10000,'pnl_series':pnl_s,
            'win_rate':(pnl[pnl!=0]>0).sum()/max((pnl!=0).sum(),1)}

ENSEMBLE_K = 3
print('='*70)
print(f'V9c: 4-TF + CROSS-ASSET + ENSEMBLE TOP-{ENSEMBLE_K}')
print('='*70)

all_1h = {}; all_data = {}
for sym in SYMBOLS:
    df15=pd.read_parquet(f'{DATA_DIR}/{sym}.parquet')
    df15=df15.set_index('datetime').sort_index();df15=df15[~df15.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume','quote_volume','taker_buy_volume','taker_buy_quote_volume']:
        df15[c]=pd.to_numeric(df15[c],errors='coerce')
    all_data[sym] = df15
    all_1h[sym] = df15.resample('1h').agg({'open':'first','high':'max','low':'min','close':'last',
        'volume':'sum','quote_volume':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum'}).dropna()

all_pnl={}
for sym in SYMBOLS:
    t0=time.time()
    df15 = all_data[sym]; df_1h = all_1h[sym]
    df_2h=df15.resample('2h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum'}).dropna()
    df_4h=df15.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum'}).dropna()
    df_8h=df15.resample('8h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum','quote_volume':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum'}).dropna()
    returns=df_1h['close'].pct_change()

    months=pd.date_range(start='2024-06-01',end='2025-04-01',freq='MS')
    fold_pnls=[];fold_details=[]

    for i in range(len(months)-1):
        test_start=months[i];test_end=months[i+1]
        train_start=test_start-pd.DateOffset(months=8)
        train_1h=df_1h.loc[str(train_start):str(test_start)]
        test_1h=df_1h.loc[str(test_start):str(test_end)]
        if len(train_1h)<2000 or len(test_1h)<400: continue
        train_ret=returns.loc[train_1h.index];test_ret=returns.loc[test_1h.index]

        a1h = build_1h_alphas(train_1h)
        tr_2h = df_2h.loc[str(train_start):str(test_start)]
        tr_4h = df_4h.loc[str(train_start):str(test_start)]
        tr_8h = df_8h.loc[str(train_start):str(test_start)]
        a2h = build_htf_signals(tr_2h, train_1h, 'h2', shift_n=1)
        a4h = build_htf_signals(tr_4h, train_1h, 'h4', shift_n=1)
        a8h = build_htf_signals(tr_8h, train_1h, 'h8', shift_n=1)
        across = build_cross_asset_all(all_1h, sym, train_1h)
        all_a = {**a1h, **a2h, **a4h, **a8h, **across}
        alpha_tr = pd.DataFrame(all_a, index=train_1h.index).shift(1)

        results=[]
        for col in alpha_tr.columns:
            m=eval_alpha(alpha_tr[col],train_ret)
            if m and m['nofee_sharpe']>0: results.append({'name':col,**m})
        results.sort(key=lambda x:x['nofee_sharpe'],reverse=True)
        if len(results)<3: continue

        # Collect ALL configs with their scores
        all_configs = []
        for cc in [0.40,0.50,0.60,0.70,0.80,0.90]:
            for mn in [6,8,10,12,15,20,25]:
                sel=select_orth(results,alpha_tr,max_n=mn,corr_cutoff=cc)
                if len(sel)<3: continue
                for lb in [120,180,240,360,480,720]:
                    for phl in [1,2,3,6,12]:
                        sig = get_combined_signal(alpha_tr,train_ret,sel,lb,phl)
                        if sig is not None:
                            res = strategy_from_signal(sig, train_ret)
                            if res: all_configs.append({'sharpe':res['sharpe'],'sel':sel,'lb':lb,'phl':phl})

        if not all_configs: continue
        all_configs.sort(key=lambda x: x['sharpe'], reverse=True)
        top_k = all_configs[:ENSEMBLE_K]

        # Test: ensemble top-K combined signals
        a1h_te = build_1h_alphas(test_1h)
        te_2h = df_2h.loc[str(test_start):str(test_end)]
        te_4h = df_4h.loc[str(test_start):str(test_end)]
        te_8h = df_8h.loc[str(test_start):str(test_end)]
        a2h_te = build_htf_signals(te_2h, test_1h, 'h2', shift_n=1)
        a4h_te = build_htf_signals(te_4h, test_1h, 'h4', shift_n=1)
        a8h_te = build_htf_signals(te_8h, test_1h, 'h8', shift_n=1)
        across_te = build_cross_asset_all(all_1h, sym, test_1h)
        all_a_te = {**a1h_te, **a2h_te, **a4h_te, **a8h_te, **across_te}
        alpha_te = pd.DataFrame(all_a_te, index=test_1h.index).shift(1)

        ensemble_sigs = []
        for cfg in top_k:
            sig = get_combined_signal(alpha_te, test_ret, cfg['sel'], cfg['lb'], cfg['phl'])
            if sig is not None: ensemble_sigs.append(sig)

        if len(ensemble_sigs) < 2: continue
        avg_sig = pd.concat(ensemble_sigs, axis=1).mean(axis=1)
        mte = strategy_from_signal(avg_sig, test_ret)
        if mte:
            fold_pnls.append(mte['pnl_series'])
            fold_details.append({'fold':test_start.strftime('%Y-%m'),**mte})

    if fold_pnls:
        combined_pnl=pd.concat(fold_pnls).sort_index()
        daily_comb=combined_pnl.resample('1D').sum();daily_comb=daily_comb[daily_comb!=0]
        overall_sh=daily_comb.mean()/daily_comb.std()*np.sqrt(365) if len(daily_comb)>10 else 0
        all_pnl[sym]=combined_pnl
        total=sum(f['net_pnl_bps'] for f in fold_details)
        neg_m=sum(1 for f in fold_details if f['sharpe']<0)
        elapsed=time.time()-t0
        print(f'{sym}: Overall={overall_sh:+.2f}  Total={total:+.0f}bps  Neg={neg_m}/{len(fold_details)}  ({elapsed:.0f}s)')
        for f in fold_details:
            print(f'  {f["fold"]}: SR={f["sharpe"]:+.1f} WR={f["win_rate"]:.1%} PnL={f["net_pnl_bps"]:+.0f}bps')

if len(all_pnl)>0:
    port_df=pd.DataFrame(all_pnl).fillna(0)
    port=port_df.mean(axis=1)
    daily_port=port.resample('1D').sum();daily_port=daily_port[daily_port!=0]
    coll_sh=daily_port.mean()/daily_port.std()*np.sqrt(365)
    print(f'\nCOLLECTIVE (V9c ensemble-3): {coll_sh:+.2f}')
    for sym in SYMBOLS:
        if sym in all_pnl:
            d=all_pnl[sym].resample('1D').sum();d=d[d!=0]
            sh=d.mean()/d.std()*np.sqrt(365) if len(d)>10 else 0
            print(f'  {sym}: {sh:+.2f}')
    port_df.to_parquet('data/v9c_pnl.parquet')
