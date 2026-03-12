"""
V10: Ensemble of top-K configs per fold + per-symbol train window optimization
Key idea: instead of picking best-1 config, average top-5 configs for more stable OOS.
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

def build_mtf_alphas_safe(df_1h, df_4h):
    close=df_1h['close'];volume=df_1h['volume'];high=df_1h['high'];low=df_1h['low'];opn=df_1h['open']
    qv=df_1h['quote_volume'];tbv=df_1h['taker_buy_volume']
    ret=close.pct_change();log_ret=np.log(close/close.shift(1))
    vwap=safe_div(qv,volume);taker_ratio=safe_div(tbv,volume)
    rvol_short=ret.rolling(6,min_periods=2).std()
    rvol_long=ret.rolling(72,min_periods=2).std()
    close_pos=safe_div(close-low,high-low)
    body=close-opn;candle_range=high-low
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
        plus_di=safe_div(plus_dm.rolling(w).mean(),atr);minus_di=safe_div(minus_dm.rolling(w).mean(),atr)
        alphas[f'trend_{w}']=plus_di-minus_di
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
    for w in [10,20,30,48]:
        alphas[f'obv_{w}']=-ts_zscore(obv,w)
    for w in [5,10,20,30]:
        alphas[f'tbr_{w}']=ts_zscore(taker_ratio,w)
    timb=safe_div(tbv-(volume-tbv),volume)
    for w in [5,10,20,30]:
        alphas[f'timb_{w}']=ts_zscore(timb,w)
    for w in [5,10,20]:
        vw=(ret*volume).rolling(w).sum();vs=volume.rolling(w).sum()
        alphas[f'vwret_{w}']=safe_div(vw,vs)
    for w in [5,10,20]:
        alphas[f'body_{w}']=ts_zscore(safe_div(body,candle_range),w)
    uwick=high-np.maximum(close,opn);lwick=np.minimum(close,opn)-low
    rejection=safe_div(uwick-lwick,candle_range)
    for w in [5,10]: alphas[f'reject_{w}']=-ts_zscore(rejection,w)
    atr_v=(high-low).rolling(14,min_periods=2).mean()
    for w in [5,10,20]: alphas[f'atr_mr_{w}']=-safe_div(delta(close,w),atr_v)
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
        rs=safe_div(gain,loss);rsi=100-100/(1+rs)
        alphas[f'rsi_{w}']=-(rsi-50)/50
    for w in [14,21]:
        lowest=ts_min(low,w);highest=ts_max(high,w)
        alphas[f'stoch_{w}']=-(safe_div(close-lowest,highest-lowest)*100-50)/50
    for w in [14,20]:
        tp=(high+low+close)/3;tp_sma=sma(tp,w)
        md=tp.rolling(w).apply(lambda x:np.abs(x-x.mean()).mean(),raw=True)
        alphas[f'cci_{w}']=-safe_div(tp-tp_sma,0.015*md)
    # 4H  
    c4=df_4h['close'].shift(1);h4=df_4h['high'].shift(1);l4=df_4h['low'].shift(1)
    lr4=np.log(c4/c4.shift(1));cp4=safe_div(c4-l4,h4-l4)
    for w in [3,5,8,12,20,30]:
        sig=(-ts_zscore(c4,w)).reindex(df_1h.index,method='ffill')
        alphas[f'h4_mr_{w}']=sig
    for w in [3,5,8,12,20]:
        sig=ts_sum(lr4,w).reindex(df_1h.index,method='ffill')
        alphas[f'h4_mom_{w}']=sig
    for w in [12,24,48,72,120]:
        h_=ts_max(h4,w);l_=ts_min(l4,w);rng=h_-l_
        sig=(safe_div(c4-l_,rng)*2-1).reindex(df_1h.index,method='ffill')
        alphas[f'h4_brk_{w}']=sig
    for mw in [3,6,12]:
        for dc in [0.9,0.95,0.98]:
            sig=decay_exp((c4-c4.shift(mw))*cp4,dc).reindex(df_1h.index,method='ffill')
            alphas[f'h4_dec{dc}_d{mw}_cp']=sig
    alpha_df=pd.DataFrame(alphas,index=df_1h.index).shift(1)
    return alpha_df

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
            if abs(sig.corr(alpha_matrix[sel['name']]))>corr_cutoff:
                too_corr=True;break
        if not too_corr: selected.append(r)
        if len(selected)>=max_n: break
    return selected

def strategy_adaptive_net(alpha_matrix,returns,selected,lookback=720,phl=1):
    cols=[s['name'] for s in selected if s['name'] in alpha_matrix.columns]
    if len(cols)<2: return None
    X=alpha_matrix[cols].copy();ret=returns.copy()
    valid=X.dropna().index.intersection(ret.dropna().index)
    X,ret=X.loc[valid],ret.loc[valid]; n=len(valid)
    if n<lookback+24: return None
    factor_returns=pd.DataFrame(index=valid,columns=cols,dtype=float)
    for col in cols:
        d=np.sign(X[col].values);factor_returns[col]=d*ret.values
    rolling_er=factor_returns.rolling(lookback,min_periods=min(100,lookback)).mean()
    weights=rolling_er.clip(lower=0)
    wsum=weights.sum(axis=1).replace(0,np.nan)
    weights_norm=weights.div(wsum,axis=0).fillna(0)
    if phl>1:
        weights_smooth=weights_norm.ewm(halflife=phl,min_periods=1).mean()
        wsum2=weights_smooth.sum(axis=1).replace(0,np.nan)
        weights_smooth=weights_smooth.div(wsum2,axis=0).fillna(0)
    else: weights_smooth=weights_norm
    combined=(X*weights_smooth).sum(axis=1)
    direction=np.sign(combined.values)
    pos_changes=np.abs(np.diff(np.concatenate([[0],direction])))
    pnl=direction*ret.values-FEE_FRAC*pos_changes
    pnl_s=pd.Series(pnl,index=valid)
    daily=pnl_s.resample('1D').sum();daily=daily[daily!=0]
    if len(daily)<5 or daily.std()==0: return None
    sh=daily.mean()/daily.std()*np.sqrt(365)
    return {'sharpe':sh,'net_pnl_bps':pnl.sum()*10000,'pnl_series':pnl_s,
            'win_rate':(pnl[pnl!=0]>0).sum()/max((pnl!=0).sum(),1)}

def get_direction_series(alpha_matrix, returns, selected, lookback, phl):
    """Return the direction time series for a given config (for ensembling)."""
    cols=[s['name'] for s in selected if s['name'] in alpha_matrix.columns]
    if len(cols)<2: return None
    X=alpha_matrix[cols].copy();ret=returns.copy()
    valid=X.dropna().index.intersection(ret.dropna().index)
    X,ret=X.loc[valid],ret.loc[valid]; n=len(valid)
    if n<lookback+24: return None
    factor_returns=pd.DataFrame(index=valid,columns=cols,dtype=float)
    for col in cols:
        d=np.sign(X[col].values);factor_returns[col]=d*ret.values
    rolling_er=factor_returns.rolling(lookback,min_periods=min(100,lookback)).mean()
    weights=rolling_er.clip(lower=0)
    wsum=weights.sum(axis=1).replace(0,np.nan)
    weights_norm=weights.div(wsum,axis=0).fillna(0)
    if phl>1:
        weights_smooth=weights_norm.ewm(halflife=phl,min_periods=1).mean()
        wsum2=weights_smooth.sum(axis=1).replace(0,np.nan)
        weights_smooth=weights_smooth.div(wsum2,axis=0).fillna(0)
    else: weights_smooth=weights_norm
    combined=(X*weights_smooth).sum(axis=1)
    return combined

print('='*70)
print('V10: ENSEMBLE TOP-K CONFIGS (K=5)')
print('='*70)

ENSEMBLE_K = 5

all_pnl={}
for sym in SYMBOLS:
    t0=time.time()
    df15=pd.read_parquet(f'{DATA_DIR}/{sym}.parquet')
    df15=df15.set_index('datetime').sort_index()
    df15=df15[~df15.index.duplicated(keep='last')]
    for c in ['open','high','low','close','volume','quote_volume','taker_buy_volume','taker_buy_quote_volume']:
        df15[c]=pd.to_numeric(df15[c],errors='coerce')
    df_1h=df15.resample('1h').agg({
        'open':'first','high':'max','low':'min','close':'last',
        'volume':'sum','quote_volume':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum'
    }).dropna()
    df_4h=df15.resample('4h').agg({
        'open':'first','high':'max','low':'min','close':'last',
        'volume':'sum','quote_volume':'sum','taker_buy_volume':'sum','taker_buy_quote_volume':'sum'
    }).dropna()
    returns=df_1h['close'].pct_change()

    months=pd.date_range(start='2024-06-01',end='2025-04-01',freq='MS')
    fold_pnls=[];fold_details=[]

    for i in range(len(months)-1):
        test_start=months[i];test_end=months[i+1]
        train_start=test_start-pd.DateOffset(months=8)
        train_1h=df_1h.loc[str(train_start):str(test_start)]
        test_1h=df_1h.loc[str(test_start):str(test_end)]
        train_4h=df_4h.loc[str(train_start):str(test_start)]
        test_4h=df_4h.loc[str(test_start):str(test_end)]
        if len(train_1h)<2000 or len(test_1h)<400: continue
        train_ret=returns.loc[train_1h.index]
        test_ret=returns.loc[test_1h.index]

        alpha_tr=build_mtf_alphas_safe(train_1h,train_4h)
        results=[]
        for col in alpha_tr.columns:
            m=eval_alpha(alpha_tr[col],train_ret)
            if m and m['nofee_sharpe']>0:
                results.append({'name':col,**m})
        results.sort(key=lambda x:x['nofee_sharpe'],reverse=True)
        if len(results)<3: continue

        # Collect top-K configs
        top_configs = []
        for cc in [0.50,0.60,0.70,0.80]:
            for mn in [6,8,10,12,15,20]:
                sel=select_orth(results,alpha_tr,max_n=mn,corr_cutoff=cc)
                if len(sel)<3: continue
                for lb in [180,240,360,480,720]:
                    for phl in [1,3,6,12]:
                        mt=strategy_adaptive_net(alpha_tr,train_ret,sel,lookback=lb,phl=phl)
                        if mt:
                            top_configs.append({'sharpe':mt['sharpe'],'sel':sel,'lb':lb,'phl':phl})

        if not top_configs: continue
        top_configs.sort(key=lambda x: x['sharpe'], reverse=True)
        top_k = top_configs[:ENSEMBLE_K]

        # Test: ensemble top-K by averaging their combined signals
        alpha_te=build_mtf_alphas_safe(test_1h,test_4h)
        
        ensemble_signals = []
        for cfg in top_k:
            sig = get_direction_series(alpha_te, test_ret, cfg['sel'], cfg['lb'], cfg['phl'])
            if sig is not None:
                ensemble_signals.append(sig)
        
        if len(ensemble_signals) < 2: continue
        
        # Average the raw combined signals, then take sign
        avg_sig = pd.concat(ensemble_signals, axis=1).mean(axis=1)
        valid_idx = avg_sig.dropna().index.intersection(test_ret.dropna().index)
        direction = np.sign(avg_sig.loc[valid_idx].values)
        ret_vals = test_ret.loc[valid_idx].values
        pos_changes = np.abs(np.diff(np.concatenate([[0], direction])))
        pnl = direction * ret_vals - FEE_FRAC * pos_changes
        pnl_s = pd.Series(pnl, index=valid_idx)
        daily = pnl_s.resample('1D').sum(); daily = daily[daily != 0]
        if len(daily) < 3 or daily.std() == 0: continue
        sh = daily.mean() / daily.std() * np.sqrt(365)
        wr = (pnl[pnl!=0]>0).sum()/max((pnl!=0).sum(),1)
        
        fold_pnls.append(pnl_s)
        fold_details.append({'fold':test_start.strftime('%Y-%m'),'sharpe':sh,
                           'net_pnl_bps':pnl.sum()*10000,'win_rate':wr})

    if fold_pnls:
        combined_pnl=pd.concat(fold_pnls).sort_index()
        daily_comb=combined_pnl.resample('1D').sum()
        daily_comb=daily_comb[daily_comb!=0]
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
    print(f'\nCOLLECTIVE (V10 ensemble): {coll_sh:+.2f}')
    for sym in SYMBOLS:
        if sym in all_pnl:
            d=all_pnl[sym].resample('1D').sum();d=d[d!=0]
            sh=d.mean()/d.std()*np.sqrt(365) if len(d)>10 else 0
            print(f'  {sym}: {sh:+.2f}')
    port_df.to_parquet('data/v10_pnl.parquet')
