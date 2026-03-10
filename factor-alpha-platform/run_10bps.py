import argparse
from eval_portfolio import load_raw_alpha_signals, load_alpha_signals, strategy_regime_net, VAL_START, VAL_END, plot_portfolio_robustness

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mw", type=float, default=0.015)
    parser.add_argument("--lb", type=int, default=260)
    parser.add_argument("--hl", type=int, default=45)
    args = parser.parse_args()

    raw, ret, cl, uv = load_raw_alpha_signals()
    n = len(raw)
    print(f"\nRunning 'regime_net' on {n} alphas (RAW), validation ({VAL_START} to {VAL_END}), 10.0bps fees")
    
    # 1. We must run it with the internal weighting logic using 10.0bps
    result, label = strategy_regime_net(raw, ret, cl, uv, max_wt=args.mw, lookback=args.lb, ema_halflife=args.hl, fee_bps=10.0, decay=2)

    # BUT we also want to simulate the final resulting portfolio under a 10bps environment
    # eval_portfolio sets defaults to 5.0 in its main entrypoints, so let's update strategy_regime_net 
    # to accept and pass fees_bps to simulate(). Let's mock it momentarily by re-running simulate directly:
    
