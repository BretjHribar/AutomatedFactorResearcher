from AlgorithmImports import *

from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd


# QuantConnect single-file version of the current IB paper MOC equity strategy.
# This version intentionally omits the QP layer. It mirrors:
#   pre-subindustry-neutralize each alpha -> equal-weight combine -> market demean
#   -> L1 scale -> clip at 2% -> target shares -> MOC order diffs.
#
# Notes:
# - The live project uses FMP's MCAP_100M_500M universe, FMP daily VWAP, and
#   FMP subindustry labels. This paste-only QC version uses QC FineFundamental
#   market cap for the same market-cap band, minute bars for today's live
#   bar/VWAP, and Morningstar industry code as the subindustry proxy.
# - For a byte-exact match to the project, upload the FMP matrices/universe as
#   custom data and swap the universe/history builders to read those files.


ALPHAS = [
    (61, "rank(add(add(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),
    (76, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))))"),
    (58, "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(true_divide(subtract(ts_max(high, 5), close), df_max(subtract(ts_max(high, 5), ts_min(low, 5)), 0.01)))))"),
    (75, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))))"),
    (77, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(negative(ts_delta(close, 3))), rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)))))"),
    (78, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), ts_delay(rank(negative(ts_delta(close, 3))), 1)))"),
    (67, "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),
    (73, "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 7)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),
    (51, "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))))"),
    (79, "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 10)))))"),
    (68, "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 5)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))))"),
    (74, "rank(multiply(rank(ts_zscore(volume, 20)), rank(negative(true_divide(close, vwap)))))"),
    (62, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(true_divide(add(high, low), 2.0), 5)))))"),
    (57, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 21)))))"),
    (64, "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 3))"),
    (66, "rank(decay_linear(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), 3))"),
    (52, "rank(multiply(rank(true_divide(dollars_traded, sma(dollars_traded, 20))), rank(negative(true_divide(close, vwap)))))"),
    (65, "rank(decay_linear(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 5)))), 3))"),
    (71, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), rank(negative(true_divide(subtract(close, sma(close, 5)), df_max(stddev(close, 5), 0.001))))))"),
    (72, "rank(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 1)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))))"),
    (63, "rank(multiply(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(close, vwap)))), multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(ts_delta(close, 3))))))"),
    (70, "rank(trade_when(true_divide(volume, sma(volume, 20)), multiply(rank(negative(true_divide(close, vwap))), rank(negative(ts_delta(close, 3)))), 0.0))"),
    (69, "rank(decay_linear(multiply(multiply(rank(ts_delta(true_divide(volume, sma(volume, 20)), 3)), rank(negative(true_divide(close, vwap)))), rank(negative(ts_delta(close, 3)))), 3))"),
    (60, "rank(decay_linear(negative(true_divide(close, vwap)), 5))"),
    (48, "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 3)))))"),
    (90, "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 60)))), 3))"),
    (91, "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.08)), rank(negative(ts_zscore(returns, 21)))), 3))"),
    (92, "rank(decay_linear(multiply(rank(decay_exp(negative(true_divide(close, vwap)), 0.06)), rank(negative(ts_zscore(returns, 21)))), 3))"),
    (106, "rank(multiply(rank(negative(adv20)), rank(decay_linear(negative(log_diff(close)), 5))))"),
    (59, "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 14)), df_max(stddev(close, 14), 0.001))))))"),
    (49, "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(ts_delta(close, 5)))))"),
    (56, "rank(multiply(rank(true_divide(volume, sma(volume, 20))), rank(negative(true_divide(subtract(close, sma(close, 60)), df_max(stddev(close, 60), 0.001))))))"),
    (50, "rank(multiply(rank(true_divide(dollars_traded, sma(dollars_traded, 20))), rank(negative(ts_delta(close, 5)))))"),
    (54, "rank(multiply(rank(true_divide(subtract(high, low), sma(subtract(high, low), 20))), rank(negative(true_divide(close, vwap)))))"),
    (95, "rank(multiply(rank(negative(adv20)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
    (97, "rank(multiply(rank(negative(adv60)), rank(decay_exp(negative(true_divide(close, vwap)), 0.10))))"),
]


def _field(obj, *names, default=None):
    if obj is None:
        return default
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _clean_fundamental_text(value):
    if value in (None, ""):
        return ""
    return str(value).strip().upper()


def _clean_exchange_text(value):
    text = _clean_fundamental_text(value)
    for ch in (" ", "-", "_", ".", "/", "\\", ":"):
        text = text.replace(ch, "")
    return text


class MocSessionBar:
    def __init__(self):
        self.date = None
        self.open = np.nan
        self.high = np.nan
        self.low = np.nan
        self.close = np.nan
        self.volume = 0.0
        self.vwap_num = 0.0
        self.vwap_den = 0.0

    def update(self, time, bar):
        date = time.date()
        open_ = _field(bar, "open", "Open")
        high = _field(bar, "high", "High")
        low = _field(bar, "low", "Low")
        close = _field(bar, "close", "Close")
        volume = _field(bar, "volume", "Volume", default=0)
        if self.date != date:
            self.date = date
            self.open = float(open_)
            self.high = float(high)
            self.low = float(low)
            self.close = float(close)
            self.volume = 0.0
            self.vwap_num = 0.0
            self.vwap_den = 0.0
        self.high = max(self.high, float(high))
        self.low = min(self.low, float(low))
        self.close = float(close)
        vol = float(volume or 0)
        self.volume += vol
        if vol > 0:
            self.vwap_num += float(close) * vol
            self.vwap_den += vol

    @property
    def vwap(self):
        if self.vwap_den > 0:
            return self.vwap_num / self.vwap_den
        if np.isfinite(self.high) and np.isfinite(self.low) and np.isfinite(self.close):
            return (self.high + self.low + self.close) / 3.0
        return np.nan


class IBTieredLikeFeeModel(FeeModel):
    def __init__(self, commission_per_share=0.0045, minimum=0.35, sec_fee_per_dollar=27.80e-6):
        self.commission_per_share = commission_per_share
        self.minimum = minimum
        self.sec_fee_per_dollar = sec_fee_per_dollar

    def GetOrderFee(self, parameters):
        order = _field(parameters, "order", "Order")
        security = _field(parameters, "security", "Security")
        qty = abs(float(_field(order, "quantity", "Quantity", "absolute_quantity", "AbsoluteQuantity", default=0)))
        notional = qty * float(_field(security, "price", "Price", default=0))
        commission = max(self.minimum, qty * self.commission_per_share)
        action = str(_field(order, "direction", "Direction", default="")).lower()
        sec_fee = self.sec_fee_per_dollar * notional if "sell" in action else 0.0
        return OrderFee(CashAmount(float(commission + sec_fee), "USD"))

    def get_order_fee(self, parameters):
        return self.GetOrderFee(parameters)


class IBClosingAuctionNoQP(QCAlgorithm):
    def initialize(self):
        # Change these in the QC UI or here.
        self.set_start_date(2024, 7, 1)
        self.set_end_date(2026, 5, 12)
        self.set_cash(110000)

        self.booksize = 500000.0
        self.max_stock_weight = 0.02
        self.min_order_value = 200.0
        self.min_market_cap = 100_000_000
        self.max_market_cap = 500_000_000
        self.history_days = 400
        self.min_active_symbols = 25
        self.min_live_symbols = 50
        self.leverage = 6.0
        # Project source screener requests NYSE/NASDAQ/AMEX. QC/Morningstar
        # can expose this as exchange ids, display names, or MICs, so accept
        # the common aliases instead of only NYS/NAS/ASE.
        self.allowed_exchange_aliases = {
            "NYS", "NYSE", "XNYS", "NEWYORKSTOCKEXCHANGE",
            "NAS", "NASDAQ", "XNAS", "NASDAQGLOBALSELECTMARKET",
            "NASDAQGLOBALMARKET", "NASDAQCAPITALMARKET",
            "ASE", "AMEX", "XASE", "NYSEAMERICAN", "NYSEMKT",
            "NYSEMARKET", "AMERICANSTOCKEXCHANGE",
        }
        self.allowed_country_ids = {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}
        self.active_company_statuses = {"U", "PUBLIC"}
        self.common_stock_security_types = {"ST00000001", "COMMON STOCK", "COMMON"}
        self.excluded_security_type_terms = ("PREFERRED", "PREF", "UNIT", "RIGHT", "WARRANT")
        self.excluded_common_share_subtype_terms = (
            "PREFERRED",
            "CLOSED-END FUND",
            "FOREIGN SHARE",
            "FOREIGN PARTICIPATED",
        )
        self.excluded_morningstar_industry_codes = {
            10350010,  # ShellCompanies: capital pool / blank check / shell / holding companies
            10420010, 10420020, 10420030, 10420040, 10420050,
            10420060, 10420070, 10420080, 10420090,  # REIT industries
        }
        self.excluded_industry_terms = ("REIT", "SHELL", "BLANK CHECK")

        self.selected_symbols = []
        self.groups = {}
        self.session = defaultdict(MocSessionBar)
        self.last_rebalance_date = None
        self.last_universe_debug_date = None

        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.free_portfolio_value_percentage = 0.02

        self.data_norm = DataNormalizationMode.SPLIT_ADJUSTED
        self.universe_settings.resolution = Resolution.MINUTE
        self.universe_settings.leverage = self.leverage
        self.universe_settings.data_normalization_mode = self.data_norm

        self.spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.add_universe(self.coarse_selection, self.fine_selection)

        MarketOnCloseOrder.submission_time_buffer = timedelta(minutes=10)
        self.schedule.on(
            self.date_rules.every_day(self.spy),
            self.time_rules.before_market_close(self.spy, 20),
            self.rebalance_moc
        )

    def coarse_selection(self, coarse):
        candidates = []
        for c in coarse:
            has_fundamental = bool(_field(c, "has_fundamental_data", "HasFundamentalData", default=False))
            price = _field(c, "price", "Price")
            dollar_volume = _field(c, "dollar_volume", "DollarVolume")
            if has_fundamental and price is not None and price > 1 and dollar_volume is not None and dollar_volume > 0:
                candidates.append(c)
        candidates = sorted(
            candidates,
            key=lambda c: _field(c, "dollar_volume", "DollarVolume", default=0),
            reverse=True,
        )
        return [_field(c, "symbol", "Symbol") for c in candidates]

    def fine_selection(self, fine):
        selected = []
        groups = {}
        reject_counts = defaultdict(int)
        total = 0
        tradable = 0
        for f in fine:
            total += 1
            reject_reason = self.project_universe_reject_reason(f)
            if reject_reason is not None:
                reject_counts[reject_reason] += 1
                continue
            tradable += 1

            market_cap = _field(f, "market_cap", "MarketCap", default=0)
            if market_cap is None:
                market_cap = 0
            if not (self.min_market_cap <= float(market_cap) < self.max_market_cap):
                reject_counts["market_cap_band"] += 1
                continue

            symbol = _field(f, "symbol", "Symbol")
            selected.append((symbol, float(market_cap)))
            groups[symbol] = self._industry_group(f)

        selected = sorted(selected, key=lambda x: x[1], reverse=True)
        self.selected_symbols = [x[0] for x in selected]
        self.groups = {s: groups.get(s, "UNKNOWN") for s in self.selected_symbols}
        if len(self.selected_symbols) < self.min_active_symbols and self.last_universe_debug_date != self.time.date():
            self.last_universe_debug_date = self.time.date()
            top_rejects = ", ".join(f"{k}={v}" for k, v in sorted(
                reject_counts.items(), key=lambda kv: kv[1], reverse=True
            )[:8])
            self.debug(
                f"{self.time.date()} fine universe: input {total}, project-tradable {tradable}, "
                f"selected {len(self.selected_symbols)}; rejects {top_rejects}"
            )
        return self.selected_symbols

    def on_securities_changed(self, changes):
        fee_model = IBTieredLikeFeeModel()
        added = _field(changes, "added_securities", "AddedSecurities", default=[])
        for security in added:
            if hasattr(security, "set_fee_model"):
                security.set_fee_model(fee_model)
            else:
                security.SetFeeModel(fee_model)
            if hasattr(security, "set_leverage"):
                security.set_leverage(self.leverage)
            else:
                security.SetLeverage(self.leverage)
            try:
                if hasattr(security, "set_data_normalization_mode"):
                    security.set_data_normalization_mode(self.data_norm)
                else:
                    security.SetDataNormalizationMode(self.data_norm)
            except Exception:
                pass

    def on_data(self, data):
        bars = _field(data, "bars", "Bars", default={})
        for symbol in self.selected_symbols:
            bar = None
            if hasattr(bars, "get"):
                bar = bars.get(symbol)
            elif hasattr(bars, "ContainsKey") and bars.ContainsKey(symbol):
                bar = bars[symbol]
            if bar is not None:
                self.session[symbol].update(self.time, bar)

    def rebalance_moc(self):
        if self.last_rebalance_date == self.time.date():
            return
        self.last_rebalance_date = self.time.date()

        symbols = [s for s in self.selected_symbols if self.security_for(s) is not None]
        if len(symbols) < self.min_active_symbols:
            self.debug(f"{self.time.date()} skip: only {len(symbols)} symbols selected")
            return

        matrices = self.build_matrices(symbols)
        if not matrices:
            self.debug(f"{self.time.date()} skip: no matrices")
            return
        if matrices["close"].index[-1].date() != self.time.date():
            self.debug(f"{self.time.date()} halt: last signal row is not today's estimated bar")
            return
        live_prices = int(matrices["close"].iloc[-1].notna().sum())
        if live_prices < self.min_live_symbols:
            self.debug(f"{self.time.date()} halt: only {live_prices} symbols have today's estimated live bar")
            return
        self.debug(f"{self.time.date()} delay=0: evaluating alphas on today's estimated bar ({live_prices} live prices)")

        weights = self.compute_combined_signal(matrices, symbols)
        if weights.empty or weights.abs().sum() == 0:
            self.debug(f"{self.time.date()} skip: zero signal")
            return

        close = matrices["close"].iloc[-1]
        target_shares = self.signal_to_target_shares(weights, close)
        order_diffs = self.compute_order_diffs(target_shares, close)
        if not order_diffs:
            self.debug(f"{self.time.date()} no MOC diffs")
            return

        buys = sum(1 for q in order_diffs.values() if q > 0)
        sells = sum(1 for q in order_diffs.values() if q < 0)
        gmv = float((target_shares.abs() * close.reindex(target_shares.index)).sum())
        self.debug(
            f"{self.time.date()} signal {len(target_shares)} pos, GMV ${gmv:,.0f}, "
            f"{len(order_diffs)} MOC orders ({buys} buy/{sells} sell)"
        )

        for symbol, qty in order_diffs.items():
            self.market_on_close_order(symbol, int(qty), tag="IB_MOC_no_QP", asynchronous=True)

    def build_matrices(self, symbols):
        hist = self.history(symbols, self.history_days, Resolution.DAILY)
        if hist is None or hist.empty:
            return {}

        fields = {}
        for name in ("open", "high", "low", "close", "volume"):
            df = self.history_field(hist, name)
            if df is None or df.empty:
                return {}
            df = df.reindex(columns=symbols)
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            df = df[~df.index.duplicated(keep="last")].sort_index()
            fields[name] = df

        today = pd.Timestamp(self.time.date())
        for name in fields:
            fields[name] = fields[name][fields[name].index < today]

        if fields["close"].empty:
            return {}

        current_rows = {name: {} for name in ("open", "high", "low", "close", "volume", "vwap")}
        previous_volume = fields["volume"].iloc[-1]
        for symbol in symbols:
            state = self.session.get(symbol)
            if state is not None and state.date == self.time.date() and np.isfinite(state.close):
                current_rows["open"][symbol] = state.open
                current_rows["high"][symbol] = state.high
                current_rows["low"][symbol] = state.low
                current_rows["close"][symbol] = state.close
                current_rows["volume"][symbol] = previous_volume.get(symbol, np.nan)
                current_rows["vwap"][symbol] = state.vwap
            else:
                for name in current_rows:
                    current_rows[name][symbol] = np.nan

        matrices = {}
        for name in ("open", "high", "low", "close", "volume"):
            row = pd.DataFrame(current_rows[name], index=[today]).reindex(columns=symbols)
            matrices[name] = pd.concat([fields[name], row]).tail(self.history_days)

        hist_vwap = (fields["high"] + fields["low"] + fields["close"]) / 3.0
        vwap_row = pd.DataFrame(current_rows["vwap"], index=[today]).reindex(columns=symbols)
        matrices["vwap"] = pd.concat([hist_vwap, vwap_row]).tail(self.history_days)

        returns = matrices["close"].pct_change(fill_method=None)
        matrices["returns"] = returns.where(returns.abs() <= 0.50)

        # Match prod/live_bar.py: only raw live OHLCV/VWAP are appended for
        # today. Derived fields like dollars_traded/adv20/adv60 are not
        # projected intraday, so they remain unavailable on the live row.
        hist_dollars = fields["close"] * fields["volume"]
        matrices["dollars_traded"] = hist_dollars.reindex(index=matrices["close"].index, columns=symbols)
        matrices["adv20"] = hist_dollars.rolling(20, min_periods=10).mean().reindex(
            index=matrices["close"].index, columns=symbols
        )
        matrices["adv60"] = hist_dollars.rolling(60, min_periods=20).mean().reindex(
            index=matrices["close"].index, columns=symbols
        )
        return matrices

    def compute_combined_signal(self, matrices, symbols):
        env = self.operator_env(matrices)
        groups = pd.Series({s: self.groups.get(s, "UNKNOWN") for s in symbols})

        ref_index = matrices["close"].index
        ref_columns = list(matrices["close"].columns)
        total = pd.DataFrame(0.0, index=ref_index, columns=ref_columns)
        count = pd.DataFrame(0.0, index=ref_index, columns=ref_columns)

        loaded = 0
        for alpha_id, expr in ALPHAS:
            try:
                raw = eval(expr, {"__builtins__": {}}, env)
                raw = pd.DataFrame(raw).reindex(index=ref_index, columns=ref_columns)
                raw = self.group_demean(raw, groups)
                valid = raw.replace([np.inf, -np.inf], np.nan).notna()
                total = total.where(~valid, total + raw.fillna(0.0))
                count = count.where(~valid, count + 1.0)
                loaded += 1
            except Exception as exc:
                self.debug(f"alpha {alpha_id} failed: {exc}")

        if loaded < 25:
            self.debug(f"halt: only {loaded}/{len(ALPHAS)} alphas evaluated")
            return pd.Series(dtype=float)

        avg = total / count.replace(0.0, np.nan)
        processed = self.process_signal(avg, self.max_stock_weight)
        return processed.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def signal_to_target_shares(self, weights, close):
        prices = close.reindex(weights.index).replace(0, np.nan)
        dollars = weights * self.booksize
        shares = (dollars / prices).replace([np.inf, -np.inf], np.nan).fillna(0).round().astype(int)
        return shares[shares != 0]

    def compute_order_diffs(self, target_shares, close):
        diffs = {}
        target_symbols = set(target_shares.index)

        for symbol, qty in target_shares.items():
            current = self.holding_quantity(symbol)
            diff = int(qty) - current
            if diff != 0:
                diffs[symbol] = diff

        for symbol in self.portfolio_symbols():
            qty = self.holding_quantity(symbol)
            if qty != 0 and symbol not in target_symbols:
                diffs[symbol] = -qty

        filtered = {}
        for symbol, qty in diffs.items():
            price = close.get(symbol, np.nan)
            if not np.isfinite(price) or price <= 0:
                security = self.security_for(symbol)
                sec_price = _field(security, "price", "Price", default=0) if security is not None else 0
                if sec_price > 0:
                    price = float(sec_price)
            value = abs(qty) * float(price) if np.isfinite(price) else 0.0
            if value >= self.min_order_value:
                filtered[symbol] = int(qty)
        return filtered

    @staticmethod
    def history_field(hist, field):
        if field not in hist.columns:
            camel = field[:1].upper() + field[1:]
            if camel in hist.columns:
                field = camel
            else:
                return None
        series = hist[field]
        if not isinstance(series.index, pd.MultiIndex):
            return series.to_frame()

        names = [str(n).lower() for n in series.index.names]
        symbol_level = 0 if names[0] == "symbol" else 1
        return series.unstack(level=symbol_level)

    @staticmethod
    def process_signal(alpha_df, max_wt):
        signal = alpha_df.copy().astype(float)
        signal = signal.sub(signal.mean(axis=1), axis=0)
        abs_sum = signal.abs().sum(axis=1).replace(0, np.nan)
        signal = signal.div(abs_sum, axis=0)
        signal = signal.clip(lower=-max_wt, upper=max_wt)
        return signal.fillna(0.0)

    @staticmethod
    def group_demean(signal_df, groups):
        result = signal_df.copy()
        groups = groups.reindex(signal_df.columns)
        for group in groups.dropna().unique():
            cols = groups[groups == group].index
            cols = [c for c in cols if c in result.columns]
            if len(cols) < 2:
                continue
            result[cols] = result[cols].sub(result[cols].mean(axis=1), axis=0)
        return result

    def operator_env(self, matrices):
        env = dict(matrices)
        env.update({
            "np": np,
            "pd": pd,
            "rank": rank,
            "sma": sma,
            "ts_delta": ts_delta,
            "delta": ts_delta,
            "ts_delay": ts_delay,
            "delay": ts_delay,
            "ts_max": ts_max,
            "ts_min": ts_min,
            "stddev": stddev,
            "ts_zscore": ts_zscore,
            "decay_linear": decay_linear,
            "Decay_lin": decay_linear,
            "decay_exp": decay_exp,
            "Decay_exp": decay_exp,
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "true_divide": true_divide,
            "divide": true_divide,
            "negative": negative,
            "df_max": df_max,
            "df_min": df_min,
            "trade_when": trade_when,
            "log_diff": log_diff,
        })
        return env

    def _industry_group(self, fine):
        ac = _field(fine, "asset_classification", "AssetClassification")
        if ac is not None:
            for name in (
                "morningstar_industry_code",
                "MorningstarIndustryCode",
                "morningstar_industry_group_code",
                "MorningstarIndustryGroupCode",
                "morningstar_sector_code",
                "MorningstarSectorCode",
            ):
                value = _field(ac, name)
                if value not in (None, 0, ""):
                    return str(value)
        return "UNKNOWN"

    def is_project_tradable_common_equity(self, fine):
        return self.project_universe_reject_reason(fine) is None

    def project_universe_reject_reason(self, fine):
        """Approximate the project's upstream FMP universe filter in QC.

        Project side:
          - US active stocks on NYSE/NASDAQ/AMEX
          - isEtf=false, isFund=false
          - remove REIT / shell / blank-check industries
          - MCAP band membership is then built from daily market cap + close.notna

        QC Morningstar fundamental universes already exclude ETFs, ADRs, and
        OTC equities. CFDs are not part of the QC US Equity fundamental universe.
        Here we explicitly enforce the US exchange, common-share, active-share,
        depositary-receipt, REIT, shell, and blank-check exclusions that QC
        exposes through CompanyReference/SecurityReference.
        """
        cr = _field(fine, "company_reference", "CompanyReference")
        sr = _field(fine, "security_reference", "SecurityReference")
        ac = _field(fine, "asset_classification", "AssetClassification")

        country = _clean_fundamental_text(_field(cr, "country_id", "CountryId"))
        if country and country not in self.allowed_country_ids:
            return "country"

        company_status = _clean_fundamental_text(_field(cr, "company_status", "CompanyStatus"))
        if company_status and company_status not in self.active_company_statuses:
            return "company_status"

        industry_code = _field(ac, "morningstar_industry_code", "MorningstarIndustryCode")
        try:
            if industry_code not in (None, "") and int(industry_code) in self.excluded_morningstar_industry_codes:
                return "reit_shell_industry"
        except Exception:
            pass

        exchange_values = [
            _field(cr, "primary_exchange_id", "PrimaryExchangeID"),
            _field(cr, "primary_mic", "PrimaryMIC"),
            _field(sr, "exchange_id", "ExchangeId"),
            _field(sr, "mic", "MIC"),
        ]
        exchange_values = [x for x in exchange_values if x not in (None, "")]
        if exchange_values and not any(self.is_allowed_project_exchange(x) for x in exchange_values):
            return "exchange"

        security_type = _field(sr, "security_type", "SecurityType")
        security_type_text = _clean_fundamental_text(security_type)
        if security_type_text:
            if security_type_text.startswith("ST") and security_type_text not in self.common_stock_security_types:
                return "security_type"
            if any(term in security_type_text for term in self.excluded_security_type_terms):
                return "security_type"

        share_status = _clean_fundamental_text(_field(sr, "share_class_status", "ShareClassStatus"))
        if share_status and share_status not in {"A", "ACTIVE"}:
            return "share_status"

        # Do not use SecurityReference.TradingStatus as a proxy for FMP's
        # isActivelyTrading flag. In QC historical fundamentals it is False for
        # many otherwise selected live equities and collapses the universe.

        if bool(_field(sr, "is_depositary_receipt", "IsDepositaryReceipt", default=False)):
            return "depositary_receipt"

        is_reit = _field(cr, "is_reit", "IsREIT", default=False)
        if bool(is_reit):
            return "reit_flag"

        text_fields = [
            _field(cr, "standard_name", "StandardName", default=""),
            _field(cr, "legal_name", "LegalName", default=""),
            _field(cr, "short_name", "ShortName", default=""),
            str(_field(sr, "share_class_description", "ShareClassDescription", default="")),
            str(_field(sr, "common_share_sub_type", "CommonShareSubType", default="")),
        ]
        text = " ".join(_clean_fundamental_text(x) for x in text_fields if x)
        common_share_subtype = _clean_fundamental_text(
            _field(sr, "common_share_sub_type", "CommonShareSubType", default="")
        )
        if any(term in common_share_subtype for term in self.excluded_common_share_subtype_terms):
            return "share_subtype"
        if any(term in text for term in self.excluded_industry_terms):
            return "reit_shell_text"
        return None

    def is_allowed_project_exchange(self, value):
        text = _clean_exchange_text(value)
        if not text:
            return False
        if text in self.allowed_exchange_aliases:
            return True
        if "NASDAQ" in text:
            return True
        if text.startswith("NYSE") or "NEWYORKSTOCKEXCHANGE" in text:
            return True
        if "AMEX" in text or "AMERICANSTOCKEXCHANGE" in text:
            return True
        return False

    def holding_quantity(self, symbol):
        try:
            holding = self.portfolio[symbol]
        except Exception:
            return 0
        return int(_field(holding, "quantity", "Quantity", default=0) or 0)

    def portfolio_symbols(self):
        try:
            return list(self.portfolio.keys())
        except Exception:
            return []

    def security_for(self, symbol):
        try:
            return self.securities[symbol]
        except Exception:
            return None


def _as_df(x, like=None):
    if isinstance(x, pd.DataFrame):
        return x
    if like is not None and isinstance(like, pd.DataFrame):
        return pd.DataFrame(x, index=like.index, columns=like.columns)
    return x


def rank(df):
    return pd.DataFrame(df).rank(axis=1, pct=True)


def sma(df, window=10):
    return pd.DataFrame(df).rolling(int(window), min_periods=1).mean()


def ts_delta(df, period=2):
    return pd.DataFrame(df).diff(int(period))


def ts_delay(df, period=1):
    return pd.DataFrame(df).shift(int(period))


def ts_max(df, window=10):
    return pd.DataFrame(df).rolling(int(window), min_periods=1).max()


def ts_min(df, window=10):
    return pd.DataFrame(df).rolling(int(window), min_periods=1).min()


def stddev(df, window=10):
    return pd.DataFrame(df).rolling(int(window), min_periods=2).std()


def ts_zscore(df, window=10):
    df = pd.DataFrame(df)
    r = df.rolling(window=int(window), min_periods=2)
    mean = r.mean().shift(1)
    std = r.std(ddof=0).shift(1)
    return (df - mean) / std.replace(0, np.nan)


def _rolling_decay_lin(values):
    weights = np.arange(1, values.size + 1)
    return np.sum(weights * values / weights.sum())


def decay_linear(df, window=10):
    return pd.DataFrame(df).rolling(int(window), min_periods=1).apply(_rolling_decay_lin, raw=True)


def decay_exp(df, alpha_exp=0.99):
    alpha = abs(float(alpha_exp))
    if alpha > 1:
        alpha = 0.99
    return pd.DataFrame(df).ewm(alpha=alpha).mean()


def add(left, right):
    return np.add(left, right)


def subtract(left, right):
    return np.subtract(left, right)


def multiply(left, right):
    return np.multiply(left, right)


def true_divide(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(left, right)


def negative(df):
    return np.negative(df)


def df_max(left, right):
    left_df = pd.DataFrame(left)
    if isinstance(right, pd.DataFrame):
        right_df = right.reindex(index=left_df.index, columns=left_df.columns)
    else:
        right_df = pd.DataFrame(right, index=left_df.index, columns=left_df.columns)
    return left_df.where(left_df > right_df, right_df)


def df_min(left, right):
    left_df = pd.DataFrame(left)
    if isinstance(right, pd.DataFrame):
        right_df = right.reindex(index=left_df.index, columns=left_df.columns)
    else:
        right_df = pd.DataFrame(right, index=left_df.index, columns=left_df.columns)
    return left_df.where(left_df < right_df, right_df)


def trade_when(cond, alpha, fallback):
    alpha_df = pd.DataFrame(alpha)
    if isinstance(fallback, pd.DataFrame):
        fallback_df = fallback.reindex(index=alpha_df.index, columns=alpha_df.columns)
    else:
        fallback_df = pd.DataFrame(fallback, index=alpha_df.index, columns=alpha_df.columns)
    return alpha_df.where(pd.DataFrame(cond).astype(bool), fallback_df)


def log_diff(df):
    # Matches the project operator: x - x.shift(1), not log(x).diff().
    return pd.DataFrame(df) - pd.DataFrame(df).shift(1)
