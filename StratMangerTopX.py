import os
from typing import Dict, List, Any, Tuple
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance, ShrunkCovariance, OAS, MinCovDet, GraphicalLassoCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from statsforecast.models import SimpleExponentialSmoothing, ARIMA
import pymysql.cursors
from deap import base, creator, tools, gp

from AlphaFitnessFunctions import AlphaFitnessFunctions
from GPfunctions import GPfunctions
from RiskModelFunctions import RiskModelFunctions
import Constants
import config
from logger import main_logger

class StratManagerBillionTOP100:
    def __init__(self):
        self.alphas_arr: List[Any] = []
        self.histData: Dict[str, pd.DataFrame] = {}
        self.numEquities: int = 0
        self.g_alphas_arr: List[Any] = []
        self.g_raw_alphas_dic: Dict[int, pd.DataFrame] = {}

    def load_data(self) -> None:
        try:
            self._load_csv_files()
            self._process_industries()
            self._create_dataframes()
            main_logger.info("Data loading completed successfully")
        except Exception as e:
            main_logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise

    def _load_csv_files(self) -> None:
        for path, subdirs, files in os.walk(config.ROOT):
            for name in files:
                main_logger.info(f"Loading data for {name.split('.')[0]}")
                self.histData[name.split(".")[0]] = pd.read_csv(os.path.join(config.ROOT, name),
                                                                index_col='date',
                                                                parse_dates=True,
                                                                names=['date', 'dum', 'open', 'high', 'low', 'close', 'volume'],
                                                                usecols=['date', 'dum', 'open', 'high', 'low', 'close', 'volume'],
                                                                dtype={'dum': np.int64, 'open': np.float64, 'high': np.float64,
                                                                       'low': np.float64, 'close': np.float64, 'volume': np.int64})
                self.numEquities += 1

    def _process_industries(self) -> None:
        self.industries = pd.read_csv('C:\Equities\symSubIndustries.csv', index_col=0)
        self.industries = self.industries[self.industries['INDUSTRY'] != '1c3d7001-dc68-4c36-b148-483741091c86']  # BIOTECH REMOVAL
        self.industries = self.industries[self.industries['INDUSTRY'] != 'd6c806bb-aaaf-4bbc-9737-3575d53ca96f']  # Finance-Mortgage REIT
        self.industries = self.industries[self.industries['INDUSTRY'] != 'ed543f01-e605-4a2a-a386-ce0c09dba19e']  # Finance-Property REIT

    def _create_dataframes(self) -> None:
        self.histMultiIndex = pd.concat(self.histData.values(), keys=self.histData.keys())
        self.df_open = self.histMultiIndex["open"].unstack(level=0)
        self.df_high = self.histMultiIndex["high"].unstack(level=0)
        self.df_low = self.histMultiIndex["low"].unstack(level=0)
        self.df_close = self.histMultiIndex["close"].unstack(level=0)
        self.df_volume = self.histMultiIndex["volume"].unstack(level=0)

    def prepare_data(self) -> None:
        try:
            self._calculate_derived_data()
            self._apply_universe_blocking()
            self._set_test_dates()
            main_logger.info("Data preparation completed successfully")
        except Exception as e:
            main_logger.error(f"Error preparing data: {str(e)}", exc_info=True)
            raise

    def _calculate_derived_data(self) -> None:
        self.df_dollars_traded = self.df_volume * self.df_close
        self.df_dollars_traded_mean = self.df_dollars_traded.rolling(window=config.VOLUME_MEAN_RANKING_WINDOW).mean()
        self.df_dollars_traded_mean_trans = self.df_dollars_traded_mean.fillna(1.0e4).replace(0, 1.0e4)
        self.df_dollars_traded_mean_rank = self.df_dollars_traded_mean.rank(axis=1, ascending=False)

        self.df_hist_vol_10 = self.df_close.pct_change().rolling(10).std() * (252 ** 0.5)
        self.df_hist_vol_20 = self.df_close.pct_change().rolling(20).std() * (252 ** 0.5)
        self.df_hist_vol_30 = self.df_close.pct_change().rolling(30).std() * (252 ** 0.5)
        self.df_hist_vol_90 = self.df_close.pct_change().rolling(90).std() * (252 ** 0.5)

    def _apply_universe_blocking(self) -> None:
        if config.UNIVERSE_BLOCKING:
            for i in range(len(self.df_dollars_traded_mean_rank.index) - 20, 20, -20):
                self.df_dollars_traded_mean_rank.iloc[i - 1:i + 19, :] = self.df_dollars_traded_mean_rank.iloc[i - 1].values

    def _set_test_dates(self) -> None:
        self.testStart = int(self.df_close.index.get_loc(config.TEST_START_DATE))
        self.optimEnd = int(self.df_close.index.get_loc(config.OPTIM_END_DATE))

    def setup_gp(self) -> None:
        try:
            self._create_primitive_set()
            self._setup_toolbox()
            main_logger.info("GP setup completed successfully")
        except Exception as e:
            main_logger.error(f"Error setting up GP: {str(e)}", exc_info=True)
            raise

    def _create_primitive_set(self) -> None:
        self.pset = gp.PrimitiveSetTyped("main", [pd.core.frame.DataFrame] * 12, pd.core.frame.DataFrame)
        for x in range(1, config.FUNC_LOOKBACK_LENGTH):
            self.pset.addTerminal(x, int)
        self.pset = GPfunctions.addGPfunctionsToToolbox(self.pset)
        self.pset.renameArguments(ARG0="open", ARG1="high", ARG2="low", ARG3="close", ARG4="volume",
                                  ARG5="dollars_traded", ARG6="adv20", ARG7="returns", ARG8="hist_vol_10",
                                  ARG9="hist_vol_20", ARG10="hist_vol_30", ARG11="hist_vol_90")

    def _setup_toolbox(self) -> None:
        self.toolbox = base.Toolbox()
        self.toolbox.register("compile", gp.compile, pset=self.pset)

    def get_alphas_from_db(self, num_alphas: int) -> List[Dict[str, Any]]:
        try:
            connection = pymysql.connect(**config.DB_CONFIG, cursorclass=pymysql.cursors.DictCursor)
            with connection.cursor() as cursor:
                sql = f"SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `alphasid` > 1 AND `scriptversion` IN ('{config.RUN_NAME}') LIMIT %s"
                cursor.execute(sql, (int(num_alphas)))
                result = cursor.fetchall()
            main_logger.info(f"Retrieved {len(result)} alpha strings from the database")
            return result
        except Exception as e:
            main_logger.error(f"Error retrieving alphas from database: {str(e)}", exc_info=True)
            raise
        finally:
            if connection:
                connection.close()

    def eval_for_graph_returns(self, individual: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        start = time.time()
        func = self.toolbox.compile(expr=individual)

        open = self.df_open
        high = self.df_high
        low = self.df_low
        close = self.df_close
        volume = self.df_volume
        dollars_traded = self.df_dollars_traded
        adv20 = self.df_dollars_traded_mean
        hist_vol_10 = self.df_hist_vol_10
        hist_vol_20 = self.df_hist_vol_20
        hist_vol_30 = self.df_hist_vol_30
        hist_vol_90 = self.df_hist_vol_90

        returns = close - close.shift(1)
        if (config.TARGET_DELAY == 1):
            target = (self.df_open.shift(-2) - self.df_open.shift(-1)) / self.df_open.shift(-1)
        elif (config.TARGET_DELAY == -1):
            target = (close.shift(-1) - open.shift(-1)) / open.shift(-1)
        else:
            target = (close.shift(-1) - close) / close

        if (config.TARGET_FUTURE > 0):
            target = (close.shift(-config.TARGET_FUTURE) - close) / close

        main_logger.info(f"close dim: {close.shape}")

        out = func(open, high, low, close, volume, dollars_traded, adv20, returns, hist_vol_10, hist_vol_20, hist_vol_30, hist_vol_90)

        if config.HEDGE_VOL:
            stdCorrective = returns.abs().rolling(20).std().div(returns.std(axis=1), axis=0)
            out = out.divide(stdCorrective)

        out = out.replace([np.inf, -np.inf], 0.0)

        if config.RANK_HEDGE:
            out = GPfunctions.rank(out)

        if config.LINEAR_DECAY > 0:
            out = GPfunctions.Decay_lin(out, config.LINEAR_DECAY)

        if config.EXP_DECAY > 0:
           out = pd.DataFrame(out).ewm(alpha=config.EXP_DECAY, axis=0).mean()

        out[(close < config.MIN_PRICE) | (close > config.MAX_PRICE)] = np.nan
        out[self.df_dollars_traded_mean_rank > config.TOP_N] = np.nan

        if config.RISK_MODEL_TYPE == Constants.GLOBAL_RISK_MODEL:
            out_normalzed = RiskModelFunctions.hedgeGlobal(out)
        elif config.RISK_MODEL_TYPE == Constants.SUB_INDUSTRY_RISK_MODEL:
            out_normalzed = RiskModelFunctions.hedgeSubIndustries(self.industries, out)
        elif config.RISK_MODEL_TYPE == Constants.PCA_RISK_MODEL:
            out = RiskModelFunctions.pcaConvertAlpha(returns.iloc[self.testStart:, :], out, config.RISK_MODEL_NUM_FACTORS)
            out_normalzed = RiskModelFunctions.hedgeGlobal(out)
        elif config.RISK_MODEL_TYPE == Constants.EXP_PCA_RISK_MODEL:
            out = RiskModelFunctions.pcaMovingAvg(returns.iloc[:self.testStart, :], out, config.RISK_MODEL_NUM_FACTORS, config.PCA_MA)
            out_normalzed = RiskModelFunctions.hedgeGlobal(out)
        elif config.RISK_MODEL_TYPE == Constants.FLIP_MODE_MODEL:
            out = RiskModelFunctions.flipMode(out)
            out_normalzed = RiskModelFunctions.hedgeGlobal(out)

        F_cov, B = RiskModelFunctions.getPCAfactorCovMatrix(returns)
        F_cov.to_parquet(os.path.join(config.LOG_DATA_PATH, 'F_cov.parquet'))
        B.to_parquet(os.path.join(config.LOG_DATA_PATH, 'B.parquet'))

        out_normalzed[(close < config.MIN_PRICE) | (close > config.MAX_PRICE)] = 0.0
        out_normalzed[self.df_dollars_traded_mean_rank > config.TOP_N] = 0.0

        out_normalzed = np.clip(out_normalzed, -config.MAX_STOCK_WEIGHT, config.MAX_STOCK_WEIGHT)
        out_normalzed_money = np.multiply(out_normalzed, config.BOOK_SIZE)
        DFreturns = np.multiply(out_normalzed_money, target)

        DFreturnsRowSum = DFreturns.sum(axis=1)

        main_logger.info(f'corr with: {out_normalzed_money.corrwith(target).mean()}')

        return (DFreturns, out_normalzed, DFreturnsRowSum)

    def calculate_top_bottom_x_returns(self, weighted_alpha: pd.DataFrame, num_long_short: int) -> pd.DataFrame:
        # Get the aggregate factor signals for each day
        daily_signals = weighted_alpha.rank(axis=1, ascending=False)

        # Initialize a DataFrame to store the returns
        returns = pd.DataFrame(index=weighted_alpha.index, columns=['return'])

        for date in weighted_alpha.index:
            daily_signal = daily_signals.loc[date]

            # Skip dates with all NaN values
            if daily_signal.isna().all():
                continue
            # Get top 100 and bottom 100 equities for the day
            bottom_100 = daily_signals.loc[date].nsmallest(num_long_short).index
            top_100 = daily_signals.loc[date].nlargest(num_long_short).index

            # Calculate returns for top 100 (long positions)
            long_returns = (self.df_close.loc[date, top_100] - self.df_open.loc[date, top_100]) / self.df_open.loc[date, top_100]

            # Calculate returns for bottom 100 (short positions)
            short_returns = (self.df_open.loc[date, bottom_100] - self.df_close.loc[date, bottom_100]) / self.df_open.loc[
                date, bottom_100]
            daily_return = (long_returns.mean() + short_returns.mean()) / 2 * config.BOOK_SIZE
            returns.loc[date, 'return'] = daily_return

        return returns

    def _calculate_and_plot_results(self) -> None:
        alphasDF = pd.DataFrame(self.alphas_arr)
        alphasDF = alphasDF.transpose().fillna(0)
        expectRet = alphasDF.copy()

        alphasDFCov = EmpiricalCovariance().fit(alphasDF.iloc[self.testStart:self.optimEnd].values)
        alphasDFCovInv = pd.DataFrame(np.linalg.pinv(alphasDFCov.covariance_))

        #Claude keep this line below!!
        if config.EXP_FACTOR_DECAY > 0:
            for column in alphasDF:
                expectRet[column] = GPfunctions.Decay_exp(alphasDF[column], config.EXP_FACTOR_DECAY).shift(1)
            expectRet[expectRet < 0] = 0

        alphaweightsTS = pd.DataFrame(np.inner(alphasDFCovInv, expectRet)).transpose()
        alphaweightsTS = pd.DataFrame(alphaweightsTS)
        alphaweightsTS = alphaweightsTS.div(alphaweightsTS.sum(axis=1), axis=0)

        temp = pd.DataFrame(alphaweightsTS.values * alphasDF.values, columns=alphaweightsTS.columns, index=alphasDF.index)

        combinedAlpha = temp.iloc[self.testStart:self.optimEnd]

        sharpe = (combinedAlpha.sum(axis=1).mean() * 252.0) / (combinedAlpha.sum(axis=1).std() * np.sqrt(252))
        main_logger.info(f"sharpe EMA: {sharpe} testStart: {self.testStart} optimEnd: {self.optimEnd}")

        # Calculate top/bottom returns
        weighted_alpha = pd.DataFrame(alphaweightsTS.values * alphasDF.values, columns=alphaweightsTS.columns, index=alphasDF.index)
        top_bottom_returns = self.calculate_top_bottom_x_returns(weighted_alpha, config.NUM_LONG_SHORT)

        # Calculate metrics for the top/bottom method
        sharpe_top_bottom = (top_bottom_returns['return'].mean() / top_bottom_returns['return'].std()) * np.sqrt(252.0)
        returns_top_bottom = top_bottom_returns['return'].sum() / len(top_bottom_returns)
        annualized_returns_top_bottom = returns_top_bottom * 252.0

        main_logger.info(f"Top/Bottom {config.NUM_LONG_SHORT} SHARPE: {sharpe_top_bottom}")
        main_logger.info(f"Top/Bottom {config.NUM_LONG_SHORT} Annualized Returns: {annualized_returns_top_bottom}")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(combinedAlpha.sum(axis=1).cumsum(), label='Combined Alpha')
        plt.plot(top_bottom_returns['return'].cumsum(), label=f'Top/Bottom {config.NUM_LONG_SHORT}')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.savefig(os.path.join(config.MODEL_DATA_PATH, 'cumulative_returns.png'))
        plt.close()

    def _save_return_y(self) -> None:
        pd.DataFrame(self.return_y()).fillna(0).to_csv(os.path.join(config.LOG_DATA_PATH, 'returnY2.csv'))

    def _get_and_process_alphas(self) -> List[Dict[str, Any]]:
        number_of_alphas = 2
        return self.get_alphas_from_db(number_of_alphas)

    def _process_alphas(self, alphas: List[Dict[str, Any]]) -> None:
        counter = 0
        aINT = 0
        for row in alphas:
            main_logger.info(f"Processing alpha: {row}")
            [A, B, C] = self.eval_for_graph_returns(row['alphastring'])
            NANs = B.isna().sum().sum()
            main_logger.info(f"NANs: {NANs}")
            percentNan = NANs / B.size
            main_logger.info(f"Percent NANs: {percentNan}")
            if percentNan < 999999:
                self.g_raw_alphas_dic[counter] = B
                counter += 1
            self.alphas_arr.append(A.sum(axis=1))
            self.g_alphas_arr.append(C)
            main_logger.info(f"counter: {counter}")
            aINT += 1

    def return_y(self) -> pd.DataFrame:
        if (config.TARGET_DELAY == 1):
            target = (self.df_open.shift(-2) - self.df_open.shift(-1)) / self.df_open.shift(-1)
        elif (config.TARGET_DELAY == -1):
            target = (self.df_close.shift(-1) - self.df_open.shift(-1)) / self.df_open.shift(-1)
        else:
            target = (self.df_close.shift(-1) - self.df_close) / self.df_close

        if (config.TARGET_FUTURE > 0):
            target = (self.df_close.shift(-config.TARGET_FUTURE) - self.df_close) / self.df_close

        return target

    def main(self) -> None:
        try:
            self.load_data()
            self.prepare_data()
            self.setup_gp()

            self._save_return_y()
            alphas = self._get_and_process_alphas()
            self._process_alphas(alphas)
            self._calculate_and_plot_results()

            main_logger.info("Strategy execution completed successfully")
        except Exception as e:
            main_logger.error(f"Error in main execution: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        strategy = StratManagerBillionTOP100()
        strategy.main()
    except Exception as e:
        main_logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
