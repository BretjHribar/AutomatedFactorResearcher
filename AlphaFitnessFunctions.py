import pandas as pd
import numpy as np
import math
import scipy
from scipy.stats import norm

# universal constants
gamma = 0.5772156649015328606
e = np.exp(1)

class AlphaFitnessFunctions:
    @staticmethod
    def probabalisticSharpeRatio(strategyReturns, benchmarkSharpeRatio):
        if math.isnan(strategyReturns.std()) or strategyReturns.std() == 0.0:
            return 0.0

        benchmarkSharpeRatioAdj = benchmarkSharpeRatio / math.sqrt(253)
        observedSharpeRatio = strategyReturns.mean() / strategyReturns.std()
        skewness = strategyReturns.skew()
        kurtosis = strategyReturns.kurtosis()

        operandA = skewness * observedSharpeRatio
        operandB = ((kurtosis - 1) / 4) * (math.pow(observedSharpeRatio, 2))

        estimateStandardDeviation = math.pow((1 - operandA + operandB) / (len(strategyReturns) - 1), 0.5)

        value = (observedSharpeRatio - benchmarkSharpeRatioAdj) / estimateStandardDeviation

        PSR = scipy.stats.norm.cdf(value)

        # print("strategyReturns", strategyReturns)
        # print("strategyReturns.mean()", strategyReturns.mean())
        # print("strategyReturns.std()", strategyReturns.std())
        # print("len(strategyReturns) :", len(strategyReturns))
        # print("observedSharpeRatio", observedSharpeRatio)
        # print("estimateStandardDeviation", estimateStandardDeviation)
        # print("benchmarkSharpeRatioAdj", benchmarkSharpeRatioAdj)
        print("PSR", PSR)

        return PSR

    @staticmethod
    # analytical formula for expected maximum sharpe ratio
    def approximate_expected_maximum_sharpe(mean_sharpe, var_sharpe, nb_trials):
        return mean_sharpe + np.sqrt(var_sharpe) * (
                (1 - gamma) * norm.ppf(1 - 1 / nb_trials)
                + gamma * norm.ppf(1 - ((1 / nb_trials) * (1/e)))  )

    @staticmethod
    def compute_deflated_sharpe_ratio(estimated_sharpe, mean_sharpe, sharpe_variance, nb_trials,
                                      backtest_horizon,
                                      skew,
                                      kurtosis):
        SR0 = AlphaFitnessFunctions.approximate_expected_maximum_sharpe(mean_sharpe, sharpe_variance, nb_trials)

        return norm.cdf(((estimated_sharpe - SR0) * np.sqrt(backtest_horizon - 1))
                        / np.sqrt(1 - skew * estimated_sharpe + ((kurtosis - 1) / 4) * estimated_sharpe ** 2))

    @staticmethod
    def calculateNanPercentage():
        return True

    @staticmethod
    def calculateSliceReturnsCutoff():
        return True