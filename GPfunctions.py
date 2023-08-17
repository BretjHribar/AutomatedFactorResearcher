import pandas as pd
import numpy as np
import random

#from scipy.stats import zscore

class GPfunctions:
    @staticmethod
    def true_divide(left, right):
        return np.true_divide(left, right)

    @staticmethod
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 0
        except ValueError:
            return 0
        except:
            return 0

    @staticmethod
    def ts_sum(df, window=10):
        return df.rolling(window).sum()

    @staticmethod
    def sma(df, window=10):
        return df.rolling(window).mean()

    @staticmethod
    def rank(df):
        df = pd.DataFrame(df)
        #return df.rank(axis=1).divide(df.count(axis=1), axis=0)
        return df.rank(axis=1, pct=True)

    @staticmethod
    def ts_rank(df, window=10):
        return df.rolling(window).rank(pct=True)

    @staticmethod
    def ts_min(df, window=10):
        return df.rolling(window).min()

    @staticmethod
    def ts_max(df, window=10):
        return df.rolling(window).max()

    @staticmethod
    def delta(df, period=2):
        return df.diff(period)

    @staticmethod
    def stddev(df, window=10):
        return df.rolling(window).std()

    @staticmethod
    def correlation(x, y, window=10):
        if (x.equals(y)):
            return pd.DataFrame(0, columns=x.columns, index=x.index)
        else:
            return x.rolling(window).corr(y)

    @staticmethod
    def covariance(x, y, window=10):
        if (x.equals(y)):
            return pd.DataFrame(0, columns=x.columns, index=x.index)
        else:
            return x.rolling(window).cov(y)

    @staticmethod
    def Product(df, window=10):
        return df.rolling(window).apply(lambda na: np.prod(na), raw=True)

    @staticmethod
    def delay(df, period=1):
        return df.shift(period)

    # def scale(df, k=1):
    # """
    # :param k: scaling factor.
    # :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    # """
    # return df.mul(k).div(np.abs(df).sum())

    @staticmethod
    def ArgMax(df, window=10):
        return df.rolling(window).apply(np.argmax, raw=True) + 1

    @staticmethod
    def ArgMin(df, window=10):
        return df.rolling(window).apply(np.argmin, raw=True) + 1

    @staticmethod
    def ts_skewness(df, window=10):
        return df.rolling(window).skew()

    @staticmethod
    def ts_kurtosis(df, window=10):
        return df.rolling(window).kurt()

    @staticmethod
    def extend(i):
        return i

    @staticmethod
    def npfadd(df, f):
        return np.add(df, f)

    @staticmethod
    def npfdiv(df, f):
        return np.divide(df, f)

    @staticmethod
    def npfsub(df, f):
        return np.subtract(df, f)

    @staticmethod
    def npfmul(df, f):
        return np.multiply(df, f)

    @staticmethod
    def SignedPower(df, y):
        return df.pow(abs(y))

    @staticmethod
    def Sign(df):
        return np.sign(df)

    @staticmethod
    def rolling_decay_lin(na):
        linWeights = np.arange(1, na.size + 1)
        return np.sum(np.multiply(linWeights, na) / sum(linWeights))

    @staticmethod
    def Decay_lin(df, window=10):
        return df.rolling(window).apply(GPfunctions.rolling_decay_lin, raw=True)

    @staticmethod
    def Decay_exp(df, alphaExp=0.99):
        if alphaExp > 1 or alphaExp < -1:
            alphaExp = 0.99
        return pd.DataFrame(df).ewm(alpha=abs(alphaExp), axis=0).mean()

    @staticmethod
    def Abs(df):
        return df.abs()

    @staticmethod
    def Tail(df, cutoff):
        TEST = df > -0.5
        print(TEST.shape)
        df[(df > -cutoff) & (df < cutoff)] = 0
        return df

    @staticmethod
    def Inverse(df):
        return 1.0 / df

    #### NEW FUNCTIONS ##################
    @staticmethod
    def ts_zscore(df, window=10):
        r = df.rolling(window=window)
        m = r.mean().shift(1)
        s = r.std(ddof=0).shift(1)
        z = (df - m) / s
        return z

    @staticmethod
    def log_diff(df):
        return df - df.shift(1)

    @staticmethod
    def s_log_1p(df):
        return np.sign(df) * np.log(1 + np.abs(df))

    @staticmethod
    def df_max(left, right):
        return left.where(left > right, right)

    @staticmethod
    def df_min(left, right):
        return left.where(left < right, right)

    @staticmethod
    def zscore(df):
        return df.abs()

    # @staticmethod
    # def ts_regression(y, x, d, lag = 0, rettype = 0)

    #### ADD GP FUNCTIONS TO TOOLBOX #####
    @staticmethod
    def addGPfunctionsToToolbox(pset):
        pset.addEphemeralConstant("rand1", lambda: random.uniform(-10, 10), float)
        pset.addEphemeralConstant("rand2", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant("rand3", lambda: random.uniform(0, 1), float)

        ####NEW######
        pset.addPrimitive(GPfunctions.ts_zscore, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.log_diff, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.s_log_1p, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.df_max, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.df_min, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        #############
        pset.addPrimitive(GPfunctions.npfadd, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.npfsub, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.npfmul, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.npfdiv, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)

        pset.addPrimitive(np.add, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.subtract, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.multiply, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.divide, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.negative, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.log, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.log10, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.sqrt, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.square, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.true_divide, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.SignedPower, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Sign, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.extend, [int], int)
        pset.addPrimitive(GPfunctions.extend, [float], float)
        pset.addPrimitive(GPfunctions.sma, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.delta, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.stddev, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.ts_min, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_max, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_sum, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.delay, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_skewness, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_kurtosis, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.ArgMax, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ArgMin, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Product, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_rank, [pd.core.frame.DataFrame,int], pd.core.frame.DataFrame)
        #pset.addPrimitive(scale, [pd.core.frame.DataFrame,int], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.rank, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Abs, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.Decay_lin, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Decay_exp, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.Inverse, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.correlation, [pd.core.frame.DataFrame, pd.core.frame.DataFrame, int],
                          pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.covariance, [pd.core.frame.DataFrame, pd.core.frame.DataFrame, int],
                          pd.core.frame.DataFrame)

        return pset

    @staticmethod
    def GP_rename_arguments(pset):
        pset.renameArguments(ARG0="open")
        pset.renameArguments(ARG1="high")
        pset.renameArguments(ARG2="low")
        pset.renameArguments(ARG3="close")
        pset.renameArguments(ARG4="volume")
        pset.renameArguments(ARG5="dollars_traded")
        pset.renameArguments(ARG6="adv20")
        pset.renameArguments(ARG7="returns")
        return pset

    #### ADD GP FUNCTIONS TO TOOLBOX #####
    @staticmethod
    def addGPfunctionsToToolboxFromDictionary(pset, func_dic):
        pset.addEphemeralConstant("rand1", lambda: random.uniform(-10, 10), float)
        pset.addEphemeralConstant("rand2", lambda: random.uniform(-1, 1), float)
        pset.addEphemeralConstant("rand3", lambda: random.uniform(0, 1), float)

        ####NEW######
        pset.addPrimitive(GPfunctions.ts_zscore, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.log_diff, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.s_log_1p, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.df_max, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.df_min, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        #############
        pset.addPrimitive(GPfunctions.npfadd, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.npfsub, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.npfmul, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.npfdiv, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)

        pset.addPrimitive(np.add, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.subtract, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.multiply, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.divide, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.negative, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.log, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.log10, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.sqrt, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(np.square, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.true_divide, [pd.core.frame.DataFrame, pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.SignedPower, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Sign, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.extend, [int], int)
        pset.addPrimitive(GPfunctions.extend, [float], float)
        pset.addPrimitive(GPfunctions.sma, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.delta, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.stddev, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.ts_min, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_max, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_sum, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.delay, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_skewness, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_kurtosis, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.ArgMax, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ArgMin, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Product, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.ts_rank, [pd.core.frame.DataFrame,int], pd.core.frame.DataFrame)
        #pset.addPrimitive(scale, [pd.core.frame.DataFrame,int], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.rank, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Abs, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.Decay_lin, [pd.core.frame.DataFrame, int], pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.Decay_exp, [pd.core.frame.DataFrame, float], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.Inverse, [pd.core.frame.DataFrame], pd.core.frame.DataFrame)

        pset.addPrimitive(GPfunctions.correlation, [pd.core.frame.DataFrame, pd.core.frame.DataFrame, int],
                          pd.core.frame.DataFrame)
        pset.addPrimitive(GPfunctions.covariance, [pd.core.frame.DataFrame, pd.core.frame.DataFrame, int],
                          pd.core.frame.DataFrame)

        return pset