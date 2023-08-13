import sys, os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import math
import datetime
import boto
import s3fs


from statsforecast.models import SimpleExponentialSmoothingOptimized

import pymysql.cursors

from deap import base
from deap import creator
from deap import tools
from deap import gp

from scipy.optimize import least_squares, minimize
#from skopt import gp_minimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance, ShrunkCovariance, OAS, MinCovDet, GraphicalLassoCV

from GPfunctions import GPfunctions
from RiskModelFunctions import RiskModelFunctions
import Constants

##############################
root = "C:\Equities\YFINANCE"
timeRateMin = 1440

alphas_arr = []
histData = {}
numEquities = 0
bookSize = 20000000.0 #100000000.0 #1000000000
maxStockWeight = 0.01
feesBSP = 0.0000 #0.0020
hedgeVol = False
rankHedge = False
funcLookbackLength = 90
linearDecay = 0 #7
expDecay = 0.0
topN = 3000 ##100
targetDelay = 1
targetFuture = 0
#bottomN = 5
portTail = 0.00 #0.0015##0.00000001 #0.0015#0.001 #0.0015 0.0025
universeBlocking = False
riskModelType = Constants.GLOBAL_RISK_MODEL #Constants.PCA_RISK_MODEL GLOBAL_RISK_MODEL
riskModelNumFactors = 5
pcaMA = 0.9
runName = 'NEW_DAO_1_PSR' #'EQUITIES_YAHOO_SUB_2' #'EQUITIES_YAHOO_SUB_2' CRYPTO_SMALL5 CRYPTO_SMALL4
g_alphas_arr = []
g_raw_alphas_dic = {}
testStartDate = "2021-01-04"
optimEndDate = "2022-01-03"
minPrice = 0.0 #2.0
maxPrice = 10000000.0 # 10000.0
useGammaTransactionModel = False
expFactorDecay = 0.1
volumeMeanRankingWindow = 252


LOG_DATA_PATH = 'logDataFiles/'
MODEL_DATA_PATH = 'ModelOutputs/'
MODEL_NAME = "M2_" + str(round(time.time() * 1000))

print("connecting to DB")
# connection = pymysql.connect(host='localhost',
#                              user='mysqluser',
#                              password='mysqluser',
#                              db='quantschema',
#                              charset='utf8mb4',
#                              cursorclass=pymysql.cursors.DictCursor)
#
connection = pymysql.connect(host='alphasdatabase1.cysvmgsjf7ox.us-east-1.rds.amazonaws.com',#'localhost',
                             user='admin', #mysqluser',
                             password='SALMON44', #'mysqluser',
                             db='quantschema',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

# for path, subdirs, files in os.walk(root):
#     for name in files:
#         print(name.split(".")[0])
#         histData[name.split(".")[0]] = pd.read_csv(os.path.join(root, name),
#                                                    index_col='date',
#                                                    parse_dates=True,
#                                                    # skiprows=1,
#                                                    names=['date', 'dum', 'open', 'high', 'low', 'close', 'volume'],
#                                                    usecols=['date', 'dum', 'open', 'high', 'low', 'close', 'volume'],
#                                                    dtype={'dum': np.int64, 'open': np.float64, 'high': np.float64,
#                                                           'low': np.float64, 'close': np.float64, 'volume': np.int64})
#         numEquities = numEquities + 1
#
# histMultiIndex = pd.concat(histData.values(), keys=histData.keys())
#histMultiIndex.to_parquet('EquitiesDataFiles/' + "E1000.parquet")
#histMultiIndex = pd.read_parquet("s3://brethribar-equitiesdata-1/E1000.parquet")
histMultiIndex = pd.read_parquet('EquitiesDataFiles/' + "E1000.parquet")

industries = pd.read_csv('C:\Equities\symSubIndustries.csv', index_col=0)
industries = industries[industries['INDUSTRY'] != '1c3d7001-dc68-4c36-b148-483741091c86'] # BIOTECH REMOVAL
industries = industries[industries['INDUSTRY'] != 'd6c806bb-aaaf-4bbc-9737-3575d53ca96f'] # Finance-Mortgage REIT
industries = industries[industries['INDUSTRY'] != 'ed543f01-e605-4a2a-a386-ce0c09dba19e'] # Finance-Property REIT

print(industries)

df_open = histMultiIndex["open"].unstack(level=0)
df_high = histMultiIndex["high"].unstack(level=0)
df_low = histMultiIndex["low"].unstack(level=0)
df_close = histMultiIndex["close"].unstack(level=0)
df_volume = histMultiIndex["volume"].unstack(level=0)

df_open = df_open[df_open.columns.intersection(industries.index.tolist())]
df_high = df_high[df_high.columns.intersection(industries.index.tolist())]
df_low = df_low[df_low.columns.intersection(industries.index.tolist())]
df_close = df_close[df_close.columns.intersection(industries.index.tolist())]
df_volume = df_volume[df_volume.columns.intersection(industries.index.tolist())]

#trunctateEndDate = '1614801600000'
# trunctateEndDate = '999999999999999'#'5/1/2019' #'3/09/2018' #'7/18/2018' #'4/17/2019'
# df_close = df_close.truncate(after=trunctateEndDate)
# df_open = df_open.truncate(after=trunctateEndDate)
# df_high = df_high.truncate(after=trunctateEndDate)
# df_low = df_low.truncate(after=trunctateEndDate)
# df_volume = df_volume.truncate(after=trunctateEndDate)

######################################

df_dollars_traded = df_volume * df_close
df_dollars_traded_mean = df_dollars_traded.rolling(window=volumeMeanRankingWindow).mean() #20
#fill NaN and zero volume with 10,000 to simulate illiquidity
df_dollars_traded_mean_trans = df_dollars_traded_mean.fillna(1.0e4)
df_dollars_traded_mean_trans = df_dollars_traded_mean.replace(0, 1.0e4)
df_dollars_traded_mean_rank = df_dollars_traded_mean.rank(axis=1, ascending=False)

# df_hist_vol_10 = df_close.pct_change().rolling(10).std()*(252**0.5)
# df_hist_vol_20 = df_close.pct_change().rolling(20).std()*(252**0.5)
# df_hist_vol_30 = df_close.pct_change().rolling(30).std()*(252**0.5)
# df_hist_vol_90 = df_close.pct_change().rolling(90).std()*(252**0.5)

if (universeBlocking):
    for i in range(len(df_dollars_traded_mean_rank.index)-20, 20, -20):
        print(i)
        df_dollars_traded_mean_rank.iloc[i - 1:i + 19, :] = df_dollars_traded_mean_rank.iloc[i - 1].values

#df_close.to_csv('C:\Crypto\df_close.csv')
testStart = int(df_close.index.get_loc(testStartDate))
optimEnd = int(df_close.index.get_loc(optimEndDate))

# startTest = time.time()
# T = GPfunctions.ArgMax(df_close)
# endtest = time.time()
# print("ArgMax eval time: ", startTest - endtest)
# startTest = time.time()
# TT = GPfunctions.ArgMax2(df_close)
# endtest = time.time()
# print("ArgMax2 eval time: ", startTest - endtest)
# print(" equal? "+str(T.equals(TT)))
########################################################################
def evalForGraphReturns(individual):
    start = time.time()
    func = toolbox.compile(expr=individual)

    open = df_open
    high = df_high
    low = df_low
    close = df_close
    volume = df_volume
    dollars_traded = df_dollars_traded
    adv20 = df_dollars_traded_mean

    # nextOpen = open.shift(-1)
    returns = close - close.shift(1)
    if (targetDelay == 1):
        target = (df_open.shift(-2) - df_open.shift(-1)) / df_open.shift(-1)
    elif (targetDelay == -1):
        target = (close.shift(-1) - open.shift(-1)) / open.shift(-1)
    else:
        target = (close.shift(-1) - close) / close

    if (targetFuture > 0):
        target = (close.shift(-targetFuture) - close) / close

    print("close dim:", close.shape)

    # apply the GP tree to the DataFrames
    out = func(open, high, low, close, volume, dollars_traded, adv20, returns)

    if hedgeVol:
        stdCorrective = returns.abs().rolling(20).std().div(returns.std(axis=1), axis=0)
        out = out.divide(stdCorrective)

    #out = out.replace([np.inf, -np.inf], np.nan)
    out = out.replace([np.inf, -np.inf], 0.0)

    # RANK
    if rankHedge:
        out = GPfunctions.rank(out)

    if linearDecay > 0:
        out = GPfunctions.Decay_lin(out, linearDecay)

    if expDecay > 0:
       out = pd.DataFrame(out).ewm(alpha=expDecay, axis=0).mean()

    out[(df_close < minPrice) | (df_close > maxPrice)] = np.nan
    out[df_dollars_traded_mean_rank > topN] = np.nan

    ### RISK MODEL SECTION ###
    if riskModelType == Constants.GLOBAL_RISK_MODEL:
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)
    elif riskModelType == Constants.SUB_INDUSTRY_RISK_MODEL:
        startTest = time.time()
        out_normalzed = RiskModelFunctions.hedgeSubIndustries(industries, out)
        endtest = time.time()
        print("RiskModelFunctions.hedgeSubIndustries eval time: ", startTest - endtest)
        # startTest = time.time()
        # out_normalzed2 = RiskModelFunctions.hedgeSubIndustries2(industries, out)
        # endtest = time.time()
        # print("RiskModelFunctions.hedgeSubIndustries2 eval time: ", startTest - endtest)
        # print("dataframes are equal?: ", out_normalzed.equals(out_normalzed2))
        TEST = 1
    elif riskModelType == Constants.PCA_RISK_MODEL:
        out = RiskModelFunctions.pcaConvertAlpha(returns.iloc[:testStart, :], out, riskModelNumFactors)
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)
    elif riskModelType == Constants.EXP_PCA_RISK_MODEL:
        out = RiskModelFunctions.pcaMovingAvg(returns.iloc[:testStart, :], out, riskModelNumFactors, pcaMA)
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)
    ##########################

    out_normalzed[(df_close < minPrice) | (df_close > maxPrice)] = 0.0
    out_normalzed[df_dollars_traded_mean_rank > topN] = 0.0

    # clip to maxStockWeight
    out_normalzed = np.clip(out_normalzed, -maxStockWeight, maxStockWeight)
    out_normalzed_money = np.multiply(out_normalzed, bookSize)
    #turnover = out_normalzed_money.diff().abs().sum(axis=1).mean() / (bookSize)
    DFreturns = np.multiply(out_normalzed_money, target)

    #turnoverAdj = out_normalzed_money.diff().abs().sum(axis=1)
    DFreturnsRowSum = DFreturns.sum(axis=1) #- (turnoverAdj * feesBSP)

    print('corr with: ', out_normalzed_money.corrwith(target).mean())

    return ( DFreturns, out_normalzed, DFreturnsRowSum)
########################################################################
########################################################################
def ReturnY():
    open = df_open
    close = df_close

    if (targetDelay == 1):
        target = (df_open.shift(-2) - df_open.shift(-1)) / df_open.shift(-1)
    elif (targetDelay == -1):
        target = (close.shift(-1) - open.shift(-1)) / open.shift(-1)
    else:
        target = (close.shift(-1) - close) / close

    if (targetFuture > 0):
        target = (close.shift(-targetFuture) - close) / close

    return (target)

#######################################################################
########################################################################
pset = gp.PrimitiveSetTyped("main", [pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame,
                                     #pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame,
                                     pd.core.frame.DataFrame], pd.core.frame.DataFrame)

for x in range(1, funcLookbackLength):
    pset.addTerminal(x, int)

pset = GPfunctions.addGPfunctionsToToolbox(pset)

pset.renameArguments(ARG0="open")
pset.renameArguments(ARG1="high")
pset.renameArguments(ARG2="low")
pset.renameArguments(ARG3="close")
pset.renameArguments(ARG4="volume")
pset.renameArguments(ARG5="dollars_traded")
pset.renameArguments(ARG6="adv20")
pset.renameArguments(ARG7="returns")

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)

########################################################################
########################################################################
def GetAlphasFromDB(numalphas):
    try:
        with connection.cursor() as cursor:
            #  `alphasid` <= 3487653 AND
            # sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `DSRprob` IN ('TEST') ORDER BY RAND() LIMIT %s" 3484460
            #sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `alphasid` in ('3490869') ORDER BY RAND() LIMIT %s"
            #sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `turnover` < 999.1 AND `scriptversion` IN ('TEST3_U','NEW_DAO_1_PSR_SUB','NEW_DAO_2_SUB','NEW_DAO_1_SUB') LIMIT %s"
            #sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `alphasid` > 1 AND `scriptversion` IN ('" + runName + "') LIMIT %s"
            sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `PSR` > 0.99 AND `riskModelType` = 'Global' LIMIT %s"
            #sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `sharpe` > 0  AND `turnover` < 0.5 AND `scriptversion` IN ('" + runName + "') ORDER BY RAND() LIMIT %s"
            cursor.execute(sql, (int(numalphas)))
            # sql = "SELECT `alphastring` FROM `quantschema`.`distinctuniquealphas` WHERE `corr` > 0.3 LIMIT %s"
            # sql = "SELECT `alphastring` FROM `quantschema`.`permalphalist`"
            # cursor.execute(sql)
            result = cursor.fetchall()
            # print(result)
    finally:
        pass
    return result


###########################################################################
############################################################################

def calcPortReturns( weightArray ):
    counter = 0
    weightArray = pd.Series(weightArray).div(weightArray.sum(), axis=0)

    #returnsReturns = pd.Series(g_alphas_arr[0].iloc[:testStart].copy())
    returnsReturns = pd.Series(g_alphas_arr[0].iloc[testStart:optimEnd].copy())

    returnsReturns = returnsReturns * 0
    for factor in g_alphas_arr:
        returnsReturns = returnsReturns.add(weightArray[counter] * factor.iloc[testStart:optimEnd])
        counter = counter + 1
    sharpe = (returnsReturns.mean() * 252.0) / (returnsReturns.std() * math.sqrt(252))
    print("sharpe: "+str(sharpe))
    return -sharpe

############################################################################
############################################################################
def calcPortReturnsWithFees( weightArray ):
    #weightArray = pd.Series(weightArray)
    weightArray = pd.Series(weightArray).div(weightArray.sum(), axis=0)
    for counter, wMat in enumerate(g_raw_alphas_dic):
        alphaWeight = weightArray[counter]
        optimRawAlpha = g_raw_alphas_dic[counter].iloc[testStart:optimEnd]
        if counter == 0:
            weightedAlpha = optimRawAlpha.multiply(alphaWeight, axis=0).fillna(0.0)
        else:
            weightedAlpha = weightedAlpha.add(optimRawAlpha.multiply(alphaWeight, axis=0).fillna(0.0))

    ##NEW NORMALIZATION
    weightedAlpha.replace(0, np.nan, inplace=True)
    weightedAlpha = RiskModelFunctions.hedgeGlobal(weightedAlpha)
    weightedAlpha.replace(np.nan, 0, inplace=True)

    weightedAlpha = weightedAlpha * bookSize

    combinedAlpha = pd.DataFrame(weightedAlpha.values * ReturnY().iloc[testStart:optimEnd].values, columns=weightedAlpha.columns, index=weightedAlpha.index)
    #combinedAlpha = pd.DataFrame(weightedAlpha.values * ReturnY().values, columns=weightedAlpha.columns, index=weightedAlpha.index)


    turnoverAdj = weightedAlpha.diff().abs().sum(axis=1)
    feeCombinedAlpha = (combinedAlpha.sum(axis=1) - (turnoverAdj * feesBSP))

    #sharpe = (feeCombinedAlpha.mean() * 252.0) / (feeCombinedAlpha.std() * math.sqrt(252))
    sharpe = (feeCombinedAlpha.mean() * 252.0) / (feeCombinedAlpha.std() * math.sqrt(252))
    print("sharpe fees: "+str(sharpe) + " testStart: "+ str(testStart) + " optimEnd: "+ str(optimEnd))

    # l1_weight = 0.0
    # l2_weight = 0.0
    # l1_penalty = l1_weight * weightArray.abs().sum()
    # l2_penalty = l2_weight * (weightArray ** 2).sum()
    # neg_sharpe_with_penalty = -sharpe + l1_penalty + l2_penalty

    return -sharpe
############################################################################
############################################################################
#def get_grad_func(h0, risk_aversion, Q, QT, specVar, alpha_vec, Lambda):
def get_grad_func(h0, alpha_vec, bsp_costs):
    def grad_func(h):
        g = -alpha_vec
        g += (h - h0) * bsp_costs
        return np.asarray(g)
    return grad_func
############################################################################

def main():
    global testStart, optimEnd
    pd.DataFrame(ReturnY()).fillna(0).to_csv(LOG_DATA_PATH+'returnY.csv')
    numberOfAlphas = 1000
    alphas = GetAlphasFromDB(numberOfAlphas)

    counter = 0
    aINT = 0
    alphas_arr = []
    raw_alphas_dic = {}
    for row in alphas:
        print(row)
        [A, B, C] = evalForGraphReturns(row['alphastring'])
        NANs = B.isna().sum().sum()
        print("NANs: ", NANs)
        percentNan = NANs / B.size
        print("Percent NANs: ", percentNan)
        if percentNan < 999999:
            g_raw_alphas_dic[counter] = B
            raw_alphas_dic[counter] = B
            counter = counter + 1
        alphas_arr.append(A.sum(axis=1).cumsum())
        g_alphas_arr.append(C)
        print("counter: ", counter)
        aINT = aINT + 1

    alphasDF = pd.DataFrame(alphas_arr)

    ############################
    #######################################
    print('TEST!!!')
    alphasDF = alphasDF.transpose().diff().fillna(0)
    #alphasDFCov = LedoitWolf().fit(alphasDF.iloc[:testStart,:].values)
    alphasDFCov = EmpiricalCovariance().fit(alphasDF.iloc[:testStart, :].values)
    #alphasDFCovInv = pd.DataFrame(np.linalg.pinv(alphasDFCov.covariance_))
    alphasDFCovInv = pd.DataFrame(np.linalg.pinv(np.eye(len(alphasDFCov.covariance_))))
    #alphasDFCovInv = pd.DataFrame(np.linalg.pinv(np.multiply(alphasDFCov.covariance_, np.eye(len(alphasDFCov.covariance_)))))

    # npArrays = np.empty((len(alphasDF.index), len(alphasDF.columns)), float)
    # for col in alphasDF:
    #     seso = SimpleExponentialSmoothingOptimized()
    #     seso = seso.fit(y=alphasDF[col].to_numpy())
    #     pred = seso.predict_in_sample()['fitted']
    #     #pred = np.append(pred, [0])
    #     npArrays[:,col] = pred
    #
    # alphasExpectedReturns =pd.DataFrame(npArrays)

    alphasExpectedReturns = GPfunctions.Decay_exp(alphasDF, expFactorDecay).shift(1)
    alphasExpectedReturns[alphasExpectedReturns < 0] = 0

    alphaweightsTS = pd.DataFrame(np.inner(alphasDFCovInv, alphasExpectedReturns)).transpose()
    alphaweightsTS = pd.DataFrame(alphaweightsTS)
    alphaweightsTS = alphaweightsTS.div(alphaweightsTS.sum(axis=1), axis=0)


    print('alphaweightsTS: ', alphaweightsTS)

    print(alphaweightsTS)
    alphaweightsTS.to_csv(LOG_DATA_PATH+'AlphaweightsTS.csv')

    combinedAlpha = pd.DataFrame(alphaweightsTS.values * alphasDF.values, columns=alphaweightsTS.columns, index=A.index)
    combinedAlpha.to_csv(LOG_DATA_PATH+'combinedAlpha.csv')

    sharpe = (combinedAlpha.sum(axis=1).mean() * 252.0) / (combinedAlpha.sum(axis=1).std() * math.sqrt(252))
    print("PCA COMBINE SHARPE:", sharpe)
    sharpe = (combinedAlpha.iloc[testStart:, :].sum(axis=1).mean() * 252.0) / (
                combinedAlpha.iloc[testStart:, :].sum(axis=1).std() * math.sqrt(252))
    print("TEST PCA COMBINE SHARPE:", sharpe)
    returns = (pd.DataFrame(combinedAlpha).sum(axis=1).cumsum()).tail(252).diff().sum() * (252.0 / 252)
    print("returns:", returns)

    portfolioReturns = pd.DataFrame(combinedAlpha).sum(axis=1).cumsum()
    portfolioReturns.iloc[:testStart - 1].plot(color="blue", linewidth=2.0, linestyle="-", title='Portfolio Returns')
    portfolioReturns.iloc[testStart:].plot(color="red", linewidth=2.0, linestyle="-")

    plt.ylabel('Returns Unlevered')

    # ###combine so turnover is calced properly#################
    for counter in range(len(raw_alphas_dic)):
        print("wMat: ", counter, "  ", raw_alphas_dic[counter])
        NANs = raw_alphas_dic[counter].isna().sum().sum()
        print("NANs: ", NANs)
        print("Percent NANs: ", NANs / raw_alphas_dic[counter].size)
        aWeightsDF = alphaweightsTS[counter]  # .= .reindex(raw_alphas_dic[counter].index)
        aWeightsDF.index = raw_alphas_dic[counter].index
        if counter == 0:
            weightedAlpha = raw_alphas_dic[counter].multiply(aWeightsDF, axis=0).fillna(0.0)
        else:
            weightedAlpha = weightedAlpha.add(raw_alphas_dic[counter].multiply(aWeightsDF, axis=0).fillna(0.0))
            pass

    #NEW NORMALIZATION!!
    weightedAlpha.replace(0, np.nan, inplace=True)
    weightedAlpha = RiskModelFunctions.hedgeGlobal(weightedAlpha)
    weightedAlpha.replace(np.nan,0 , inplace=True)

    weightedAlpha = weightedAlpha * bookSize

    if (portTail > 0):
        #weightedAlpha = GPfunctions.Tail(weightedAlpha, portTail)
        weightedAlpha = weightedAlpha * 5.0  # 4

    print("weightedAlpha.shape", weightedAlpha.shape)

    weightedAlpha.to_csv('C:\Equities\WA.csv')

    turnoverAdj = weightedAlpha.diff().abs().sum(axis=1)

    transModelGamma = 1.0 / (10.0 * df_dollars_traded_mean_trans)
    if useGammaTransactionModel:
        #turnoverModel2 = transModelGamma * weightedAlpha.diff().abs() * bookSize
        weightedAlphaDiff = weightedAlpha.diff().abs()**2

    weightedAlphaDiff = weightedAlpha.diff().abs() ** 2
    turnoverModel = pd.DataFrame(transModelGamma.values * weightedAlphaDiff.values, columns=weightedAlpha.columns, index=A.index)

    combinedAlpha2 = pd.DataFrame(weightedAlpha.values * ReturnY().values, columns=weightedAlpha.columns, index=A.index)

    ##NEW NORMALIZATION
    combinedAlpha2.replace(np.nan, 0, inplace=True)

    weightedAlpha.to_csv(MODEL_DATA_PATH + MODEL_NAME)
    corr = pd.DataFrame(weightedAlpha.values).corrwith(pd.DataFrame(ReturnY().values)).mean()

    sharpe = (combinedAlpha2.sum(axis=1).mean() / combinedAlpha2.sum(axis=1).std()) * math.sqrt(252.0)
    print("FULL NO FEES SHARPE:", sharpe)
    sharpe = (combinedAlpha2.iloc[testStart:, :].sum(axis=1).mean() /
                combinedAlpha2.iloc[testStart:, :].sum(axis=1).std()) * math.sqrt(252)
    print("TEST NO FEES SHARPE:", sharpe)
    sharpe = (combinedAlpha2.iloc[:testStart, :].sum(axis=1).mean() /
                combinedAlpha2.iloc[:testStart, :].sum(axis=1).std()) * math.sqrt(252)
    print("TRAIN NO FEES SHARPE:", sharpe)
    # annCorrection = (1440 / timeRateMin) * 252
    # annSharpe = (combinedAlpha2.iloc[optimEnd:].sum(axis=1).mean() * annCorrection) / (
    #             combinedAlpha2.iloc[optimEnd:].sum(axis=1).std() * math.sqrt(annCorrection))  #35040 #17520
    # print("ANNUALIZED TEST SHARPE:", annSharpe)
    # returns = (pd.DataFrame(combinedAlpha2).sum(axis=1).cumsum()).tail(90).diff().sum() * (252.0 / 90)
    # print("returns:", returns)

    ((pd.DataFrame(combinedAlpha2).sum(axis=1) - (turnoverAdj * feesBSP)).cumsum()).plot()
    (((combinedAlpha2-turnoverModel).sum(axis=1)).cumsum()).plot()

    finalTurnover = turnoverAdj.mean() / bookSize
    #finalTurnoverModel = turnoverModel.mean()

    feeCombinedAlpha = (combinedAlpha2.sum(axis=1) - (turnoverAdj * feesBSP))
    #feeCombinedAlpha = (combinedAlpha2-turnoverModel).sum(axis=1)

    sharpe = (feeCombinedAlpha.mean() / feeCombinedAlpha.std()) * math.sqrt(252.0)
    print("FEES FULL SHARPE:", sharpe)
    sharpe = (feeCombinedAlpha.iloc[:testStart].mean() / feeCombinedAlpha.iloc[:testStart].std()) * math.sqrt(252.0)
    print("FEES TRAIN SHARPE:", sharpe)
    sharpe = (feeCombinedAlpha.iloc[testStart:].mean() / feeCombinedAlpha.iloc[testStart:].std()) * math.sqrt(252.0)
    print("FEES TEST SHARPE:", sharpe)

    # annFeeSharpe = (feeCombinedAlpha.iloc[optimEnd:].mean() * annCorrection) / (
    #             feeCombinedAlpha.iloc[optimEnd:].std() * math.sqrt(annCorrection))  #35040 #17520
    # print("ANNUALIZED FEES TEST SHARPE:", annFeeSharpe)

    #returns = (pd.DataFrame(feeCombinedAlpha).cumsum()).tail(90).diff().sum() * ((1400 / timeRateMin * 365 * 24) / 365.0)
    returns = ((feeCombinedAlpha.iloc[testStart:].cumsum()).diff().sum() / len(feeCombinedAlpha.iloc[testStart:])) / bookSize * 252.0

    print("FEES returns:", returns)
    print("finalTurnover: ", finalTurnover)
    #print("finalTurnover: ", finalTurnoverModel)
    print("corr with: ", corr)
    print("mean daily returns: ", returns / 252.0)
    combinedAlpha2.sum(axis=1).to_csv(LOG_DATA_PATH+'combinedAlpha2.csv')
    print("linearDecay: ", linearDecay)
    print("expDecay: ", expDecay)
    print("riskModelNumFactors: ", riskModelNumFactors)
    print("topN: ", topN)
    print("universeBlocking: ", universeBlocking)
    # print("optimStepSize: ", optimStepSize)
    # print("optimLookback: ", optimLookback)
    # print("maxiter: ", maxiter)
    print("maxStockWeight: ", maxStockWeight)
    print("feesBSP: ", feesBSP)
    print("expFactorDecay: ", expFactorDecay)


    plt.show()

    return combinedAlpha


if __name__ == "__main__":
    main()
