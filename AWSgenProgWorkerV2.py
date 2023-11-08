import sys, os
import csv
import pandas as pd
import numpy as np
import datetime
import math
import random
import time
import operator
import math
import random

import pymysql.cursors

from GPfunctions import GPfunctions
from AlphaFitnessFunctions import AlphaFitnessFunctions
from RiskModelFunctions import RiskModelFunctions
import Constants

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

##############################
universe = "YFINANCE"
root = os.path.join("C:\Equities", universe)

alphas_arr = []
histData = {}
corrTScache = {}
featuresList = []
GPfunctionsList = []
market_data_inputs = []
psrCutoff = 0.99 #0.99
corrCutoff = 0.7
fitnessCutoff = 0.5
turnoverMax = 0.3 #0.05
turnoverMin = 0.01
addToDB = True
bookSize = 20000000.0
maxStockWeight = 0.01 #0.01
feesBSP = 0.000  # 0.0010
hedgeVol = False
hedgeIndustry = False
rankHedge = False
funcLookbackLength = 40 #90
linearDecay = 0
topN = 3000
strategyName = 'TEST_STRATEGY_12'
strategyId = -1
runName = 'TEST_STRATEGY_12_A' #EQUITIES_YAHOO_SUB_2
targetDelay = 1
# bottomN = 0
trialCounter = 0
maxTreeDepth = 6 #6 #4
universeBlocking = False
riskModelNumFactors = 5
minPrice = -1.0 #2.0
maxPrice = 10000000.0 # 10000.0
gp_population = 60
riskModelType = Constants.SUB_INDUSTRY_RISK_MODEL
useGammaTransactionModel = False

trunctateEndDate =  '1/1/2020' #'1/1/2021'
trunctateBeginDate = '7/1/2000'

print("connecting to DB")
connection = pymysql.connect(host='alphasdatabase1.cysvmgsjf7ox.us-east-1.rds.amazonaws.com',#'localhost',
                             user='admin', #mysqluser',
                             password='SALMON44', #'mysqluser',
                             db='quantschema',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

for path, subdirs, files in os.walk(root):
    for name in files:
        print(name.split(".")[0])
        histData[name.split(".")[0]] = pd.read_csv(os.path.join(root, name),
                                                   index_col='date',
                                                   parse_dates=True,
                                                   # skiprows=1,
                                                   names=['date', 'dum', 'open', 'high', 'low', 'close', 'volume'],
                                                   usecols=['date', 'dum', 'open', 'high', 'low', 'close', 'volume'],
                                                   dtype={'dum': np.int64, 'open': np.float64, 'high': np.float64,
                                                          'low': np.float64, 'close': np.float64, 'volume': np.int64})

histMultiIndex = pd.concat(histData.values(), keys=histData.keys())

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

df_open_full = df_open.copy()
df_close_full = df_close.copy()
df_volume_full = df_volume.copy()

df_close = df_close.truncate(after=trunctateEndDate)
df_open = df_open.truncate(after=trunctateEndDate)
df_high = df_high.truncate(after=trunctateEndDate)
df_low = df_low.truncate(after=trunctateEndDate)
df_volume = df_volume.truncate(after=trunctateEndDate)

df_open_full = df_open_full.truncate(after=trunctateEndDate)
df_close_full = df_close_full.truncate(after=trunctateEndDate)
df_volume_full = df_volume_full.truncate(after=trunctateEndDate)

df_close = df_close.truncate(before=trunctateBeginDate)
df_open = df_open.truncate(before=trunctateBeginDate)
df_high = df_high.truncate(before=trunctateBeginDate)
df_low = df_low.truncate(before=trunctateBeginDate)
df_volume = df_volume.truncate(before=trunctateBeginDate)

df_open_full = df_open_full.truncate(before=trunctateBeginDate)
df_close_full = df_close_full.truncate(before=trunctateBeginDate)
df_volume_full = df_volume_full.truncate(before=trunctateBeginDate)

df_dollars_traded = df_volume_full * df_close_full
df_dollars_traded_mean = df_dollars_traded.rolling(window=20).mean()
df_dollars_traded_mean_trans = df_dollars_traded_mean.fillna(1.0e4)
df_dollars_traded_mean_trans = df_dollars_traded_mean.replace(0, 1.0e4)
df_dollars_traded_mean_rank = df_dollars_traded_mean.rank(axis=1, ascending=False)

df_hist_vol_10 = df_close.pct_change().rolling(10).std()*(252**0.5)
df_hist_vol_20 = df_close.pct_change().rolling(20).std()*(252**0.5)
df_hist_vol_30 = df_close.pct_change().rolling(30).std()*(252**0.5)
df_hist_vol_90 = df_close.pct_change().rolling(90).std()*(252**0.5)

if (universeBlocking):
    for i in range(len(df_dollars_traded_mean_rank.index)-20, 20, -20):
        print(i)
        df_dollars_traded_mean_rank.iloc[i - 1:i + 19, :] = df_dollars_traded_mean_rank.iloc[i - 1].values

########################################################################
def GetStrategyParameters(stratName):
    try:
        print(f"selecting strategy: {stratName}")
        with connection.cursor() as cursor:
            sql = ("SELECT `strategy_id`,`riskModelType`,`featuresList`,`GPfunctionsList` " +
                   "FROM `quantschema`.`strategies` WHERE `name` = '" + stratName + "'")
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        pass
    return result
########################################################################
########################################################################
def StoreAlpha(alphaString, strategyId, universe, scriptName, sharpe, turnover, returns, margin, fitness, lineardecay, topN, trialCounter,
               feesBSP, targetDelay, universeBlocking,hedgeIndustry,corrCutoff,rankHedge,PSR,riskModelType,minPrice,maxPrice,
               maxStockWeight):
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO `quantschema`.`alphas` " \
                  "(`strategy_id`,`universe`,`scriptversion`,`topN`,`alphastring`,`sharpe`,`turnover`,`returns`," \
                  "`margin`,`fitness`,`lineardecay`,`trialCounter`,`feesBSP`,`targetDelay`," \
                  "`universeBlocking`,`hedgeIndustry`,`corrCutoff`,`rankHedge`,`PSR`,`riskModelType`," \
                  "`minPrice`,`maxPrice`,`maxStockWeight`) " \
                  "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s )"
            # if the connection was lost, then it reconnects
            connection.ping(reconnect=True)
            cursor.execute(sql, (
            str(strategyId), str(universe), str(scriptName), int(topN), str(alphaString), float(sharpe), float(turnover), float(returns), float(margin),
            float(fitness), int(lineardecay), int(trialCounter), float(feesBSP), int(targetDelay),
            int(universeBlocking), int(hedgeIndustry), float(corrCutoff), int(rankHedge), float(PSR), str(riskModelType),
            float(minPrice),float(maxPrice),float(maxStockWeight)))
            connection.commit()
            print("inserted alpha into DB")
    finally:
        pass
    return 1

########################################################################
def GetAlphasFromDB():
    try:
        with connection.cursor() as cursor:
            sql = "SELECT `alphastring` FROM `quantschema`.`alphas` WHERE `scriptversion` = '" + runName + "' ORDER BY RAND()"
            cursor.execute(sql)
            result = cursor.fetchall()
    finally:
        pass
    return result
###########################################################################
def UpdateExistingAlphas():
    alphas = GetAlphasFromDB()
    alphas_arr.clear()
    for row in alphas:
        print(row)
        [A, B] = evalForGraphReturns(row['alphastring'])
        corrTScache[row['alphastring']] = A.sum(axis=1)
        alphas_arr.append(corrTScache[row['alphastring']])
        # if row['alphastring'] in corrTScache.keys():
        #     print("in corrTScache :", row['alphastring'])
        #     alphas_arr.append(corrTScache[row['alphastring']])
        # else:
        #     [A, B] = evalForGraphReturns(row['alphastring'])
        #     alphas_arr.append(A.sum(axis=1).cumsum())
        #     corrTScache[row['alphastring']] = A.sum(axis=1)

########################################################################
def evalForGraphReturns(individual):
    start = time.time()
    func = toolbox.compile(expr=individual)

    # #dum = df_vwap
    # open = df_open
    # high = df_high
    # low = df_low
    # close = df_close
    # volume = df_volume
    # dollars_traded = df_dollars_traded
    # adv20 = df_dollars_traded_mean
    # hist_vol_10 = df_hist_vol_10
    # hist_vol_20 = df_hist_vol_20
    # hist_vol_30 = df_hist_vol_30
    # hist_vol_90 = df_hist_vol_90

    close = df_close
    # market_data_inputs = [df_open,
    #                       df_high,
    #                       df_low,
    #                       df_close,
    #                       df_volume,
    #                       df_dollars_traded,
    #                       df_dollars_traded_mean,
    #                       df_hist_vol_10,
    #                       df_hist_vol_20,
    #                       df_hist_vol_30,
    #                       df_hist_vol_90]

    # nextOpen = open.shift(-1)
    returns = close - close.shift(1)

    if (targetDelay == 1):
        target = (df_open_full.shift(-2) - df_open_full.shift(-1)) / df_open_full.shift(-1)
    elif (targetDelay == -1):
        target = (df_close_full.shift(-1) - df_open_full.shift(-1)) / df_open_full.shift(-1)
    else:
        target = (df_close_full.shift(-1) - df_close_full) / df_close_full

    # apply the GP tree to the DataFrames
    #out = func(dum, open, high, low, close, volume, dollars_traded, adv20, returns)
    #out = func(open, high, low, close, volume, dollars_traded, adv20, returns, hist_vol_10, hist_vol_20, hist_vol_30, hist_vol_90)
    out = func(*market_data_inputs)

    if hedgeVol:
        # stdCorrective = returns.rolling(20).std()
        stdCorrective = returns.abs().rolling(20).std().div(returns.std(axis=1), axis=0)
        out = out.divide(stdCorrective)

    out = out.replace([np.inf, -np.inf], 0.0)

    if rankHedge:
        out = GPfunctions.rank(out)

    if linearDecay > 0:
        out = GPfunctions.Decay_lin(out, linearDecay)

    out[(df_close < minPrice) | (df_close > maxPrice)] = np.nan
    out[df_dollars_traded_mean_rank > topN] = np.nan

    ### RISK MODEL SECTION ###
    if riskModelType == Constants.GLOBAL_RISK_MODEL:
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)
    elif riskModelType == Constants.SUB_INDUSTRY_RISK_MODEL:
        out_normalzed = RiskModelFunctions.hedgeSubIndustries(industries, out)
    elif riskModelType == Constants.PCA_RISK_MODEL:
        out = RiskModelFunctions.pcaConvertAlpha(returns, out, riskModelNumFactors)
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)

    out_normalzed[(df_close < minPrice) | (df_close > maxPrice)] = 0.0
    out_normalzed[df_dollars_traded_mean_rank > topN] = 0.0

        # clip to maxStockWeight
    out_normalzed = np.clip(out_normalzed, -maxStockWeight, maxStockWeight)
    out_normalzed_money = np.multiply(out_normalzed, bookSize)
    turnover = out_normalzed_money.diff().abs().sum(axis=1).mean() / (bookSize)
    print("turnover inside", turnover)
    DFreturns = np.multiply(out_normalzed_money, target)

    # turnoverAdj = out_normalzed_money.diff().abs().sum(axis=1)
    # DFreturns = DFreturns.sum(axis=1) - (turnoverAdj * feesBSP)

    print('corr with: ', out_normalzed_money.corrwith(target).mean())

    return ([DFreturns, out_normalzed])


########################################################################
########################################################################
def evalAlphaGen(individual):
    global trialCounter, market_data_inputs
    print("individual: ", individual)

    trialCounter = trialCounter + 1
    print("trialCounter: ", trialCounter)

    start = time.time()
    func = toolbox.compile(expr=individual)

    close = df_close
    returns = df_close - df_close.shift(1)
    # market_data_inputs = [df_open,
    #                       df_high,
    #                       df_low,
    #                       df_close,
    #                       df_volume,
    #                       df_dollars_traded,
    #                       df_dollars_traded_mean,
    #                       returns,
    #                       df_hist_vol_10,
    #                       df_hist_vol_20,
    #                       df_hist_vol_30,
    #                       df_hist_vol_90]

    # # nextOpen = open.shift(-1)
    # returns = close - close.shift(1)
    if (targetDelay == 1):
        target = (df_open_full.shift(-2) - df_open_full.shift(-1)) / df_open_full.shift(-1)
    elif (targetDelay == -1):
        target = (df_close_full.shift(-1) - df_open_full.shift(-1)) / df_open_full.shift(-1)
    else:
        target = (df_close_full.shift(-1) - df_close_full) / df_close_full

    # apply the GP tree to the DataFrames
    out = func(*market_data_inputs)

    if hedgeVol:
        # stdCorrective = returns.rolling(20).std()
        stdCorrective = returns.abs().rolling(20).std().div(returns.std(axis=1), axis=0)
        out = out.divide(stdCorrective)

    out = out.replace([np.inf, -np.inf], 0.0)

    if rankHedge:
        out = GPfunctions.rank(out)

    if linearDecay > 0:
        out = GPfunctions.Decay_lin(out, linearDecay)

    out[(df_close < minPrice) | (df_close > maxPrice)] = np.nan
    out[df_dollars_traded_mean_rank > topN] = np.nan

    ### RISK MODEL SECTION ###
    if riskModelType == Constants.GLOBAL_RISK_MODEL:
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)
    elif riskModelType == Constants.SUB_INDUSTRY_RISK_MODEL:
        out_normalzed = RiskModelFunctions.hedgeSubIndustries(industries, out)
    elif riskModelType == Constants.PCA_RISK_MODEL:
        out = RiskModelFunctions.pcaConvertAlpha(returns, out, riskModelNumFactors)
        out_normalzed = RiskModelFunctions.hedgeGlobal(out)

    out[(df_close < minPrice) | (df_close > maxPrice)] = 0.0
    out[df_dollars_traded_mean_rank > topN] = 0.0

    # clip to maxStockWeight
    out_normalzed = np.clip(out_normalzed, -maxStockWeight, maxStockWeight)
    out_normalzed_money = np.multiply(out_normalzed, bookSize)
    turnover = out_normalzed_money.diff().abs().sum(axis=1).mean() / (bookSize)
    DFreturns = np.multiply(out_normalzed_money, target)

    turnoverAdj = out_normalzed_money.diff().abs().sum(axis=1)

    # CPS is computed as the total P&L divided by total shares traded.
    sharesTraded = out_normalzed_money.diff().abs() / close

    DFreturnsRowSum = DFreturns.sum(axis=1) - (turnoverAdj * feesBSP)
    DFtotalReturns = DFreturnsRowSum.sum()

    sharpe = GPfunctions.protectedDiv(DFreturnsRowSum.mean(), DFreturnsRowSum.std()) * math.sqrt(252.0)
    returnsPerc = GPfunctions.protectedDiv((DFreturnsRowSum.mean() * 252.0), (bookSize * 0.5))

    margin = GPfunctions.protectedDiv(DFtotalReturns, sharesTraded.sum(axis=1).sum()) * 100.0
    # margin = (DFtotalReturns / sharesTraded.sum(axis=1).sum()) * 100.0

    # fitness = sharpe
    if (turnover > turnoverMin):
        # fitness =  sharpe * math.sqrt(abs(returnsPerc) / turnover)
        fitness = sharpe * math.sqrt(abs(returnsPerc) / max(turnover, 0.125))
        #fitness = sharpe
    else:
        fitness = -1

    PSR = AlphaFitnessFunctions.probabalisticSharpeRatio(DFreturns.sum(axis=1), fitnessCutoff)

    if turnover > turnoverMax and PSR > psrCutoff and fitness > fitnessCutoff:
        originalInd = str(individual)
        #for linearDecayLookback in range(2, 40):
        for exponentialDecayLookback in [x * 0.05 for x in range(19, 1, -1)]:
            individual = "Decay_exp(" + originalInd + "," + str(exponentialDecayLookback) + ")"
            func = toolbox.compile(expr=individual)
            print("NEW INDIVIDUAL :", individual)
            out = func(*market_data_inputs)
            out = out.replace([np.inf, -np.inf], np.nan)
            ### RISK MODEL SECTION ###
            if riskModelType == Constants.GLOBAL_RISK_MODEL:
                out_normalzed = RiskModelFunctions.hedgeGlobal(out)
            elif riskModelType == Constants.SUB_INDUSTRY_RISK_MODEL:
                out_normalzed = RiskModelFunctions.hedgeSubIndustries(industries, out)
            elif riskModelType == Constants.PCA_RISK_MODEL:
                out = RiskModelFunctions.pcaConvertAlpha(returns, out, riskModelNumFactors)
                out_normalzed = RiskModelFunctions.hedgeGlobal(out)
            ##########################

            # clip to maxStockWeight
            out_normalzed = np.clip(out_normalzed, -maxStockWeight, maxStockWeight)
            out_normalzed_money = np.multiply(out_normalzed, bookSize)
            turnover = out_normalzed_money.diff().abs().sum(axis=1).mean() / (bookSize)
            DFreturns = np.multiply(out_normalzed_money, target)

            turnoverAdj = out_normalzed_money.diff().abs().sum(axis=1)

            # CPS is computed as the total P&L divided by total shares traded.
            sharesTraded = out_normalzed_money.diff().abs() / close
            DFreturnsRowSum = DFreturns.sum(axis=1) - (turnoverAdj * feesBSP)
            DFtotalReturns = DFreturnsRowSum.sum()
            sharpe = GPfunctions.protectedDiv(DFreturnsRowSum.mean(), DFreturnsRowSum.std()) * math.sqrt(252)
            fitness = sharpe * math.sqrt(abs(returnsPerc) / max(turnover, 0.125))
            print("COUNTOWN SHARPE:",sharpe)
            returnsPerc = GPfunctions.protectedDiv((DFreturnsRowSum.mean() * 252.0), (bookSize * 0.5))

            margin = GPfunctions.protectedDiv(DFtotalReturns, sharesTraded.sum(axis=1).sum()) * 100.0
            PSR = AlphaFitnessFunctions.probabalisticSharpeRatio(DFreturns.sum(axis=1), fitnessCutoff)
            #PSR = 1.0
            if turnover < turnoverMax and PSR > psrCutoff and fitness > fitnessCutoff:
                break

    if (PSR > psrCutoff and fitness > fitnessCutoff and addToDB and turnover < turnoverMax and turnover > turnoverMin):
        # if (fitness>fitnessCutoff and turnover>turnoverMin):
        # fitness = sharpe
        UpdateExistingAlphas()
        corrTestalphas = pd.DataFrame(alphas_arr)
        print('corrTestalphas', corrTestalphas)
        corrTest = corrTestalphas._append(DFreturns.sum(axis=1), ignore_index=True).transpose().fillna(0).diff().corr()
        print(corrTest)
        maxCorr = corrTest.tail(1).iloc[:, :-1]
        print(maxCorr)
        # print(corrTest.tail(1).iloc[:,:-1])
        # fitness = fitness * -0.2
        if (maxCorr.empty):
            StoreAlpha(individual, strategyId, universe, runName, sharpe, turnover, returnsPerc, margin, fitness, linearDecay, topN,
                       trialCounter, feesBSP, targetDelay, universeBlocking,hedgeIndustry, corrCutoff, rankHedge, PSR, riskModelType,
                       minPrice, maxPrice, maxStockWeight)
        elif (maxCorr.transpose().max().iloc[0] < corrCutoff):
            print('maxCorr.max().iloc[0]', maxCorr.transpose().max().iloc[0])
            #fitness = sharpe
            fitness = sharpe * math.sqrt(abs(returnsPerc) / max(turnover, 0.125))
            StoreAlpha(individual, strategyId, universe, runName, sharpe, turnover, returnsPerc, margin, fitness, linearDecay, topN,
                       trialCounter, feesBSP, targetDelay, universeBlocking, hedgeIndustry, corrCutoff, rankHedge, PSR, riskModelType,
                       minPrice, maxPrice, maxStockWeight)
        fitness = fitness * -0.2

    print("DFtotalReturns:", DFtotalReturns)
    print("returnsPerc:", returnsPerc)
    print("sharpe:", sharpe)
    print("turnover:", turnover)
    print("margin BSP:", margin)
    print("fitness:", fitness)
    end = time.time()
    print("eval time: ", end - start)

    if (math.isnan(sharpe)):
        fitness = 0
    return ([fitness])


########################################################################
#Get strategy parameters
# market_dataframes =  [df_open,
#                       df_high,
#                       df_low,
#                       df_close,
#                       df_volume,
#                       df_dollars_traded,
#                       df_dollars_traded_mean,
#                       df_close - df_close.shift(1),
#                       df_hist_vol_10,
#                       df_hist_vol_20,
#                       df_hist_vol_30,
#                       df_hist_vol_90]
market_dataframes_map =   { 'open': df_open,
                            'high': df_high,
                            'low': df_low,
                            'close': df_close,
                            'volume': df_volume,
                            'dollars_traded': df_dollars_traded,
                            'adv20': df_dollars_traded_mean,
                            'returns': df_close - df_close.shift(1),
                            'hist_vol_10': df_hist_vol_10,
                            'hist_vol_20': df_hist_vol_20,
                            'hist_vol_30': df_hist_vol_30,
                            'hist_vol_90': df_hist_vol_90}

strategyParams = GetStrategyParameters(strategyName)
strategyId = strategyParams[0]['strategy_id']
featuresList = eval(strategyParams[0]['featuresList'])
GPfunctionsList = eval(strategyParams[0]['GPfunctionsList'])
for f in featuresList:
    market_data_inputs.append(market_dataframes_map[f])

########################################################################
gpDataFramList = []
for f in featuresList:
    gpDataFramList.append(pd.core.frame.DataFrame)

pset = gp.PrimitiveSetTyped("main", gpDataFramList, pd.core.frame.DataFrame)

for x in range(1, funcLookbackLength):
    pset.addTerminal(x, int)

pset = GPfunctions.addGPfunctionsToToolboxFromDictionary(pset, GPfunctionsList)
pset = GPfunctions.GP_rename_arguments(pset, featuresList)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evalAlphaGen)
toolbox.register("select", tools.selTournament, tournsize=7)  # 7
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)  # 2
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxTreeDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxTreeDepth))  # 4 5


def main():
    pop = toolbox.population(n=gp_population)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, random.random(), random.random(), 50000, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(len(hof))
    print(hof[0])

    return pop, log, hof


if __name__ == "__main__":
    main()
