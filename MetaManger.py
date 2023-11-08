import sys, os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import operator
import math

import scipy
from scipy.optimize import minimize
#from skopt import gp_minimize
from RiskModelFunctions import RiskModelFunctions
import Constants
##############################
root = "C:\Equities\YFINANCE"
MODEL_DATA_PATH = 'ModelOutputs/'
LOG_DATA_PATH = 'logDataFiles/'

g_alphas_arr = []

bookSize = 20000000.0
feesBSP = 0.0000
annCorrection = 252
maxStockWeight = 0.005

returnY = pd.read_csv(LOG_DATA_PATH+"returnY.csv",index_col='date', parse_dates=True)

LambdaMatrix = pd.read_parquet('EquitiesDataFiles/' + "Lambda1000.parquet")
LambdaMatrix[:] = 0
B = pd.read_parquet('logDataFiles/' + "B.parquet")
F_cov = pd.read_parquet('logDataFiles/' + "F_cov.parquet")
BT = B.transpose()
risk_aversion = 1.0e-2 #1.0e-6

Q = np.matmul(scipy.linalg.sqrtm(F_cov), BT.T)
QT = Q.transpose()


for path, subdirs, files in os.walk(MODEL_DATA_PATH):
    for name in files:
        g_alphas_arr.append(pd.read_csv(MODEL_DATA_PATH+name,index_col='date',parse_dates=True))

testStartDate = "2021-01-04"
optimEndDate = "2022-01-03"
testStart = int(g_alphas_arr[0].index.get_loc(testStartDate))
optimEnd = int(g_alphas_arr[0].index.get_loc(optimEndDate))

############################################################################
############################################################################
def get_obj_func(h0, risk_aversion, alpha_vec, Q, Lambda):
    def obj_func(h):
        #f = 0
        #f = 0.5 * risk_aversion * np.sum(np.matmul(Q, h) ** 2)
        #f = np.dot(h, alpha_vec)
        f = np.sum(h - alpha_vec)
        ########f += np.dot((h - h0) ** 2, Lambda)
        f += np.nansum(((h - h0) ** 2) * Lambda)
        return f
    return obj_func
############################################################################
#def get_grad_func(h0, risk_aversion, Q, QT, specVar, alpha_vec, Lambda):
def get_grad_func(h0, risk_aversion, Q, QT, alpha_vec, Lambda):
    def grad_func(h):
        #g = risk_aversion * np.matmul(QT, np.matmul(Q, h))
        g = np.dot(alpha_vec,-1.0)
        #g += 2 * (h - h0) * Lambda
        return np.asarray(g)
    return grad_func
############################################################################
def get_h_star(alpha_vec, h0, Q, QT, Lambda):
    # obj_func = get_obj_func(h0, alpha_vec, Lambda)
    # grad_func = get_grad_func(h0, alpha_vec, Lambda)
    obj_func = get_obj_func(h0, risk_aversion, alpha_vec, Q, Lambda)
    grad_func = get_grad_func(h0, risk_aversion, Q, QT, alpha_vec, Lambda)

    weightBounds = ((-maxStockWeight * bookSize, maxStockWeight * bookSize) for i in range(alpha_vec.size))
    weightBounds_lbfgs = [(-maxStockWeight * bookSize, maxStockWeight * bookSize) for i in range(alpha_vec.size)]

    #optimizer_result = scipy.optimize.fmin_l_bfgs_b(obj_func, h0, fprime=grad_func)
    #optimizer_result = minimize(obj_func, h0, bounds=weightBounds, method='Nelder-Mead',options={'maxiter':200})
    ####optimizer_result = minimize(obj_func, h0, bounds=weightBounds, method='Nelder-Mead', options={'maxiter': 200})

    #optimizer_result = minimize(obj_func, h0, method='Nelder-Mead', options={'maxiter': 10})
    #optimizer_result = minimize(obj_func, h0, method='Nelder-Mead')
    optimizer_result = scipy.optimize.fmin_l_bfgs_b(obj_func, h0,grad_func, bounds=weightBounds_lbfgs)

    #return optimizer_result.x
    return optimizer_result[0]
############################################################################

def main():

    #alphaweightsTS = pd.DataFrame(np.tile(optimizedWeights, (len(g_alphas_arr[0]), 1)))
    zero_data = np.ones(shape=(len(g_alphas_arr[0]), len(g_alphas_arr)))
    alphaweightsTS = pd.DataFrame(zero_data)
    alphaweightsTS = alphaweightsTS.div(alphaweightsTS.sum(axis=1), axis=0)

    testStart = int(g_alphas_arr[0].index.get_loc(testStartDate))
    optimEnd = int(g_alphas_arr[0].index.get_loc(optimEndDate))
    #######################################

    for counter in range(len(g_alphas_arr)):
        print("wMat: ", counter, "  ", g_alphas_arr[counter])
        NANs = g_alphas_arr[counter].isna().sum().sum()
        print("NANs: ", NANs)
        print("Percent NANs: ", NANs / g_alphas_arr[counter].size)
        aWeightsDF = alphaweightsTS[[counter]]  # .= .reindex(raw_alphas_dic[counter].index)
        #aWeightsDF.index = g_alphas_arr[counter].index
        if counter == 0:
            weightedAlpha = g_alphas_arr[counter].multiply(1, axis=0).fillna(0.0)
        else:
            weightedAlpha = weightedAlpha.add(g_alphas_arr[counter].multiply(1, axis=0).fillna(0.0))
            pass

    # #NEW NORMALIZATION!!
    # weightedAlpha.replace(0, np.nan, inplace=True)
    # weightedAlpha = RiskModelFunctions.hedgeGlobal(weightedAlpha)
    # weightedAlpha.replace(np.nan,0 , inplace=True)

    #weightedAlpha = weightedAlpha * bookSize
    turnoverAdj = weightedAlpha.diff().abs().sum(axis=1)

    combinedAlpha2 = pd.DataFrame(weightedAlpha.values * returnY.values, columns=weightedAlpha.columns, index=returnY.index)
    feeCombinedAlpha = (combinedAlpha2.sum(axis=1) - (turnoverAdj * feesBSP))
    feeCombinedAlpha.iloc[testStart:].cumsum().plot()

###################OPTIM NEW####################################
    tradeOptimDF = pd.DataFrame().reindex_like(weightedAlpha)
    # oldTrades = get_h_star(weightedAlpha.iloc[testStart].to_numpy(),
    #                        weightedAlpha.iloc[testStart].to_numpy(),
    #                        Q,
    #                        QT,
    #                        LambdaMatrix.iloc[testStart].to_numpy())
    oldPositions = weightedAlpha.iloc[testStart].to_numpy()
    tradeOptimDF.iloc[testStart] = oldPositions
    for index, row in weightedAlpha[testStart+1:].iterrows():
        print("index:", index)
        index_num = tradeOptimDF.index.get_loc(index)
        if index_num < len(tradeOptimDF)-5:  #100 500
            newPositions = get_h_star(row.to_numpy(),
                                   oldPositions,
                                   Q,
                                   QT,
                                   LambdaMatrix.iloc[testStart].to_numpy())
            #tradeOptimDF.iloc[index_num+1] = newTrades
            tradeOptimDF.loc[index] = newPositions
            oldPositions = newPositions


    weightedAlphaDiff = tradeOptimDF.diff().abs() ** 2
    turnoverModel = pd.DataFrame(LambdaMatrix.values * weightedAlphaDiff.values, columns=weightedAlpha.columns, index=weightedAlpha.index)

    tradeOptimReturns = pd.DataFrame(tradeOptimDF.values * returnY.values, columns=weightedAlpha.columns,
                                  index=returnY.index)
    tradeOptimReturns.iloc[testStart:, :].sum(axis=1).cumsum().plot()

    #tradeOptimReturnsAgg = pd.DataFrame(tradeOptimReturns.iloc[testStart:,:] - turnoverModel.iloc[testStart:,:]).sum(axis=1)
    tradeOptimPostFeesAgg = pd.DataFrame(tradeOptimReturns - turnoverModel).sum(axis=1)
    noOptimWithFees = pd.DataFrame(weightedAlpha - turnoverModel).sum(axis=1)
    tradeOptimPostFeesAgg.iloc[testStart:].cumsum().plot()
    #noOptimWithFees.iloc[testStart:].cumsum().plot()

    sharpe = (feeCombinedAlpha.mean() * 252.0) / (feeCombinedAlpha.std() * math.sqrt(252.0))
    print("feeCombinedAlpha FULL SHARPE:", sharpe)
    sharpe = (feeCombinedAlpha.iloc[optimEnd:].mean() * 252.0) / (feeCombinedAlpha.iloc[optimEnd:].std() * math.sqrt(252.0))
    print("feeCombinedAlpha TRAIN SHARPE:", sharpe)
    sharpe = (tradeOptimPostFeesAgg.iloc[testStart:].mean() * 252.0) / (
                tradeOptimPostFeesAgg.iloc[testStart:].std() * math.sqrt(252.0))
    print("QP OPTIM FEES TEST SHARPE:", sharpe)

    annFeeSharpe = (tradeOptimPostFeesAgg.iloc[testStart:].mean() * annCorrection) / (
                tradeOptimPostFeesAgg.iloc[testStart:].std() * math.sqrt(annCorrection))  #35040 #17520
    print("ANNUALIZED FEES TEST SHARPE:", annFeeSharpe)
    #((pd.DataFrame(combinedAlpha2).sum(axis=1) - (turnoverAdj * feesBSP)).cumsum()).plot()
    print("feesBSP: ",feesBSP)
    TEST = 1
    plt.show()

    return True

if __name__ == "__main__":
    main()