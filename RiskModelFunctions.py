import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA
import numpy.linalg as lin
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt

class RiskModelFunctions:
    @staticmethod
    def pcaConvertAlpha(ret, out, numberOfFactors=1, fillNanAlpha=True):
        ret = ret.fillna(0)

        if fillNanAlpha:
            out = out.fillna(0)

        pca = PCA(n_components=numberOfFactors)
        principalComponents = pca.fit(ret)

        eigVectors = pd.DataFrame(principalComponents.components_)
        #adjEigVectors = pd.DataFrame((pca.components_.T * pca.singular_values_).T)

        eigVectors.columns = out.columns
        # adjEigVectors.columns = out.columns
        # adjEigVectors.to_csv('C:\\Crypto\\QuoteEigVectors.csv')

        # factorMap = np.linalg.lstsq(eigVectors.transpose(), out.transpose(), rcond=None)[0]
        # outPostRisk = np.inner(factorMap.transpose(), eigVectors.transpose())

        tmp = np.linalg.lstsq(eigVectors.transpose(), out.transpose(), rcond=None)
        factorMap = tmp[0]
        #factorMap.plot()
        outPostRisk = np.inner(factorMap.transpose(), eigVectors.transpose())

        outPostRisk = pd.DataFrame(outPostRisk, index=out.index, columns=out.columns)
        # adjOut = outPostRisk - out
        adjOut = out - outPostRisk
        return adjOut

    @staticmethod
    def hedgeSubIndustries(industriesMap, signal):
        indCodes = industriesMap.iloc[:, 0]
        indGroups = np.unique(indCodes)

        for ind in indGroups:
            subsetSymbols = industriesMap.index[industriesMap.iloc[:, 0] == ind]
            intersection = list(set(signal.columns) & set(subsetSymbols))
            subSetIndOut = signal[list(intersection)]
            subSetIndOutMean = subSetIndOut.mean(axis=1)
            subset_out_minus_mean = subSetIndOut.sub(subSetIndOutMean.squeeze(), axis=0)
            # subset_out_minus_mean = subset_out_minus_mean.round(6)
            signal[subset_out_minus_mean.columns] = subset_out_minus_mean.copy()
        omm_abs = signal.abs()
        omm_abs_sum = omm_abs.sum(axis=1)
        out_normalzed = signal.div(omm_abs_sum.squeeze(), axis=0)
        # out_normalzed = out_normalzed.round(6)

        return out_normalzed

    @staticmethod
    def hedgeSubIndustries2(industriesMap, signal):
        indCodes = industriesMap.iloc[:, 0]
        indGroups = np.unique(indCodes)

        for ind in indGroups:

            subsetSymbols = industriesMap.index[industriesMap.iloc[:, 0] == ind]
            intersection = list(set(signal.columns) & set(subsetSymbols))
            subSetIndOut = signal[list(intersection)]
            subSetIndOutMean = subSetIndOut.mean(axis=1)
            subset_out_minus_mean = subSetIndOut.sub(subSetIndOutMean.squeeze(), axis=0)
            # subset_out_minus_mean = subset_out_minus_mean.round(6)
            signal[subset_out_minus_mean.columns] = subset_out_minus_mean
        omm_abs = signal.abs()
        omm_abs_sum = omm_abs.sum(axis=1)
        out_normalzed = signal.div(omm_abs_sum.squeeze(), axis=0)
        # out_normalzed = out_normalzed.round(6)

        return out_normalzed

    @staticmethod
    def hedgeSubIndustries3(industriesMap, signal):
        indCodes = industriesMap.iloc[:, 0]
        indGroups = np.unique(indCodes)

        for ind in indGroups:
            subsetSymbols = industriesMap.index[industriesMap.iloc[:, 0] == ind]
            intersection = list(set(signal.columns) & set(subsetSymbols))
            subSetIndOut = signal[list(intersection)]
            subSetIndOutMean = subSetIndOut.mean(axis=1)
            subset_out_minus_mean = subSetIndOut.sub(subSetIndOutMean.squeeze(), axis=0)
            sub_omm_abs = subSetIndOut.abs()
            sub_omm_abs_sum = sub_omm_abs.sum(axis=1)
            sub_out_normalzed = subSetIndOut.div(sub_omm_abs_sum.squeeze(), axis=0)
            # subset_out_minus_mean = subset_out_minus_mean.round(6)
            signal[subset_out_minus_mean.columns] = sub_out_normalzed.copy()
        #
        # omm_abs = signal.abs()
        # omm_abs_sum = omm_abs.sum(axis=1)
        # out_normalzed = signal.div(omm_abs_sum.squeeze(), axis=0)
        # # out_normalzed = out_normalzed.round(6)

        return signal

    @staticmethod
    def hedgeGlobal(signal):
        signal_mean = pd.Series(signal.mean(axis=1))
        signal_minus_mean = signal.sub(signal_mean.squeeze(), axis=0)
        # round to zero for super small values
        # out_minus_mean = out_minus_mean.round(6)
        omm_abs = signal_minus_mean.abs()
        omm_abs_sum = omm_abs.sum(axis=1)
        signal_normalized = signal_minus_mean.div(omm_abs_sum.squeeze(), axis=0)
        return signal_normalized

    @staticmethod
    def flipMode(signal):
        signal_mean = pd.Series(signal.mean(axis=0))
        signal_minus_mean = signal.sub(signal_mean.squeeze(), axis=1)
        omm_abs = signal_minus_mean.abs()
        omm_abs_sum = omm_abs.sum(axis=0)
        signal_normalized = signal_minus_mean.div(omm_abs_sum.squeeze(), axis=1)
        sizeAdjustment = signal_normalized.shape[0] / signal_normalized.shape[1]
        signal_normalized = signal_normalized * sizeAdjustment
        return signal_normalized

    @staticmethod
    def pcaMovingAvg(ret, out, numberOfFactors=1, expMAalpha=0.9, fillNanAlpha=True):
        ret = ret.fillna(0)

        if fillNanAlpha:
            out = out.fillna(0)

        pca = PCA()
        principalComponents = pca.fit(ret)

        principalDf = pd.DataFrame(principalComponents.components_)
        eigVectors = principalDf  # .head(numberOfFactors)
        print("matrix_rank(principalComponents.components_)", matrix_rank(principalComponents.components_))
        factorMap = np.linalg.lstsq(eigVectors.transpose(), out.transpose(), rcond=None)[0]

        factorMap = pd.DataFrame(factorMap)

        fmMovingAvg = factorMap.iloc[numberOfFactors:, :]
        fmMovingAvg = pd.DataFrame(fmMovingAvg).ewm(alpha=expMAalpha, axis=1).mean()
        factorMap.iloc[numberOfFactors:, :] = fmMovingAvg
        factorMap = factorMap.values

        outPostRisk = np.inner(factorMap.transpose(), eigVectors.transpose())
        outPostRisk = pd.DataFrame(outPostRisk, index=out.index, columns=out.columns)
        # adjOut = outPostRisk - out
        adjOut = out - outPostRisk
        return adjOut

    @staticmethod
    def icaConvertAlpha(ret, out, numberOfFactors=1, fillNanAlpha=True):
        ret = ret.fillna(0)

        if fillNanAlpha:
            out = out.fillna(0)

        ica = FastICA(n_components=numberOfFactors)
        components = ica.fit(ret)

        indComponents = pd.DataFrame(components.components_)
        #adjEigVectors = pd.DataFrame((pca.components_.T * pca.singular_values_).T)

        indComponents.columns = out.columns
        #adjEigVectors.columns = out.columns
        #adjEigVectors.to_csv('C:\\Crypto\\QuoteEigVectors.csv')

        # factorMap = np.linalg.lstsq(eigVectors.transpose(), out.transpose(), rcond=None)[0]
        # outPostRisk = np.inner(factorMap.transpose(), eigVectors.transpose())

        tmp = np.linalg.lstsq(indComponents.transpose(), out.transpose(), rcond=None)
        factorMap = tmp[0]
        outPostRisk = np.inner(factorMap.transpose(), indComponents.transpose())

        outPostRisk = pd.DataFrame(outPostRisk, index=out.index, columns=out.columns)
        # adjOut = outPostRisk - out
        adjOut = out - outPostRisk
        return adjOut

    @staticmethod
    def getPCAfactorCovMatrix(ret, numberOfFactors=5 ):
        ret = ret.fillna(0)

        pca = PCA(n_components=numberOfFactors)
        principalComponents = pca.fit(ret)

        eigVectors = pd.DataFrame(principalComponents.components_)
        eigVectors.columns = ret.columns

        factorReturns = np.linalg.lstsq(eigVectors.transpose(), ret.transpose(), rcond=None)[0].T
        return pd.DataFrame(factorReturns).cov(), eigVectors