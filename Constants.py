

#### RISK MODEL #####
GLOBAL_RISK_MODEL = 'Global'
SUB_INDUSTRY_RISK_MODEL = 'subIndustry'
PCA_RISK_MODEL = 'PCA'
EXP_PCA_RISK_MODEL = 'exponentialPCA'
FLIP_MODE_MODEL = 'flipMode'
ICA_RISK_MODEL = 'independentComponent'

MASTER_FEATURES_LIST = ["open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "dollars_traded",
                        "adv20",
                        "returns",
                        "hist_vol_10",
                        "hist_vol_20",
                        "hist_vol_30",
                        "hist_vol_90"]


MASTER_GPFUNCTIONS_LIST = [ "ts_zscore",
                            "log_diff",
                            "s_log_1p",
                            "df_max",
                            "df_min",
                            "npfadd",
                            "npfsub",
                            "npfmul",
                            "npfdiv",
                            "add",
                            "subtract",
                            "multiply",
                            "divide",
                            "negative",
                            "log",
                            "log10",
                            "sqrt",
                            "square",
                            "true_divide",
                            "SignedPower",
                            "Sign",
                            "sma",
                            "delta",
                            "stddev",
                            "ts_min",
                            "ts_max",
                            "ts_sum",
                            "delay",
                            "ts_skewness",
                            "ts_kurtosis",
                            "ArgMax",
                            "ArgMin",
                            "Product",
                            "ts_rank",
                            "rank",
                            "Abs",
                            "Decay_lin",
                            "Inverse",
                            "correlation",
                            "covariance" ]