import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.anova as anova
from scipy.stats import f_oneway
import pandas as pd
import time
import numpy as np


def one_anova(gt, pred):
    # dataframe = pd.DataFrame({'Prediction': gt,
    #                           'GroundTruth': pred})
    # print(dataframe)
    #
    # # start = time.time()
    # model = ols('GroundTruth ~ C(Prediction) + C(GroundTruth) + C(Prediction):C(GroundTruth)', data=dataframe).fit()
    # result = anova.anova_lm(model, type=2)
    # # end = time.time()
    # # print(f'Time Used: {end - start}s')
    # print(result)
    # MSR = max([0, result['mean_sq'][0]])
    # MSE = max([0, result['mean_sq'][1]])
    # k = 2
    # n = 12
    # ICC = (MSR - MSE) / (MSR + MSE * (k - 1))
    # print(ICC)
    # if np.isnan(result['PR(>F)'][0]):
    #     return 1
    # else:
    #     return result['PR(>F)'][0]
    result = f_oneway(gt, pred)
    return result[1]
#
# if __name__ == '__main__':
#     gt = [14, 15, 16, 17, 18, 20, 14, 15, 16, 17, 19, 20]
#     pred = [0, 15, 16, 17, 0, 20, 14, 15, 1, 17, 19,0]
#
#     p = one_anova(gt, pred)
