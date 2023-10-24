import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.anova as anova
import pandas as pd
import time
import numpy as np


def one_anova(gt, pred):
    dataframe = pd.DataFrame({'Prediction': gt,
                              'GroundTruth': pred})
    # print(dataframe)

    start = time.time()
    model = ols('GroundTruth ~ C(Prediction)', data=dataframe).fit()
    result = anova.anova_lm(model)
    end = time.time()
    # print(f'Time Used: {end - start}s')
    # print(result)
    if result['PR(>F)'][0] is np.nan:
        return 1
    else:
        return result['PR(>F)'][0]

#
# if __name__ == '__main__':
#     gt = [14, 15, 16, 17, 18, 20, 14, 15, 16, 17, 18, 20]
#     pred = [14, 15, 16, 17, 18, 20, 14, 15, 16, 17, 19, 20]
#
#     p = two_anova(gt, pred)
