import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd


def two_anova(gt, pred):
    dataframe = pd.DataFrame({'Prediction': gt,
                              'GroundTruth': pred})
    print(dataframe)

    model = ols('GroundTruth ~ C(Prediction)', data=dataframe).fit()
    result = sm.stats.anova_lm(model, type=2)

    return result['PR(>F)'][0]
