from scipy.stats import f_oneway
import pandas as pd
import time
import numpy as np


def ICC(gt, pred):
    var_gt = np.var(gt)
    var_pred = np.var(pred)
    return var_gt / (var_gt + var_pred)
#
# if __name__ == '__main__':
#     gt = [14, 15, 16, 17, 18, 20, 14, 15, 16, 17, 19, 20]
#     pred = [0, 15, 16, 17, 0, 20, 14, 15, 1, 17, 19,0]
#
#     p = one_anova(gt, pred)
