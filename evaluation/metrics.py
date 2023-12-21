import numpy as np
from scipy.stats import pearsonr
from joblib import Parallel, delayed


def ICC(GT, Pred):
    def icc(gt, pred):
        var_gt = np.var(gt)
        var_pred = np.var(pred)
        return var_gt / (var_gt + var_pred)

    p = [Parallel(n_jobs=12)(
        delayed(icc)(GT[i:i + 200], Pred[i:i + 200]) for i in range(0, len(GT), 200))]
    p = p[0]
    p = np.mean(p)
    return p if p != 1 else -1


def pearson(gt, pred):
    ro = pearsonr(gt, pred)[0]
    if np.isnan(ro):
        ro = -1
    return ro


def cMAE(gt, pred):
    return sum(abs(pred - gt)) / pred.shape[0]
