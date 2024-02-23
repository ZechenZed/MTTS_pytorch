import numpy as np
from scipy.stats import pearsonr
from joblib import Parallel, delayed


# def ICC(GT, Pred):
#     def icc(gt, pred):
#         var_gt = np.var(gt)
#         var_pred = np.var(pred)
#         return var_gt / (var_gt + var_pred)
#
#     p = [Parallel(n_jobs=12)(
#         delayed(icc)(GT[i:i + 200], Pred[i:i + 200]) for i in range(0, len(GT), 200))]
#     p = p[0]
#     p = np.mean(p)
#     return p if p != 1 else -1
def ICC_A_1(data):
    """
    Calculate the single rater agreement intraclass correlation coefficient (ICC) and its confidence interval.

    Parameters:
    - data: numpy array of ratings, with rows as items and columns as raters. Missing values should be NaN.
    - alpha: Type I error rate for the confidence interval (default 0.05).

    Returns:
    - ICC: Intraclass correlation coefficient for single rater agreement.
    - LB: Lower bound of the confidence interval for ICC.
    - UB: Upper bound of the confidence interval for ICC.
    """

    n, k = data.shape  # n is number of items, k is number of raters

    # Mean per item, mean per rater, and overall mean
    mean_per_item = np.nanmean(data, axis=1)
    mean_per_rater = np.nanmean(data, axis=0)
    mean_overall = np.nanmean(data)

    # Sum of squares between items
    ss_between = k * np.nansum((mean_per_item - mean_overall) ** 2)

    # Sum of squares within items
    ss_within = np.nansum((data - np.tile(mean_per_item, (k, 1)).T) ** 2)

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    # ICC calculation
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)

    return icc


def pearson(gt, pred):
    ro = pearsonr(gt, pred)[0]
    if np.isnan(ro):
        ro = -1
    return ro


def cMAE(gt, pred):
    return sum(abs(pred - gt)) / pred.shape[0]
