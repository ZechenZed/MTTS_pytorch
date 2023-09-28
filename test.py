import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter, percentile_filter
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

label = np.random.randint(0, 10, 20)
prediction = np.ones(20)

ro = pearsonr(label, prediction)
print(np.isnan(ro[0]))
print('')
