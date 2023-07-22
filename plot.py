import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error

valid = np.load('C:/Users/Zed/Desktop/V4V/preprocessed_v4v/valid_BP_systolic.npy')
valid = valid.reshape(-1, 1)
after_valid = gaussian_filter(valid, sigma=25)
print(mean_squared_error(valid,after_valid))
plt.plot(valid,label='before')
plt.plot(after_valid,label='after')
plt.show()
