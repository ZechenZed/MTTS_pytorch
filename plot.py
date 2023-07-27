import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter, percentile_filter
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt

# valid = np.load('C:/Users/Zed/Desktop/V4V/preprocessed_v4v/valid_BP_systolic_test.npy')
# # valid_120 = np.load('C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_BP_systolic_a120.npy')
# valid = valid.reshape(-1)
# valid = np.where(valid > 0, valid, 0)
# index = np.where(valid == 0)
# valid = np.delete(valid, index)
# index = np.where(valid == 0)[0]
# print(index)
# # valid_120 = valid_120.reshape(-1,1)
# fs = 25
# # after_valid = gaussian_filter(valid, sigma=120)
# # after_valid = percentile_filter(after_valid, 90, size=90)
# # print(mean_squared_error(valid, after_valid))
# plt.plot(valid, label='before')
# # plt.plot(valid_120, label='after')
# plt.show()

data_folder_path = "C:/Users/Zed/Desktop/V4V/"
video_folder_path = f'{data_folder_path}Phase1_data/Videos/train/'
BP_folder_path = f'{data_folder_path}Phase1_data/Ground_truth/BP_raw_1KHz/'
# video_folder_path = f'{data_folder_path}Phase2_data/Videos/test/'
# BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/test_set_bp/'
# video_folder_path = f'{data_folder_path}Phase1_data/Videos/valid/'
# BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/val_set_bp/'
# valid_120 = np.load('C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_BP_systolic_a120.npy')

############## Systolic BP Extraction ##############
BP_file_path = []
for path in sorted(os.listdir(BP_folder_path)):
    if os.path.isfile(os.path.join(BP_folder_path, path)):
        BP_file_path.append(path)

for i in range(724):
    temp_BP = np.loadtxt(BP_folder_path + BP_file_path[i])  # BP loading
    if min(temp_BP) < 0:
        print(f'warning {BP_file_path[i]} is incorrect')
