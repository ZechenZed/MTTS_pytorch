import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter, percentile_filter
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, filtfilt

# path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
path = '/edrive2/zechenzh/preprocessed_v4v_minibatch/'
train_frames = np.load(path + 'train_frames_face_large.npy')
train_BP = np.load(path + 'train_BP_systolic_a25.npy')

frames = train_frames.reshape((-1, 6, 36, 36))
BP = train_BP.reshape(-1)
ind_BP = np.where(BP == 0)[0][0]
print(ind_BP)
frames = frames[0:ind_BP]
BP = BP[0:ind_BP]
frames = frames.reshape((-1, 10, 6, 36, 36))
BP = BP.reshape((-1, 10))
np.save(path + 'train_frames_face_large.npy', frames)
np.save(path + 'train_BP_systolic_a25.npy', BP)

valid_frames = np.load(path+'valid_frames_face_large.npy')
valid_BP = np.load(path+'valid_BP_systolic_a25.npy')

frames = valid_frames.reshape((-1, 6, 36, 36))
BP = valid_BP.reshape(-1)
ind_BP = np.where(BP == 0)[0][0]
print(ind_BP)
frames = frames[0:ind_BP]
BP = BP[0:ind_BP]
frames = frames.reshape((-1, 10, 6, 36, 36))
BP = BP.reshape((-1, 10))
np.save(path + 'valid_frames_face_large.npy', frames)
np.save(path + 'valid_BP_systolic_a25.npy', BP)

test_frames = np.load(path+'test_frames_face_large.npy')
test_BP = np.load(path+'test_BP_systolic_a25.npy')

frames = test_frames.reshape((-1, 6, 36, 36))
BP = test_BP.reshape(-1)
ind_BP = np.where(BP == 0)[0][0]
print(ind_BP)
frames = frames[0:ind_BP]
BP = BP[0:ind_BP]
frames = frames.reshape((-1, 10, 6, 36, 36))
BP = BP.reshape((-1, 10))
np.save(path + 'test_frames_face_large.npy', frames)
np.save(path + 'test_BP_systolic_a25.npy', BP)
