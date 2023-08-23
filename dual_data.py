import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/Zed/Desktop/Dual_Camera/bp/S55.csv')
PPG = np.array(df['PPG'])

plt.plot(PPG)
plt.show()

