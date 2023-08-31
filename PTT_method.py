import matplotlib.pyplot as plt
import numpy as np
import os

# print(os.cpu_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import pandas as pd
from preprocess.video_preprocess import preprocess_raw_video_unsupervised, preprocess_finger
from unsupervised_method.CHROME import CHROME_DEHAAN
from unsupervised_method.GREEN import GREEN
from unsupervised_method.ICA_POH import ICA_POH
from scipy.signal import find_peaks


def PTT():
    env_path = '/edrive2/zechenzh/dual_camera/'
    face_video_path = env_path + 'face/Subject S055 3487906/iphone_2022_10_20_16_43_10.4220_0.mp4'
    finger_video_path = env_path + 'finger/Subject S055 3487906/iphone_2022_10_20_16_43_10.2670_0.mp4'

    # bio_path = 'C:/Users/Zed/Desktop/Dual_Camera/bp/S55.csv'
    # bio = pd.read_csv(bio_path)['SystolicBP']

    # finger_frames = preprocess_finger(finger_video_path)
    # frames, fps = preprocess_raw_video_unsupervised(face_video_path)
    #
    # np.save(env_path+'/face/Subject S055 3487906/s55_face.npy', frames)
    # np.save(env_path+'/finger/Subject S055 3487906/s55_finger.npy', finger_frames)
    print('Loading Face Frames')
    face_frames = np.load(env_path + '/face/Subject S055 3487906/s55_face.npy')[5000:-1]
    print('Loading Finger Frames')
    finger_frames = np.load(env_path + '/finger/Subject S055 3487906/s55_finger.npy')[5000:-1]

    print('Processing CHROME')
    chrome_faceBVP = CHROME_DEHAAN(face_frames, 240)
    chrome_fingerBVP = CHROME_DEHAAN(finger_frames, 240)

    print('Processing ICA')
    ICA_faceBVP = ICA_POH(face_frames, 240)
    ICA_fingerBVP = ICA_POH(finger_frames, 240)

    fig = plt.figure(figsize=(20, 18))
    print('Plotting figures')
    plt.plot(chrome_faceBVP, label='Chrome face')
    plt.plot(chrome_fingerBVP, label='Chrome finger')
    plt.plot(ICA_faceBVP, label='ICA face')
    plt.plot(ICA_fingerBVP, label='ICA finger')

    print('Saving plot')
    plt.savefig(env_path + 'PTTplot.png',dpi=1200)
    # plt.legend()
    # plt.show()
    #
    # peaks, _ = find_peaks(BVP, distance=5)
    # finger_peaks, _ = find_peaks(finger_BVP, distance=5)


if __name__ == '__main__':
    PTT()
