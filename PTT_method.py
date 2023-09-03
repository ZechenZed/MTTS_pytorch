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


def video_process(device_type):
    if device_type == "local":
        env_path = 'C:/Users/Zed/Desktop/'
    else:
        env_path = '/edrive2/zechenzh/'

    face_video_folder_path = env_path + 'Dual_Camera/face/'
    video_file_path = []
    for path in sorted(os.listdir(face_video_folder_path)):
        if os.path.isfile(os.path.join(face_video_folder_path, path)):
            video_file_path.append(path)
    video_file_path = video_file_path[0:1]
    print(video_file_path)

    for video in video_file_path:
        finger_frames = preprocess_finger(env_path + 'Dual_Camera/finger/' + video)
        frames, fps = preprocess_raw_video_unsupervised(face_video_folder_path + video)
        np.save(f'{env_path}/preprocessed_DC/face/{video[0:4]}.npy', frames)
        np.save(f'{env_path}/preprocessed_DC/finger/{video[0:4]}.npy', finger_frames)


def PTT(device_type):
    if device_type == "local":
        env_path = 'C:/Users/Zed/Desktop/preprocessed_DC/'
    else:
        env_path = '/edrive2/zechenzh/preprocessed_DC/'

    face_video_folder_path = env_path + 'face/'
    video_file_path = []
    for path in sorted(os.listdir(face_video_folder_path)):
        if os.path.isfile(os.path.join(face_video_folder_path, path)):
            video_file_path.append(path)

    print(video_file_path)

    print('Loading Face Frames')
    face_frames = np.load(env_path + 'face/S055.npy')
    print('Loading Finger Frames')
    finger_frames = np.load(env_path + 'finger/S055.npy')

    start_frame = 5000
    end_frame = min(len(face_frames), len(finger_frames))

    face_frames = face_frames[start_frame:end_frame]
    finger_frames = finger_frames[start_frame:end_frame]

    print('Processing CHROME')
    chrome_faceBVP = CHROME_DEHAAN(face_frames, 240)
    chrome_fingerBVP = CHROME_DEHAAN(finger_frames, 240)

    ICA_faceBVP = ICA_POH(face_frames, 240)
    ICA_fingerBVP = ICA_POH(finger_frames, 240)

    fig = plt.figure(figsize=(20, 18))
    print('Plotting figures')

    plt.plot(chrome_faceBVP, label='Chrome face')
    plt.plot(chrome_fingerBVP, label='Chrome finger')
    plt.plot(ICA_fingerBVP, label='ICA finger')
    plt.plot(ICA_faceBVP, label='ICA face')
    plt.legend()
    plt.show()

    chrome_peaks, _ = find_peaks(chrome_faceBVP, distance=5)
    chorme_finger_peaks, _ = find_peaks(chrome_fingerBVP, distance=5)
    ICA_peaks, _ = find_peaks(chrome_faceBVP, distance=5)
    ICA_finger_peaks, _ = find_peaks(chrome_fingerBVP, distance=5)
    if len(ICA_peaks) == len(ICA_finger_peaks):
        print('synchronized')


if __name__ == '__main__':
    device_type = 'remote'
    # PTT(device_type)
    video_process(device_type)
