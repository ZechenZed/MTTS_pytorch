import matplotlib.pyplot as plt
import numpy as np
import os
import global_align as ga

# print(os.cpu_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import pandas as pd
from preprocess.video_preprocess import preprocess_raw_video_unsupervised, preprocess_finger
from unsupervised_method.CHROME import CHROME_DEHAAN
from unsupervised_method.GREEN import GREEN
from unsupervised_method.ICA_POH import ICA_POH
from scipy.signal import find_peaks
from unsupervised_method.POS_WANG import POS_WANG


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
        # finger_frames = preprocess_finger(env_path + 'Dual_Camera/finger/' + video)
        frames, fps = preprocess_raw_video_unsupervised(face_video_folder_path + video)
        np.save(f'{env_path}/preprocessed_DC/face/{video[0:4]}.npy', frames)
        # np.save(f'{env_path}/preprocessed_DC/finger/{video[0:4]}.npy', finger_frames)


def plotting(method_name, faceBVP, fingerBVP):
    face_peaks, _ = find_peaks(faceBVP, distance=80)
    finger_peaks, _ = find_peaks(fingerBVP, distance=80)
    print(f'Number of face peaks:{len(face_peaks)} and Number of finger peaks{len(finger_peaks)}')
    plt.plot(face_peaks, faceBVP[face_peaks], 'o', label=f'{method_name} face peaks')
    plt.plot(finger_peaks, fingerBVP[finger_peaks], 'x', label=f'{method_name}finger peaks')
    plt.plot(faceBVP, label=f'{method_name}face')
    plt.plot(fingerBVP, label=f'{method_name}finger')
    plt.legend()
    plt.show()


def normalization(bvp):
    bvp = bvp / max(abs(bvp))
    return bvp


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
    end_frame = int(min(len(face_frames), len(finger_frames)) / 4)

    face_frames = face_frames[start_frame:end_frame]
    finger_frames = finger_frames[start_frame:end_frame]

    print('Processing CHROME')
    chrome_faceBVP = CHROME_DEHAAN(face_frames, 240)
    chrome_fingerBVP = CHROME_DEHAAN(finger_frames, 240)
    chrome_faceBVP = normalization(chrome_faceBVP)
    chrome_fingerBVP = normalization(chrome_fingerBVP)
    plotting('Chrome', chrome_faceBVP, chrome_fingerBVP)

    ICA_faceBVP = ICA_POH(face_frames, 240)
    ICA_fingerBVP = ICA_POH(finger_frames, 240)
    ICA_faceBVP = normalization(ICA_faceBVP)
    ICA_fingerBVP = normalization(ICA_fingerBVP)
    plotting('ICA', ICA_faceBVP, ICA_fingerBVP)

    POS_faceBVP = POS_WANG(face_frames, 240)
    POS_fingerBVP = POS_WANG(finger_frames, 240)
    POS_faceBVP = normalization(POS_faceBVP)
    POS_fingerBVP = normalization(POS_fingerBVP)
    plotting('POS', POS_faceBVP, POS_fingerBVP)


if __name__ == '__main__':
    device_type = 'local'
    # video_process(device_type)
    PTT(device_type)
