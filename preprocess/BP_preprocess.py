import os

print(os.cpu_count())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from video_preprocess import preprocess_raw_video, count_frames


def data_process(data_type, device_type, image=str(), dim=36):
    ############## Data folder path setting ##############
    if device_type == "local":
        data_folder_path = "C:/Users/Zed/Desktop/V4V/"
    else:
        data_folder_path = "/edrive2/zechenzh/V4V/"

    if data_type == "train":
        video_folder_path = f'{data_folder_path}Phase1_data/Videos/train/'
        BP_folder_path = f'{data_folder_path}Phase1_data/Ground_truth/BP_raw_1KHz/'
    elif data_type == "test":
        video_folder_path = f'{data_folder_path}Phase2_data/Videos/test/'
        BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/test_set_bp/'
    else:
        video_folder_path = f'{data_folder_path}Phase1_data/Videos/valid/'
        BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/val_set_bp/'

    ############## Video processing ##############
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)
    video_file_path = video_file_path[0:10]
    num_video = len(video_file_path)
    print('Processing ' + str(num_video) + ' Videos')

    videos = [Parallel(n_jobs=12)(
        delayed(preprocess_raw_video)(video_folder_path + video, dim) for video in video_file_path)]
    videos = videos[0]


    tt_frame = 0
    for i in range(num_video):
        tt_frame += videos[i].shape[0] // 120 * 120

        ############## Systolic BP Extraction ##############
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)
    # BP_file_path = BP_file_path[6:10]

    frames = np.zeros(shape=(tt_frame, 6, dim, dim))
    BP_lf = np.zeros(shape=tt_frame)
    frame_ind = 0
    for i in range(num_video):
        temp_BP = np.loadtxt(BP_folder_path + BP_file_path[i])  # BP loading
        current_frames = videos[i].shape[0] // 120 * 120
        temp_BP_lf = np.zeros(current_frames)
        # Down-sample BP 1000Hz --> 25Hz
        for j in range(0, current_frames):
            temp_BP_lf[j] = mean(temp_BP[j * 40:(j + 1) * 40])

        # Systolic BP finding and linear interp
        temp_BP_lf_systolic_peaks, _ = find_peaks(temp_BP_lf, distance=10)
        temp_BP_lf_systolic_inter = np.zeros(current_frames)
        prev_index = 0
        for index in temp_BP_lf_systolic_peaks:
            y_interp = interp1d([prev_index, index], [temp_BP_lf[prev_index], temp_BP_lf[index]])
            for k in range(prev_index, index + 1):
                temp_BP_lf_systolic_inter[k] = y_interp(k)
            prev_index = index
        y_interp = interp1d([prev_index, current_frames - 1], [temp_BP_lf[prev_index], temp_BP_lf[current_frames - 1]])
        for l in range(prev_index, current_frames):
            temp_BP_lf_systolic_inter[l] = y_interp(l)
        temp_BP_lf_systolic_inter = gaussian_filter(temp_BP_lf_systolic_inter, sigma=120)
        BP_lf[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter

        # Video Batches
        frames[frame_ind:frame_ind + current_frames, :, :, :] = videos[i][0:current_frames, :, :, :]
        frame_ind += current_frames

    plt.plot(BP_lf)
    plt.show()
    frames = frames.reshape((-1, 10, 6, dim, dim))
    BP_lf = BP_lf.reshape((-1, 10))
    ############## Save the preprocessed model ##############
    if device_type == "remote":
        saving_path = '/edrive2/zechenzh/preprocessed_v4v_minibatch/'
    else:
        saving_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
    np.save(saving_path + data_type + '_frames_' + image + '.npy', frames)
    np.save(saving_path + data_type + '_BP_systolic.npy', BP_lf)


def only_BP(data_type, device_type, image=str(), dim=36):
    ############## Data folder path setting ##############
    if device_type == "local":
        data_folder_path = "C:/Users/Zed/Desktop/V4V/"
    else:
        data_folder_path = "/edrive2/zechenzh/V4V/"

    if data_type == "train":
        video_folder_path = f'{data_folder_path}Phase1_data/Videos/train/'
        BP_folder_path = f'{data_folder_path}Phase1_data/Ground_truth/BP_raw_1KHz/'
    elif data_type == "test":
        video_folder_path = f'{data_folder_path}Phase2_data/Videos/test/'
        BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/test_set_bp/'
    else:
        video_folder_path = f'{data_folder_path}Phase1_data/Videos/valid/'
        BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/val_set_bp/'

    ############## Video processing ##############
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)
    video_file_path = video_file_path[0:10]
    num_video = len(video_file_path)
    print('Processing ' + str(num_video) + ' Videos')

    videos = [Parallel(n_jobs=12)(
        delayed(count_frames)(video_folder_path + video) for video in video_file_path)]
    videos = videos[0]

    tt_frame = 0
    for i in range(num_video):
        tt_frame += videos[i]// 120 * 120

        ############## Systolic BP Extraction ##############
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)
    # BP_file_path = BP_file_path[6:10]

    BP_lf = np.zeros(shape=tt_frame)
    # BP_lf_25 = np.zeros(shape=tt_frame)

    frame_ind = 0
    for i in range(num_video):
        temp_BP = np.loadtxt(BP_folder_path + BP_file_path[i])  # BP loading
        current_frames = videos[i] // 120 * 120
        temp_BP_lf = np.zeros(current_frames)
        # Down-sample BP 1000Hz --> 25Hz
        for j in range(0, current_frames):
            temp_BP_lf[j] = mean(temp_BP[j * 40:(j + 1) * 40])

        # Systolic BP finding and linear interp
        temp_BP_lf_systolic_peaks, _ = find_peaks(temp_BP_lf, distance=10)
        temp_BP_lf_systolic_inter = np.zeros(current_frames)
        prev_index = 0
        for index in temp_BP_lf_systolic_peaks:
            y_interp = interp1d([prev_index, index], [temp_BP_lf[prev_index], temp_BP_lf[index]])
            for k in range(prev_index, index + 1):
                temp_BP_lf_systolic_inter[k] = y_interp(k)
            prev_index = index
        y_interp = interp1d([prev_index, current_frames - 1], [temp_BP_lf[prev_index], temp_BP_lf[current_frames - 1]])
        for l in range(prev_index, current_frames):
            temp_BP_lf_systolic_inter[l] = y_interp(l)
        # temp_BP_lf_systolic_inter_25 = gaussian_filter(temp_BP_lf_systolic_inter,sigma=25)
        temp_BP_lf_systolic_inter_120 = gaussian_filter(temp_BP_lf_systolic_inter, sigma=25)
        BP_lf[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter_120
        # BP_lf_25[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter_25
        # Video Batches
        frame_ind += current_frames

    plt.plot(BP_lf, label='sigma=120')
    # plt.plot(BP_lf_25, label='sigma=25')
    plt.legend()
    plt.show()
    BP_lf = BP_lf.reshape((-1, 10))
    ############## Save the preprocessed model ##############
    if device_type == "remote":
        saving_path = '/edrive2/zechenzh/preprocessed_v4v_minibatch/'
    else:
        saving_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
    np.save(saving_path + data_type + '_BP_systolic_a120.npy', BP_lf)


if __name__ == '__main__':
    # data_process('train', 'remote', 'face_large')
    # data_process('valid', 'remote', 'face_large')
    data_process('test', 'remote', 'face_large')
    # only_BP('train', 'local', 'face_large')
    # only_BP('valid', 'remote', 'face_large')
    # only_BP('test', 'remote', 'face_large')


