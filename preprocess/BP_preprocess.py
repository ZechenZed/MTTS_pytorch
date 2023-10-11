import os
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt, resample
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from video_preprocess import preprocess_raw_video, count_frames

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def data_process(data_type, device_type, image=str(), dim=36):
    ############## Data folder path setting ##############
    if device_type == 'local':
        data_folder_path = "C:/Users/Zed/Desktop/V4V/"
    elif device_type == 'disk':
        data_folder_path = 'D:/V4V/'
    else:
        data_folder_path = "/edrive2/zechenzh/V4V/"

    # if data_type == "train":
    #     video_folder_path = f'{data_folder_path}Phase1_data/Videos/train/'
    #     BP_folder_path = f'{data_folder_path}Phase1_data/Ground_truth/BP_raw_1KHz/'
    # elif data_type == "test":
    #     video_folder_path = f'{data_folder_path}Phase2_data/Videos/test/'
    #     BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/test_set_bp/'
    # else:
    #     video_folder_path = f'{data_folder_path}Phase1_data/Videos/valid/'
    #     BP_folder_path = f'{data_folder_path}Phase2_data/blood_pressure/val_set_bp/'

    if data_type == "train":
        video_folder_path = f'{data_folder_path}Phase1_data/Videos/new_train/'
        BP_folder_path = f'{data_folder_path}Phase1_data/Ground_truth/new_train_BP/'
    else:
        video_folder_path = f'{data_folder_path}Phase1_data/Videos/new_test/'
        BP_folder_path = f'{data_folder_path}Phase1_data/Ground_truth/new_test_BP/'

    ############## Video processing ##############
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)

    # video_file_path = video_file_path[0:10]
    num_video = len(video_file_path)
    print('Processing ' + str(num_video) + ' Videos')

    videos = [Parallel(n_jobs=20)(
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

    # BP_file_path = BP_file_path[0:1]

    print(tt_frame)
    frames = np.zeros(shape=(tt_frame, 6, dim, dim))
    BP_lf = np.zeros(shape=tt_frame)
    frame_ind = 0
    for i in range(num_video):
        print(f'BP process on {BP_file_path[i]}')
        temp_BP = np.loadtxt(BP_folder_path + BP_file_path[i])  # BP loading
        # plt.plot(temp_BP,label='opiginal')
        temp_BP = gaussian_filter(temp_BP,10) # Smoothing before downsampling to better find the peaks
        # plt.plot(temp_BP,label='after')
        # plt.legend()
        # plt.show()
        BP_lf_len = len(temp_BP) // 40
        temp_BP_lf = np.zeros(BP_lf_len)  # Create new lf BP

        ################# Down-sample BP 1000Hz --> 25Hz using moving mean window ################
        for j in range(0, len(temp_BP_lf)):
            temp_BP_lf[j] = np.average(temp_BP[j * 40:(j + 1) * 40])
        # plt.plot(temp_BP_lf)

        video_len = videos[i].shape[0]
        current_frames = min(BP_lf_len, video_len)
        temp_BP_lf = temp_BP_lf[0:current_frames]

        ############# Systolic BP finding and linear interp #############
        temp_BP_lf_systolic_peaks, _ = find_peaks(temp_BP_lf, distance=15)
        # plt.plot(temp_BP_lf_systolic_peaks,temp_BP_lf[temp_BP_lf_systolic_peaks],'o')
        # plt.show()

        temp_BP_lf_systolic_inter = np.zeros(current_frames)
        first_index = temp_BP_lf_systolic_peaks[0]
        prev_index = 0
        for index in temp_BP_lf_systolic_peaks:
            y_interp = interp1d([prev_index, index], [temp_BP_lf[prev_index], temp_BP_lf[index]])
            for k in range(prev_index, index + 1):
                temp_BP_lf_systolic_inter[k] = y_interp(k)
            prev_index = index
        y_interp = interp1d([prev_index, current_frames - 1],
                            [temp_BP_lf[prev_index], temp_BP_lf[current_frames - 1]])
        for l in range(prev_index, current_frames):
            temp_BP_lf_systolic_inter[l] = y_interp(l)
        temp_BP_lf_systolic_inter = temp_BP_lf_systolic_inter[first_index:-1]

        ################ Find incorrect BP values ################
        invalid_index_BP = np.where((temp_BP_lf_systolic_inter < 60) | (temp_BP_lf_systolic_inter > 250))[0]

        if len(invalid_index_BP) != 0:
            current_frames = invalid_index_BP[0] // 120 * 120
        else:
            current_frames = len(temp_BP_lf_systolic_inter) // 120 * 120

        if current_frames == 0 or current_frames < 0:
            print(f'Skip video: {BP_file_path[i]}')
            continue
        else:
            temp_BP_lf_systolic_inter = temp_BP_lf_systolic_inter[0:current_frames]

        ############# BP smoothing #############
        # plt.plot(temp_BP_lf_systolic_inter)
        temp_BP_lf_systolic_inter = gaussian_filter(temp_BP_lf_systolic_inter, sigma=3)
        # plt.plot(temp_BP_lf_systolic_inter)
        # plt.legend()
        # plt.show()
        BP_lf[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter

        ############# Video Batches #############
        frames[frame_ind:frame_ind + current_frames] = videos[i][first_index:first_index + current_frames]
        frame_ind += current_frames

    ind_BP_rest = np.where(BP_lf == 0)[0][0]
    print(f'Valid train dataset length:{ind_BP_rest}')
    BP_lf = BP_lf[0:ind_BP_rest]
    frames = frames[0:ind_BP_rest]

    frames = frames.reshape((-1, 10, 6, dim, dim))
    BP_lf = BP_lf.reshape((-1, 10))
    print(f'Shape of BP_lf{BP_lf.shape}')
    ############## Save the preprocessed model ##############
    saving_path = ''
    if device_type == 'remote':
        saving_path = '/edrive1/zechenzh/preprocessed_v4v/'
    elif device_type == 'local':
        saving_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'
    np.save(saving_path + data_type + '_frames_' + image + '.npy', frames)
    np.save(saving_path + data_type + '_BP_systolic.npy', BP_lf)


def data_process_DC(device_type, image=str(), dim=128):
    ############## Data folder path setting ##############
    if device_type == 'local':
        data_folder_path = "C:/Users/Zed/Desktop/DC/"
    elif device_type == 'disk':
        data_folder_path = 'D:/DC/'
    else:
        data_folder_path = "/edrive1/zechenzh/DC/"

    video_folder_path = f'{data_folder_path}face/'
    BP_folder_path = f'{data_folder_path}bp/'

    ############## Video processing ##############
    video_file_path = []
    for path in sorted(os.listdir(video_folder_path)):
        if os.path.isfile(os.path.join(video_folder_path, path)):
            video_file_path.append(path)

    # video_file_path = video_file_path[0:1]
    num_video = len(video_file_path)
    print('Processing ' + str(num_video) + ' Videos')

    videos = [Parallel(n_jobs=16)(
        delayed(preprocess_raw_video)(video_folder_path + video, dim) for video in video_file_path)]
    videos = videos[0]

    tt_frame = 0
    for i in range(num_video):
        tt_frame += videos[i].shape[0] // 240 * 240

    ############## Systolic BP Extraction ##############
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)

    # BP_file_path = BP_file_path[0:1]
    test_frames = tt_frame // 4
    train_frames = test_frames * 3
    frames_train = np.zeros(shape=(train_frames, 6, dim, dim))
    frames_test = np.zeros(shape=(test_frames, 6, dim, dim))
    BP_lf_train = np.zeros(shape=train_frames)
    BP_lf_test = np.zeros(shape=test_frames)
    frame_ind_train = 0
    frame_ind_test = 0
    for i in range(num_video):
        print(f'BP process on {BP_file_path[i]}')
        temp_BP = pd.read_csv(BP_folder_path + BP_file_path[i])['MeanBP']

        lf_len = int(len(temp_BP) / 1000 * 240)
        temp_BP_lf = resample(temp_BP, lf_len)
        # plt.plot(temp_BP_lf,label='before')
        temp_BP_lf = gaussian_filter(temp_BP_lf, 240)
        # plt.plot(temp_BP_lf,label='after')
        # plt.show()
        video_len = videos[i].shape[0]

        current_frames = min(lf_len, video_len)
        current_frames = current_frames // 240 * 240
        curr_test_frames = current_frames // 4
        curr_train_frames = curr_test_frames * 3
        ############# BP smoothing #############
        BP_lf_train[frame_ind_train:frame_ind_train + curr_train_frames] = temp_BP_lf[0:curr_train_frames]
        BP_lf_test[frame_ind_test:frame_ind_test + curr_test_frames] = \
            temp_BP_lf[curr_train_frames:curr_train_frames+curr_test_frames]

        ############# Video Batches #############
        frames_train[frame_ind_train:frame_ind_train + curr_train_frames] = videos[i][0:curr_train_frames]
        frames_test[frame_ind_test:frame_ind_test + curr_test_frames] = \
            videos[i][curr_train_frames:curr_train_frames+curr_test_frames]

        frame_ind_train += curr_train_frames
        frame_ind_test += curr_test_frames
    #
    ind_BP_rest = np.where(BP_lf_train == 0)[0][0]
    BP_lf_train = BP_lf_train[0:ind_BP_rest]
    frames_train = frames_train[0:ind_BP_rest]

    ind_BP_rest = np.where(BP_lf_test == 0)[0][0]
    BP_lf_test = BP_lf_test[0:ind_BP_rest]
    frames_test = frames_test[0:ind_BP_rest]

    # plt.plot(BP_lf_train,label='training data')
    # plot_test = np.arange(train_frames,int(train_frames+test_frames))
    # plt.plot(plot_test, BP_lf_test,label='testing data')
    # plt.legend()
    # plt.show()

    frames_train = frames_train.reshape((-1, 10, 6, dim, dim))
    frames_test = frames_test.reshape((-1, 10, 6, dim, dim))

    BP_lf_train = BP_lf_train.reshape((-1, 10))
    BP_lf_test = BP_lf_test.reshape((-1, 10))

    ############## Save the preprocessed model ##############

    saving_path = '/edrive1/zechenzh/preprocessed_DC/'
    np.save(saving_path + 'train_frames_' + image + '.npy', frames_train)
    np.save(saving_path + 'test_frames_' + image + '.npy', frames_test)
    np.save(saving_path + 'train_BP_systolic.npy', BP_lf_train)
    np.save(saving_path + 'test_BP_systolic.npy', BP_lf_test)


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
        tt_frame += videos[i] // 120 * 120

        ############## Systolic BP Extraction ##############
    BP_file_path = []
    for path in sorted(os.listdir(BP_folder_path)):
        if os.path.isfile(os.path.join(BP_folder_path, path)):
            BP_file_path.append(path)
    # BP_file_path = BP_file_path[6:10]

    BP_lf = np.zeros(shape=tt_frame)
    # BP_lf_25 = np.zeros(shape=tt_frame)
    # BP_lf_120 = np.zeros(shape=tt_frame)
    # BP_lf_med = np.zeros(shape=tt_frame)

    frame_ind = 0
    for i in range(num_video):
        print(f'Processing Video {BP_file_path[i]}')
        temp_BP = np.loadtxt(BP_folder_path + BP_file_path[i])  # BP loading
        temp_BP = np.where(temp_BP > 0, temp_BP, 0)
        invalid_index = np.where(temp_BP == 0)
        temp_BP = np.delete(temp_BP, invalid_index)

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

        # Different filtering
        # temp_BP_med = medfilt(temp_BP_lf_systolic_inter, 51)
        # temp_BP_lf_systolic_inter_25 = gaussian_filter(temp_BP_lf_systolic_inter, sigma=25)
        # temp_BP_lf_systolic_inter_120 = gaussian_filter(temp_BP_lf_systolic_inter, sigma=120)
        # BP_lf_med[frame_ind:frame_ind + current_frames] = temp_BP_med
        # BP_lf_25[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter_25
        # BP_lf_120[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter_120

        BP_lf[frame_ind:frame_ind + current_frames] = temp_BP_lf_systolic_inter
        frame_ind += current_frames

    fig = plt.figure(figsize=(10, 12))
    plt.plot(BP_lf, label='original')
    # plt.plot(BP_lf_25, label='sigma = 25')
    # plt.plot(BP_lf_120, label='sigma = 120')
    # plt.plot(BP_lf_med, label='median 51')
    plt.savefig('/edrive2/zechenzh/PTTplot.jpg', dpi=1200)

    plt.legend()
    plt.show()

    # BP_lf = BP_lf.reshape((-1, 10))
    # BP_lf_25 = BP_lf_25.reshape((-1, 10))
    # BP_lf_120 = BP_lf_120.reshape((-1, 10))

    ############## Save the preprocessed model ##############
    # if device_type == "remote":
    #     saving_path = '/edrive2/zechenzh/preprocessed_v4v_minibatch/'
    # else:
    #     saving_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/'


if __name__ == '__main__':
    # data_process('valid', 'remote', 'face_large')
    data_process('train', 'remote', 'face_large')
    data_process('test', 'remote', 'face_large')
    # only_BP('train', 'remote', 'face_large')
    # only_BP('valid', 'remote', 'face_large')
    # only_BP('test', 'local', 'face_large')
    # data_process_DC('local', 'face_large')
