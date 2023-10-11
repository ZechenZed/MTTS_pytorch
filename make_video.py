import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_video():
    test_array = np.load('C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_frames_face_large.npy')
    test_array = test_array.reshape((209 * 10, 6, 72, 72))[:, 3:, :, :]
    trans_array = np.transpose(test_array, (0, 2, 3, 1))
    reshaped_arr = trans_array.reshape(2090, 72, 72, 3)
    print(test_array.shape)
    fps = 25
    height = 72
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_frames.mp4'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (72, 72))
    for image in reshaped_arr:
        plt.matshow(image)
        plt.show()
        video_writer.write(image)

    # Release the VideoWriter object
    video_writer.release()


def plot_BP():
    path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_BP_systolic.npy'
    BP = np.load(path)
    BP = BP.reshape(-1)
    plt.plot(BP)
    plt.show()
    # BP_path = '/edrive1/zechenzh/preprocessed_DC/train_BP_systolic.npy'
    # frame_path = '/edrive1/zechenzh/preprocessed_DC/train_frames_face_large.npy'
    # BP = np.load(BP_path)
    # frame = np.load(frame_path)
    # print(f'BP len:{len(BP)}, frame len:{len(frame)}')


if __name__ == '__main__':
    # make_video()
    plot_BP()
