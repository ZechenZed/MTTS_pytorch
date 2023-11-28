import cv2
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
# from evaluation.ICC_C_1 import

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


def plot_BP(dim=72):
    path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_BP_systolic.npy'
    BP = np.load(path).astype(np.float32)
    BP_flat = BP.reshape(-1)
    var = np.var(BP_flat)
    std = np.std(BP.flat)
    print(f'var:{var}, std:{std}')

    x_axis = np.arange(BP.shape[1]) / 25
    i = 0
    for temp in BP[0:200:5]:
        i += 1
        plt.plot(x_axis, temp, label=f'Chunk:{i}')
    plt.legend()
    plt.show()
    # BP_path = '/edrive1/zechenzh/preprocessed_DC/train_BP_systolic.npy'
    # frame_path = '/edrive1/zechenzh/preprocessed_DC/train_frames_face_large.npy'
    # BP = np.load(BP_path)
    # frame = np.load(frame_path)
    # print(f'BP len:{len(BP)}, frame len:{len(frame)}')

    # video_file_path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/train_frames_face_large.npy'
    # video_file = np.load(video_file_path).reshape((-1, 6, dim, dim))
    #
    # t = []
    # i = 0
    #
    # ############## Reading frame by frame ##############
    # for i in range(video_file.shape[0]):
    #     img = video_file[i, :, :, 0:3]
    #     cv2.imshow('Frame', img)
    #     # Press 'q' to quit
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break


def ICC_matlab():
    eng = matlab.engine.start_matlab()
    print('check')
    # y = eng.feval('ICC_C_1.m')


if __name__ == '__main__':
    # make_video()
    # plot_BP()
    ICC_matlab()
