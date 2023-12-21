import cv2
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine


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
    # ##################### BP visualization #####################
    # path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/train_BP_female.npy'
    # BP = np.load(path).astype(np.float32)
    # BP_flat = BP.reshape(-1)
    # var = np.var(BP_flat)
    # std = np.std(BP.flat)
    # print(f'var:{var}, std:{std}')
    #
    # plt.plot(BP_flat)
    # plt.legend()
    # plt.show()

    ##################### Video visualization #####################

    path = 'C:/Users/Zed/Desktop/V4V/preprocessed_v4v/test_frames_female_face_large.npy'
    frames = np.load(path)
    frames = frames.reshape((-1, 6, 72, 72))

    for img in frames:
        transposed_arr = np.transpose(img, (1, 2, 0))
        frames = transposed_arr.reshape((72, 72, 6))

        cv2.imshow('Frame', frames[:, :, -3:])
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def ICC_matlab():
    eng = matlab.engine.start_matlab()
    print('check')
    # y = eng.feval('ICC_C_1.m')


if __name__ == '__main__':
    # make_video()
    plot_BP()
    # ICC_matlab()
