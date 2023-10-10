import random
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.util import img_as_float
from scipy.sparse import spdiags
import matplotlib.pyplot as plt


def is_consecutive(l, n):
    """
  Checks if there is a continuous number in a list for n indexes.

  Args:
    list: The list to check.
    n: The number of indexes to check.

  Returns:
    True if there is a continuous number in the list for n indexes, False otherwise.
  """
    if len(l) < 25:
        return False
    for z in range(len(l) - n):
        if l[z] + n - 1 == l[z + n - 1]:
            return True
    return False


def preprocess_raw_video(video_file_path, dim=72, plot=True, face_crop=True):
    # set up
    print("***********Processing " + video_file_path[-12:] + "***********")
    t = []
    i = 0
    invalid_frames = []
    vidObj = cv2.VideoCapture(video_file_path)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    success, img = vidObj.read()
    # height, width = img.shape[:2]

    # # OpenCV
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # OpenCV Cuda
    # txt_path = '/home/zechenzh/MTTS_pytorch/preprocess/res10_300x300_ssd_iter_140000.prototxt'
    # model_path = '/home/zechenzh/MTTS_pytorch/preprocess/res10_300x300_ssd_iter_140000.caffemodel'
    # # txt_path = 'res10_300x300_ssd_iter_140000.prototxt'
    # # model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    # faceModel = cv2.dnn.readNetFromCaffe(txt_path,caffeModel=model_path)
    # faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # # Mediapipe
    first = 0
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)
    prev_roi = img_as_float(img)

    ############## Reading frame by frame ##############
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))
        if len(invalid_frames) / totalFrames > 0.25:
            print('Too many invalid frames')
            break
        if is_consecutive(invalid_frames, 50):
            print('Invalid frames more than 2s')
            break

        # # Opencv Cuda
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 0, 177.0, 123.0), swapRB=False, crop=False)
        # faceModel.setInput(blob)
        # predictions = faceModel.forward()
        # roi = 0
        # for i in range(0, predictions.shape[2]):
        #     if predictions[0, 0, i, 2] > 0.5:
        #         bbox = predictions[0, 0, i, 3:7] * np.array([height, width, height, width])
        #         (xmin, ymin, xmax, ymax) = bbox.astype('int')
        #         # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        #         roi = img_as_float(img[ymin:ymax, xmin:xmax, :])
        # # cv2.imshow('Frame', img)
        # # # Press 'q' to quit
        # # if cv2.waitKey(1) & 0xFF == ord('q'):
        # #     break

        ################### Add edge to the Img ###################
        # width_edge = 100
        # height_edge = height * (width_edge / width)
        # original_cf = np.float32([[0, 0], [width - 1, 0], [(width - 1) / 2, height - 1]])
        # transed_cf = np.float32([[width_edge - 1, height_edge - 1], [width - width_edge - 1, height_edge - 1],
        #                          [(width - 1) / 2, height - height_edge - 1]])
        # matrix = cv2.getAffineTransform(original_cf, transed_cf)
        # img = cv2.warpAffine(img, matrix, (height, width))

        ################## Face detection with OPENCV ###################
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_gpu = cv2.cuda_GpuMat(img)
        # faces = face_cascade.detectMultiScale(img_gpu, 1.1, 4)
        # faces = faces.download()
        # roi = 0
        # if faces.all():
        #     for (x, y, w, h) in faces:
        #         xmax = x+w
        #         ymax = y+h
        #         roi = img_as_float(img[y:y + h, x:x + w, :])
        #         cv2.rectangle(img, (x, y), (xmax, ymax), (0, 255, 0), 2)
        #
        # else:
        #     roi = prev_roi
        #     invalid_frames.append(i)
        #     print(f'Warning No Face Detected in {i}th frame')
        #     ####### Video #######
        #     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # prev_roi = roi
        # ################## Video ###################
        # cv2.imshow('Frame', img)
        # # Press 'q' to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        ################### Face detection with MediaPipe ###################
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        results = face_detection.process(img)
        roi = 0
        if results.detections:
            if first == 0:
                first = i

            for face in results.detections:
                bbox = face.location_data.relative_bounding_box
                xmin = int(bbox.xmin * img.shape[1])
                ymin = int(bbox.ymin * img.shape[0])
                xmax = int(bbox.xmin * img.shape[1] + bbox.width * img.shape[1])
                ymax = int(bbox.ymin * img.shape[0] + 1.2 * bbox.height * img.shape[0])

                xmin = max(xmin, 0)

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                roi = img_as_float(img[ymin:ymax, xmin:xmax, :])

            vidLxL = cv2.resize(roi, (dim, dim), interpolation=cv2.INTER_AREA)
            prev_roi = roi
        else:
            # print(f'No Face Detected in {video_file_path[-12:]} at {i}th Frame')
            invalid_frames.append(i)
            vidLxL = cv2.resize(prev_roi, (dim, dim), interpolation=cv2.INTER_AREA)

        # ################## Video ###################
        # cv2.imshow('Frame', img)
        # # Press 'q' to quit
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL

        success, img = vidObj.read()
        i = i + 1
    Xsub[0:first] = Xsub[first]
    Xsub = Xsub[0:len(t) - 1]

    ########################## Normalize raw frames in the appearance branch ##########################
    normalized_len = len(t) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j + 1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j + 1, :, :, :] + Xsub[j, :, :, :])
    dXsub = dXsub / np.std(dXsub)
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:totalFrames - 1, :, :, :]
    dXsub = np.concatenate((dXsub, Xsub), axis=3)

    ##########################  Video array transpose ##########################
    transposed_arr = np.transpose(dXsub, (0, 3, 1, 2))
    dXsub = transposed_arr.reshape((normalized_len, 6, dim, dim))
    # plt.matshow(dXsub[200, 5, :, :])
    # plt.matshow(dXsub[200, 2, :, :])
    # plt.show()
    return dXsub


def preprocess_raw_video_unsupervised(video_file_path, dim=108, face_crop=True):
    # set up
    print("***********Processing " + video_file_path[-8:] + "***********")
    t = []
    i = 0
    vidObj = cv2.VideoCapture(video_file_path)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    print(fps)
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    height = rows
    width = cols
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))

        # Without considering the ratio
        # vidLxL = cv2.resize(img_as_float(img[:, :, :]), (dim, dim), interpolation=cv2.INTER_AREA)

        if face_crop:
            # Face cropping with black edge around each frame of picture
            width_edge = 300
            height_edge = height * (width_edge / width)
            original_cf = np.float32([[0, 0], [width - 1, 0], [(width - 1) / 2, height - 1]])
            transed_cf = np.float32([[width_edge - 1, height_edge - 1], [width - width_edge - 1, height_edge - 1],
                                     [(width - 1) / 2, height - height_edge - 1]])
            matrix = cv2.getAffineTransform(original_cf, transed_cf)
            img = cv2.warpAffine(img, matrix, (cols, rows))

            # Face detection in gray scale image
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            # print(faces.shape[0])
            if not np.any(faces): print('warning')
            # Cropping out ROI from the original image based on the "1:1:1"ish face ratio
            roi = 0
            for (x, y, w, h) in faces:
                roi = img_as_float(img[int(y + 0.05 * h):int(y + 0.15 * h), int(x + 0.45 * w):int(x + 0.55 * w), :])

            # Original resizing from MTTS_CAN
            vidLxL = cv2.resize(roi, (dim, dim), interpolation=cv2.INTER_AREA)

        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL
        # plt.matshow(vidLxL)
        # plt.show()
        success, img = vidObj.read()
        i = i + 1

        # plt.matshow(vidLxL)
        # plt.title('Sample Preprocessed Frame')
        # plt.show()

    return Xsub, fps


def preprocess_finger(video_file_path, dim=108):
    # set up
    print("***********Processing " + video_file_path[-8:] + "***********")
    t = []
    i = 0
    vidObj = cv2.VideoCapture(video_file_path)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    print(fps)
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    height = rows
    width = cols

    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))

        # Without considering the ratio
        vidLxL = cv2.resize(img_as_float(img[:, :, :]), (dim, dim), interpolation=cv2.INTER_AREA)
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL

        success, img = vidObj.read()
        i = i + 1

        # plt.matshow(vidLxL)
        # plt.title('Sample Preprocessed Frame')
        # plt.show()

    return Xsub


def count_frames(video_file_path):
    # print("***********Processing " + video_file_path[-12:] + "***********")
    t = []
    i = 0
    vidObj = cv2.VideoCapture(video_file_path)
    success, img = vidObj.read()
    rows, cols, _ = img.shape
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))
        success, img = vidObj.read()
        i = i + 1
    normalized_len = len(t) - 1
    return normalized_len


def detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal
