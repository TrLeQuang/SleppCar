import imutils
import numpy as np
import requests
import time
import dlib
import cv2, os, sys
import collections
import random
import face_recognition
import pickle
import math
import threading
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils


################## PHẦN ĐỊNH NGHĨA CLASS, FUNCTION #######################


# Class định nghĩa vị trí 2 mắt con người
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Hàm dự đoán mắt đóng hay mở
def predict_eye_state(model, image):
    # Resize ảnh về 20x10
    image = cv2.resize(image, (20, 10))
    image = image.astype(dtype=np.float32)

    # Chuyển thành tensor
    image_batch = np.reshape(image, (1, 10, 20, 1))
    # Đưa vào mạng mobilenet để xem mắt đóng hay mở
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)

    return np.argmax(model.predict(image_batch)[0])


################ CHƯƠNG TRÌNH CHÍNH ##############################

# Load model dlib để phát hiện các điểm trên mặt người - lansmark
facial_landmarks_predictor = '68_face_landmarks_predictor.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

# Load model predict xem mắt người đang đóng hay mở
model = load_model('weights.149-0.01.hdf5')

# Lấy ảnh từ Webcam
cap = cv2.VideoCapture(0)
scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 5


while (True):
    c = time.time()
    # Đọc ảnh từ webcam và chuyển thành RGB
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize ảnh còn 50% kích thuớc góc
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Chuyển sang hệ màu LAB để lấy thành lan Lightness
    lab = cv2.cvtColor(resized_image, cv2.COLOR_RGB2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

    # Tìm kiếm khuôn mặt bằng HOG
    face_locations = face_recognition.face_locations(l, model='hog')

    # Nếu tìm thấy ít nhất 1 khuôn mặt
    if len(face_locations):

        # Lấy vị trí khuôn mặt
        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)

        # Trích xuất vị trí 2 mắt

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([left_eye_indices]))
        left_eye = gray[y:y + h, x:x + w]

        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]

        (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
        right_eye = gray[y:y + h, x:x + w]

        # Dùng mobilenet để xem từng mắt là MỞ hay ĐÓNG

        left_eye_open = 'yes' if predict_eye_state(model=model, image=left_eye) else 'no'
        right_eye_open = 'yes' if predict_eye_state(model=model, image=right_eye) else 'no'

        print('left eye open: {0}    right eye open: {1}'.format(left_eye_open, right_eye_open))

        # Nếu 2 mắt mở thì hiển thị màu xanh còn không thì màu đỏ
        if left_eye_open == 'yes' and right_eye_open == 'yes':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            currState = 0
            countClose = 0

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            currState = 1
            countClose +=1


    frame = cv2.flip(frame, 1)
    if countClose > alarmThreshold:
        cv2.putText(frame, "Sleep detected! Alarm!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    cv2.imshow('Sleep Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()