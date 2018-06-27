import FaceFilters as FF
import cv2
import dlib
import time
import numpy as np
from math import asin, pi

# Carregando classificador de faces e landmarks
face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# Leitura da imagem
ori_img = cv2.imread('images/group_2.jpg')

# Transformando a imagem para escala de cinza
gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

# Detecção das faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Para cada face, coloca um retângulo em volta e identifica olhos, nariz e boca
for (x, y, w, h) in faces:
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(ori_img, dlib_rect).parts()])

#    FF.put_debug(ori_img, landmarks, x, y, w, h)
#    FF.Mustache.put(ori_img, landmarks, w, h, x, y)
#    FF.FlowerCrown.put(ori_img, landmarks, w, h, x, y)
    FF.DogNose.put(ori_img, landmarks, w, h, x, y)
    FF.DogTongue.put(ori_img, landmarks, w, h, x, y)
    FF.DogLeftEar.put(ori_img, landmarks, w, h, x, y)
    FF.DogRightEar.put(ori_img, landmarks, w, h, x, y)

    ori_img = FF.put_blur(ori_img, landmarks, x, y, w, y)

# Mostrando a imagem
cv2.imshow('image', ori_img)
cv2.waitKey(0)
