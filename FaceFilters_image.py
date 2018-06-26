import cv2
import dlib
import time
import numpy as np

# Definição dos pontos referentes a cada parte do rosto
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_INNER_POINTS = list(range(61, 68))  

LEFT_EYE_POINT = 36
RIGHT_EYE_POINT = 45
CENTER_MOUTH_POINT = 62

# Carregando classificador de faces e landmarks
face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

# Leitura da imagem
ori_img = cv2.imread('images/neymar.jpg')

# Transformando a imagem para escala de cinza
gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

# Detecção das faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Para cada face, coloca um retângulo em volta e identifica olhos, nariz e boca
for (x, y, w, h) in faces:
    cv2.rectangle(ori_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(ori_img, dlib_rect).parts()])
    landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_INNER_POINTS]

    # Reta que liga os 2 olhos
    point_left = (landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = (landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    cv2.line(ori_img, point_left, point_right, color=(0, 0, 255), thickness=2)

    for idx, point in enumerate(landmarks_display):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(ori_img, pos, 2, color=(0, 255, 255), thickness=-1)


# Mostrando a imagem
cv2.imshow('image', ori_img)
cv2.waitKey(0)
