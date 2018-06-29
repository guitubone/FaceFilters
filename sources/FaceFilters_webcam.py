import FaceFilters as FF
import cv2
import dlib
import time
import numpy as np
from math import asin, pi

# setas do teclado
LEFT_ARROW = [65361, 63234]
UP_ARROW = [65362, 63232]
DOWN_ARROW = [65364, 63233]
RIGHT_ARROW = [65363, 63235]
# numero de filtros
NUM_FILTERS = 8

# Carregando classificador de faces e landmarks
face_cascade = cv2.CascadeClassifier('../data/lbpcascade_frontalface.xml')
predictor = dlib.shape_predictor('../data/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

id = 0
while(True):
	# Captura frame-a-frame
    ret, frame = cap.read()

	# Transformando o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecção das faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Para cada face, coloca um retângulo em volta e identifica olhos, nariz e boca
    for (x, y, w, h) in faces:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])

        if id == -1:
            # Filtro de debug
            FF.put_debug(frame, landmarks, x, y, w, h)
        elif id == -2:
            # Filtro de blur
            frame = FF.put_blur(frame, landmarks, x, y, w, h)
        elif id == 1:
            # Mustache
            FF.Mustache.put(frame, landmarks, w, h, x, y)
        elif id == 2:
            # Flower Crown
            FF.FlowerCrown.put(frame, landmarks, w, h, x, y)
        elif id == 3:
            # Dog
            FF.DogNose.put(frame, landmarks, w, h, x, y)
            if(FF.mouth_open(landmarks, w, h)):
                FF.DogTongue.put(frame, landmarks, w, h, x, y)
            FF.DogLeftEar.put(frame, landmarks, w, h, x, y)
            FF.DogRightEar.put(frame, landmarks, w, h, x, y)
        elif id == 4:
            # Glasses
            FF.Glasses.put(frame, landmarks, w, h, x, y)
        elif id == 5:
            # Glasses + Mustache
            FF.Glasses.put(frame, landmarks, w, h, x, y)
            FF.Mustache.put(frame, landmarks, w, h, x, y)
        elif id == 6:
            # Glasses + Mustache + Dog Tongue
            FF.Glasses.put(frame, landmarks, w, h, x, y)
            if(FF.mouth_open(landmarks, w, h)):
                FF.DogTongue.put(frame, landmarks, w, h, x, y)
            FF.Mustache.put(frame, landmarks, w, h, x, y)
        elif id == 7:
            # Pixel Sunglasses
            FF.PixelSunglasses.put(frame, landmarks, w, h, x, y)

    # Mostrando frame
    cv2.imshow('frame',frame)
    key = cv2.waitKeyEx(1)
    if key == ord('q') or key == ord('Q'):
        # Tecla 'Q' encerra
        break
    elif key in LEFT_ARROW:
        # Seta para a esquerda alterna entre filtros
        if id < 0:
            id = 0
        id = (id-1)%NUM_FILTERS
    elif key in RIGHT_ARROW:
        # Seta para a direita alterna entre filtros
        if id < 0:
            id = 0
        id = (id+1)%NUM_FILTERS
    elif key in UP_ARROW:
        # Seta para cima liga o filtro de blur
        id = -2
    elif key in DOWN_ARROW:
        # Seta para cima liga o filtro de debug
        id = -1


# Parando a captura e fechando janelas
cap.release()
cv2.destroyAllWindows()
