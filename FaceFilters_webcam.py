import FaceFilters as FF
import cv2
import dlib
import time
import numpy as np
from math import asin, pi

# Carregando classificador de faces e landmarks
face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

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

#        FF.put_blur(frame, [int(x), int(y), int(x+w), int(y+h)])

#        FF.put_debug(frame, landmarks, x, y, w, h)
#        FF.Mustache.put(frame, landmarks, w, h, x, y)
#        FF.FlowerCrown.put(frame, landmarks, w, h, x, y#)
        FF.DogNose.put(frame, landmarks, w, h, x, y)
        FF.DogTongue.put(frame, landmarks, w, h, x, y)
        FF.DogLeftEar.put(frame, landmarks, w, h, x, y)
        FF.DogRightEar.put(frame, landmarks, w, h, x, y)

    # Mostrando frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Parando a captura e fechando janelas
cap.release()
cv2.destroyAllWindows()
