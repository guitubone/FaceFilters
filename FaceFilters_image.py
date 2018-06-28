import FaceFilters as FF
import cv2
import dlib
import time
import numpy as np
from math import asin, pi

LEFT_ARROW = [65361, 63234]
UP_ARROW = [65362, 63232]
DOWN_ARROW = [65364, 63233]
RIGHT_ARROW = [65363, 63235]
NUM_FILTERS = 7

# Carregando classificador de faces e landmarks
face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

id = 0
while(True):
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

		if id == -1:
			FF.put_debug(ori_img, landmarks, x, y, w, h)
		elif id == -2:
			ori_img = FF.put_blur(ori_img, landmarks, x, y, w, h)
		elif id == 1:
			FF.Mustache.put(ori_img, landmarks, w, h, x, y)
		elif id == 2:
			FF.FlowerCrown.put(ori_img, landmarks, w, h, x, y)
		elif id == 3:
			FF.DogNose.put(ori_img, landmarks, w, h, x, y)
			if(FF.mouth_open(landmarks, w, h)):
				FF.DogTongue.put(ori_img, landmarks, w, h, x, y)
			FF.DogLeftEar.put(ori_img, landmarks, w, h, x, y)
			FF.DogRightEar.put(ori_img, landmarks, w, h, x, y)
		elif id == 4:
			FF.Glasses.put(ori_img, landmarks, w, h, x, y)
		elif id == 5:
			FF.Glasses.put(ori_img, landmarks, w, h, x, y)
			FF.Mustache.put(ori_img, landmarks, w, h, x, y)
		elif id == 6:
			FF.Glasses.put(ori_img, landmarks, w, h, x, y)
			if(FF.mouth_open(landmarks, w, h)):
				FF.DogTongue.put(ori_img, landmarks, w, h, x, y)
			FF.Mustache.put(ori_img, landmarks, w, h, x, y)

	# Mostrando a imagem
	cv2.imshow('image', ori_img)

	key = cv2.waitKeyEx(0)
	if key == ord('q') or key == ord('Q'):
		break
	elif key in LEFT_ARROW:
		if id < 0:
			id = 0
		id = (id-1)%NUM_FILTERS
	elif key in RIGHT_ARROW:
		if id < 0:
			id = 0
		id = (id+1)%NUM_FILTERS
	elif key in UP_ARROW:
		id = -2
	elif key in DOWN_ARROW:
		id = -1

cv2.destroyAllWindows()
