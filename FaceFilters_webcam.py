import cv2
import dlib
import time
import numpy as np

# Definição dos pontos referentes a cada parte do rosto
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_INNER_POINTS = list(range(61, 68))  

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
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
		landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
		landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_INNER_POINTS]

		for idx, point in enumerate(landmarks_display):
			pos = (point[0, 0], point[0, 1])
			cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

	# Mostrando frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Parando a captura e fechando janelas
cap.release()
cv2.destroyAllWindows()
