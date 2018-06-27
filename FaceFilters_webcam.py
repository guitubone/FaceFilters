import cv2
import dlib
import time
import numpy as np

# Classe para representar um ponto/vetor, facilitar codigo para as funcoes de operacoes geometricas
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return str(self.x) + " " + str(self.y)

    # Soma de dois pointos
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    # Subtracao de dois pointos
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    # Multiplicacao de ponto por escalar
    def __mul__(self, t):
        return Point(self.x * t, self.y * t)

    # Tamanho do vetor (0,0)->(self.x,self.y)
    def len(self):
        return np.sqrt(self.x*self.x + self.y*self.y)

    def tuple(self):
        return int(self.x), int(self.y)

    # Projecao do vetor self em other
    def relative_proj(self, other):
        return dot(self,other)/(other.len() * other.len())

    # Retorna o ponto da linha que liga self a reta representada que passa por a e b
    def intersect_line(self, a, b):
        p = self
        if(a == b):
            return a
        ap = p - a
        ab = b - a
        u = ap.relative_proj(ab)
        return a + ab*u

# Produto escalar entre pontos a e b
def dot(a, b):
    return a.x*b.x + a.y*b.y

def draw_line(img, a, b):
    cv2.line(img, a.tuple(), b.tuple(), color=(0, 0, 255), thickness=2)

def put_mask(img, mid_eye, dist, mask_img):
	ori_mask = mask_img[:,:,3]
	ori_mask_inv = cv2.bitwise_not(ori_mask)
	mask_img = mask_img[:,:,0:3]

	x1 = int(mid_eye.x-(dist/2))
	x2 = int(mid_eye.x+(dist/2))
	y1 = int(mid_eye.y-100)
	y2 = int(mid_eye.y+100)

	if x1 < 0:
		x1 = 0
	if y1 < 0:
		y1 = 0
	if x2 > img.shape[0]:
		x2 = img.shape[0]
	if y2 > img.shape[1]:
		y2 = img.shape[1]

	mask_overlay = cv2.resize(mask_img, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)
	mask = cv2.resize(ori_mask, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)
	mask_inv = cv2.resize(ori_mask_inv, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)

	roi = img[y1:y2, x1:x2]

	roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
	roi_fg = cv2.bitwise_and(mask_overlay, mask_overlay, mask = mask)
	dst = cv2.add(roi_bg,roi_fg)
	img[y1:y2, x1:x2] = dst

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

cap = cv2.VideoCapture(0)

while(True):
	# Captura frame-a-frame
    ret, frame = cap.read()

	# Transformando o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecção das faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    mask_img = cv2.imread('filters/eye_mask.png', -1)

        # Para cada face, coloca um retângulo em volta e identifica olhos, nariz e boca
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_INNER_POINTS]

        point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
        point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
        point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
        point_inter = point_mouth.intersect_line(point_left, point_right)
        #draw_line(frame, point_left, point_right)
        #draw_line(frame, point_mouth, point_inter)

        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            #cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

            put_mask(frame, point_inter, np.sqrt((point_left.x-point_right.x)**2 + (point_left.y-point_right.y)**2), mask_img)

    # Mostrando frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Parando a captura e fechando janelas
cap.release()
cv2.destroyAllWindows()
