import cv2
import dlib
import time
import numpy as np
from math import asin, pi
from utils.geometry import Point, dot

class Filter:
    """Classe Filter
        file_path: file path do arquivo de imagem do filtro
        get_center: funcao que calcula o centro da imagem do filtro
        w_ratio: porcentagem da largura da imagem em relacao a face
        h_ratio: porcentagem da altura da imagem em relacao a face
    """
    def __init__(self, file_path, get_center, w_ratio, h_ratio):
        """ construtor
        """
        self.file_path = file_path
        self.get_center = get_center
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio

    def put(self, img, landmarks, w, h, x, y):
        """ coloca o filtro self na imagem img sobre a face reference a ladmarks
                img: imagem sobre a qual sera colocada o filtro
                landmarks: pontos da face
                w: largura da face
                h: altura da face
                x: x do ponto mais a esquerda e a cima do retangula da face
                y: y do ponto mais a esquerda e a cima do retangula da face
        """
        # carrega imagem do filtro
        mask_img = cv2.imread(self.file_path, -1)

        # calcula angulo de rotacao da face
        ang = get_ang(landmarks)

        # rotaciona a imagem do filtro
        rows, cols, _ = mask_img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -ang, 1)
        mask_img = cv2.warpAffine(mask_img, M, (cols, rows))

        # recupera o ponto de centro da imagem
        center = self.get_center(landmarks, x, y)

        # (x1, y1) ponto mais a esquerda em cima da imagem do fitro
        # (x2, y2) ponto mais a direita em baixo da imagem do fitro
        x1 = int(center.x-(self.w_ratio*w/2.0))
        x2 = int(center.x+(self.w_ratio*w/2.0))

        y1 = int(center.y-self.h_ratio*h/2.0)
        y2 = int(center.y+self.h_ratio*h/2.0)

        x1 = fix(x1, img.shape[1])
        x2 = fix(x2, img.shape[1])
        y1 = fix(y1, img.shape[0])
        y2 = fix(y2, img.shape[0])

        if(x1 == x2 or y1 == y2):
            return

        # coloca o filtro na imagem
        ori_mask = mask_img[:,:,3]
        ori_mask_inv = cv2.bitwise_not(ori_mask)
        mask_img = mask_img[:,:,0:3]

        mask_overlay = cv2.resize(mask_img, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(ori_mask, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(ori_mask_inv, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)

        roi = img[y1:y2, x1:x2]

        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        roi_fg = cv2.bitwise_and(mask_overlay, mask_overlay, mask = mask)
        dst = cv2.add(roi_bg,roi_fg)
        img[y1:y2, x1:x2] = dst

def get_ang(landmarks):
    """ Calcula o angulo de rotacao da face
        landmarks: Pontos da Face
        retorno(float): angulo em radianos
    """
    # Ponto a extrema esquerda do olho esquerdo
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    # Ponto a extrama direita do olho direito
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])

    # vetor da linha que liga point_left ao point_right
    line = point_right - point_left

    # retorna o angulo de inclinacao dessa linha em relacao ao eixo horizontal
    return asin(line.y / line.x) * 180.0 / pi

def draw_line(img, a, b):
    """ Desenha linha do ponto a ao ponto b
        img: imagem em que sera desenhada a linha
        a: ponto a
        b: ponto b
    """
    cv2.line(img, a.tuple(), b.tuple(), color=(0, 0, 255), thickness=2)

def put_debug(img, landmarks, x, y, w, h):
    """ Filtro de debug, imprime alguns pontos de landmarks, retangulo da face e retas de inclinacao da face
        img: imagem em que sera desenhado o filtro
        landmarks: array de pontos da face
        (x, y): ponto mais a esquerda em cima do retangulo da face
        w: largura do retanculo da face
        h: altura do retangulo da face
    """
    # desenha retangulo da face
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # pontos da face a serem mostrados (olho direito, olho esquerdo, naris e boca)
    landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_INNER_POINTS]

    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    # calcula reta perpendicular a reta que liga os 2 olhos
    point_inter = point_mouth.intersect_line(point_left, point_right)
    # desenha reta que liga os 2 olhos
    draw_line(img, point_left, point_right)
    # desenha reta perpendicular a reta que liga os 2 olhos
    draw_line(img, point_mouth, point_inter)

    # imprime os pontos da face
    for idx, point in enumerate(landmarks_display):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img, pos, 2, color=(0, 255, 255), thickness=-1)

def fix(p, lim):
    """ Corrige valores fora do limite [0, lim)
        p: valor
        lim: limite
    """
    if p < 0:
        return 0
    if p >= lim:
        return lim-1
    return p

def put_blur(img, points, x, y, w, h):
    """ Filtro para borrar o rosto
        img: imagem em que sera desenhado o filtro
        points: array de pontos da face
        (x, y): ponto mais a esquerda em cima do retangulo da face
        w: largura do retanculo da face
        h: altura do retangulo da face
        retorno: imagem com a face borrada
    """
    # pontos do retangulo
    x1, y1, x2, y2 = x, y, x+w, y+h

    # ponto mais a esquerda e mais a direta das sombrancelhas
    EB_beg, EB_end = EYEBROWS_POINTS[0], EYEBROWS_POINTS[-1]
    # maior ponto y das sombrancelhas
    highest_point = np.min(points[EYEBROWS_POINTS, 1])
    # aumenta o y dos pontos das sombrancelhas, normalizando-os com o y e highest_point
    points[EB_beg:EB_end, 1] = y + (points[EB_beg:EB_end, 1] - highest_point)

    # calcula o feixo convexo dos pontos da face, e calcula uma mascara com eles
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))

    # realiza o blur nos pontos do convex hull calculado
    aux_img = np.copy(img)
    aux_img = cv2.blur(aux_img, (25, 25))

    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(img, img, mask = mask_inv)
    fg = cv2.bitwise_and(aux_img, aux_img, mask = mask)
    img = cv2.add(bg, fg)

    return img

def mouth_open(landmarks, w, h):
    """ Verifica se uma face contem uma boca aberta
        landmarks: array de pontos da face
        w: largura do retangulo da face
        h: altura do retangulo da face
        returno: True se a boca esta aberta, False caso contrario
    """
    # Ponto dos labios superiores
    point_top = Point(landmarks[TOP_TONGUE_POINT, 0], landmarks[TOP_TONGUE_POINT, 1])
    # Ponto dos labios inferiores
    point_bot = Point(landmarks[BOT_TONGUE_POINT, 0], landmarks[BOT_TONGUE_POINT, 1])
    # se a distancia entre os pontos for maior que o 4 por cento do maximo entre a altura e largura da face considera boca aberta
    return ((point_top - point_bot).len() > max(w, h) * 0.04)

# Definição dos pontos referentes a cada parte do rosto
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_INNER_POINTS = list(range(61, 68))  
EYEBROWS_POINTS = list(range(17, 28))  

LEFT_EYE_POINT = 36
RIGHT_EYE_POINT = 45
CENTER_MOUTH_POINT = 63
TOP_NOSE_POINT = 28
UNDER_NOSE_POINT = 32
NOSE_POINT = 30
TOP_TONGUE_POINT = 62
BOT_TONGUE_POINT = 66
LEFT_EAR_POINT = 17
RIGHT_EAR_POINT = 26

# Inicializacao dos Filtros, todos os get_center sao definidos de acordo com o filtro

def get_center_mustache(landmarks, x, y):
    mouth_point = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    nose_point = Point(landmarks[UNDER_NOSE_POINT, 0], landmarks[UNDER_NOSE_POINT, 1])
    center = mouth_point + ((nose_point - mouth_point)*(1.0/3.0))   # Altura do bigode
    return center
Mustache = Filter('filters/mustache.png', get_center_mustache, 0.7, 0.2)

def get_center_flower_crown(landmarks, x, y):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    center = point_left + ((point_right - point_left)*(1.0/2.0))
    return Point(center.x, y)
FlowerCrown = Filter('filters/flower_crown.png', get_center_flower_crown, 1.2, 0.6)

def get_center_dog_nose(landmarks, x, y):
    return Point(landmarks[NOSE_POINT, 0], landmarks[NOSE_POINT, 1])
DogNose = Filter('filters/dog_nose.png', get_center_dog_nose, 0.4, 0.3)

def get_center_dog_tongue(landmarks, x, y):
    point_top = Point(landmarks[TOP_TONGUE_POINT, 0], landmarks[TOP_TONGUE_POINT, 1])
    point_bot = Point(landmarks[BOT_TONGUE_POINT, 0], landmarks[BOT_TONGUE_POINT, 1])
    return point_bot + (point_top - point_bot)*0.5
DogTongue = Filter('filters/dog_tongue_2.png', get_center_dog_tongue, 0.4, 0.6)

def get_center_dog_left_ear(landmarks, x, y):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    point_inter = point_mouth.intersect_line(point_left, point_right)
    p = Point(landmarks[LEFT_EAR_POINT,0], landmarks[LEFT_EAR_POINT, 1])
    return p + (point_inter - point_mouth)*1.0
DogLeftEar = Filter('filters/dog_left_ear.png', get_center_dog_left_ear, 0.4, 0.3)

def get_center_dog_right_ear(landmarks, x, y):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    point_inter = point_mouth.intersect_line(point_left, point_right)
    p = Point(landmarks[RIGHT_EAR_POINT,0], landmarks[RIGHT_EAR_POINT, 1])
    return p + (point_inter - point_mouth)*1.0
DogRightEar = Filter('filters/dog_right_ear.png', get_center_dog_right_ear, 0.4, 0.3)

def get_center_glasses(landmarks, x, y):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    point_inter = point_mouth.intersect_line(point_left, point_right)
    p = Point(landmarks[TOP_NOSE_POINT, 0], landmarks[TOP_NOSE_POINT, 1])
    return p + (point_inter - point_mouth)*0.05
Glasses = Filter('filters/glasses_2.png', get_center_glasses, 1.0, 0.6)

def get_center_pixel_sunglasses(landmarks, x, y):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    point_inter = point_mouth.intersect_line(point_left, point_right)
    p = Point(landmarks[TOP_NOSE_POINT, 0], landmarks[TOP_NOSE_POINT, 1])
    return p + (point_inter - point_mouth)*0.05
PixelSunglasses = Filter('filters/pixel_sunglasses_2.png', get_center_glasses, 0.9, 0.6)
