import cv2
import dlib
import time
import numpy as np
from math import asin, pi
from utils.geometry import Point, dot

class Filter:
    def __init__(self, file_path, get_center, w_ratio, h_ratio):
        self.file_path = file_path
        self.get_center = get_center
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio

    def put(self, img, landmarks, w, h, x, y):
        mask_img = cv2.imread(self.file_path, -1)

        ang = get_ang(landmarks)

        rows, cols, _ = mask_img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), -ang, 1)
        mask_img = cv2.warpAffine(mask_img, M, (cols, rows))

        ori_mask = mask_img[:,:,3]
        ori_mask_inv = cv2.bitwise_not(ori_mask)
        mask_img = mask_img[:,:,0:3]

        center = self.get_center(landmarks, x, y)

        x1 = int(center.x-(self.w_ratio*w/2.0))
        x2 = int(center.x+(self.w_ratio*w/2.0))

        y1 = int(center.y-self.h_ratio*h/2.0)
        y2 = int(center.y+self.h_ratio*h/2.0)

        x1 = fix(x1, img.shape[1])
        x2 = fix(x2, img.shape[1])
        y1 = fix(y1, img.shape[0])
        y2 = fix(y2, img.shape[0])

        mask_overlay = cv2.resize(mask_img, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(ori_mask, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(ori_mask_inv, (x2-x1, y2-y1), interpolation = cv2.INTER_AREA)

        roi = img[y1:y2, x1:x2]

        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        roi_fg = cv2.bitwise_and(mask_overlay, mask_overlay, mask = mask)
        dst = cv2.add(roi_bg,roi_fg)
        img[y1:y2, x1:x2] = dst

def get_ang(landmarks):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])

    line = point_right - point_left

    return asin(line.y / line.x) * 180.0 / pi

def draw_line(img, a, b):
    cv2.line(img, a.tuple(), b.tuple(), color=(0, 0, 255), thickness=2)

def put_debug(img, landmarks, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    landmarks_display = landmarks[RIGHT_EYE_POINTS + LEFT_EYE_POINTS + NOSE_POINTS + MOUTH_INNER_POINTS]

    # Reta que liga os 2 olhos
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    point_mouth = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    point_inter = point_mouth.intersect_line(point_left, point_right)
    draw_line(img, point_left, point_right)
    draw_line(img, point_mouth, point_inter)

    for idx, point in enumerate(landmarks_display):
        pos = (point[0, 0], point[0, 1])
        cv2.circle(img, pos, 2, color=(0, 255, 255), thickness=-1)

def fix(p, lim):
    if p < 0:
        return 0
    if p >= lim:
        return lim-1
    return p

def put_blur(img, points, x, y, w, h):
	x1, y1, x2, y2 = x, y, x+w, y+h

	mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
	hull = cv2.convexHull(points)
	cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))

	aux_img = np.copy(img)
	aux_img = cv2.blur(aux_img, (25, 25))

	mask_inv = cv2.bitwise_not(mask)
	bg = cv2.bitwise_and(img, img, mask = mask_inv)
	fg = cv2.bitwise_and(aux_img, aux_img, mask = mask)
	img = cv2.add(bg, fg)

	return img

# Definição dos pontos referentes a cada parte do rosto
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_INNER_POINTS = list(range(61, 68))  

LEFT_EYE_POINT = 36
RIGHT_EYE_POINT = 45
CENTER_MOUTH_POINT = 63
UNDER_NOSE_POINT = 32
NOSE_POINT = 30
TOP_TONGUE_POINT = 62
BOT_TONGUE_POINT = 66
LEFT_EAR_POINT = 17
RIGHT_EAR_POINT = 26

def get_center_mustache(landmarks, x, y):
    mouth_point = Point(landmarks[CENTER_MOUTH_POINT, 0], landmarks[CENTER_MOUTH_POINT, 1])
    nose_point = Point(landmarks[UNDER_NOSE_POINT, 0], landmarks[UNDER_NOSE_POINT, 1])
    center = mouth_point + ((nose_point - mouth_point)*(1.0/3.0))   # Altura do bigode
    return center
Mustache = Filter('filters/mustache.png', get_center_mustache, 0.7, 0.2)

def get_center_flower_crown(landmarks, x, y):
    point_left = Point(landmarks[LEFT_EYE_POINT, 0], landmarks[LEFT_EYE_POINT, 1])
    point_right = Point(landmarks[RIGHT_EYE_POINT, 0], landmarks[RIGHT_EYE_POINT, 1])
    center = point_left + ((point_right - point_left)*(1.0/2.0))   # Altura do bigode
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
