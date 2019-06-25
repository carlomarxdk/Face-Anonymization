import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
from simple_ad_hoc import detect

CASC_PATH = "haarcascades/haarcascade_frontalface_default.xml"
IMAGE_PATH = 'results/grey/'
IMG_NUM = 147
faceCascade = cv2.CascadeClassifier(CASC_PATH)

def anonymise(method:'str', k):

    for i in range(0, IMG_NUM):
        image = cv2.imread(IMAGE_PATH + str(i) + '.png')
        faces = detect_face(image)
        for (x, y, w, h) in faces:
            if method == 'blur':
                image = blur(image,x,y,w,h, k=k)
            elif method == 'pixelate':
                image = pixelate(image,x,y,w,h, scale=101-k)
            # image_d = mask(image,x,y,w,h)
            elif method == 'noise':
                image = noise(image, x, y, w, h, threshold=1 - k/100)

        path = 'results' + '/' + method + '/' + str(k)
        try:
            os.makedirs(path)
        except:
            pass

        path = 'results' + '/' + method + '/' + str(k) + '/' + str(i) + '.png'
        cv2.imwrite(path, image)


def blur(img, x, y, w, h, k=211):
    startY = y
    endY = y + h
    startX = x
    endX = x + w
    img[startY:endY, startX:endX] = cv2.blur(img[startY:endY, startX:endX],
                                             (k, k))
    return img


def pixelate(img, x, y, w, h, scale=16):
    startY = y
    endY = y + h
    startX = x
    endX = x + w
    # Resize input to "pixelated" size
    temp = cv2.resize(img[startY:endY, startX:endX], (scale, scale),
                      interpolation=cv2.INTER_LINEAR)
    # Initialize output image
    img[startY:endY, startX:endX] = cv2.resize(temp, (h, w), interpolation=cv2.INTER_NEAREST)
    return img


def mask(img, x, y, w, h):
    startY = y
    endY = y + h
    startX = x
    endX = x + w
    img[startY:endY, startX:endX] = 0
    return img


def noise(img, x, y, w, h, threshold=0.6):
    startY = y
    endY = y + h
    startX = x
    endX = x + w
    random_map = np.random.random((h, w, 1)).astype(np.float16)
    random_mask = np.random.randint(low=0, high=255, size=(h, w, 3))
    temp = np.asarray(img[startY:endY, startX:endX], dtype="int32")
    temp = np.where(random_map < threshold, temp, random_mask)
    img[startY:endY, startX:endX] = temp
    return img


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=10,
                                        minSize=(40, 40)
                                        )
