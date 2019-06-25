import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
CASC_PATH = "haarcascades/haarcascade_frontalface_default.xml"
MASK_PATH = 'mask/'

def load(path:'str', shape, mask_=True):
    instances = []
    # Load in the images
    mask = cv2.imread(MASK_PATH+'mask_blur.png').astype(np.uint8)
    mask = cv2.resize(mask, dsize=(shape[1],shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


    for filepath in os.listdir(path):
        img = cv2.imread(path + '/{0}'.format(filepath))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(shape[1],shape[0]),
                         interpolation=cv2.INTER_CUBIC)
        if mask_:
            img = cv2.bitwise_and(img,img,mask = mask)
        img = np.asarray(img, dtype=np.float16)
        img = img.reshape((1, -1)).astype(np.float32)
        instances.append(img)
    output = np.zeros(shape = (len(instances), shape[0]*shape[1]))
    for i, image in enumerate(instances):
        output[i, ] = image

    output = np.nan_to_num(output)
    return output

def extract_face(database, shape=(959,720)):
    entries = database.shape[0]
    #faces = np.zeros((entries, shape[0]*shape[1]))
    faces = []
    for indx in range(0, entries):
        img = database[indx, ].reshape(shape)
        face_info = detect_face(img)
        for (x,y,h,w) in face_info:
            faces.append(img[y,y+w])





# Create the haar cascade
faceCascade = cv2.CascadeClassifier(CASC_PATH)
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=10,
                                        minSize=(40, 40)
                                       )