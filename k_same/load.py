import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def load(path:'str', shape=(959,720)):
    instances = []
    (h,w) = (0,0)
    # Load in the images
    for filepath in os.listdir(path):
        img = cv2.imread(path + '/{0}'.format(filepath), 0)


        img = np.asarray(img, dtype="int32")
        img = img[0:shape[0], 0:shape[1]]
        img = np.array(img)
        img = img.reshape((1,-1))
        instances.append(img)

    output = np.zeros(shape = (len(instances), shape[0]*shape[1]))
    for i, image in enumerate(instances):
        output[i, ] = image


    return output