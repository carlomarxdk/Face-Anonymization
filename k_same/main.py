import numpy as np
import cv2
from matplotlib import pyplot as plt
from k_same.load import *


IMAGE_PATH = 'images/'
shape = (959, 720) #for FERET database
gallery = load(IMAGE_PATH, shape=shape)
av = np.mean(gallery, axis=0)
plt.imsave('test.png', av.reshape(shape), cmap='gray', vmin=0, vmax=255)