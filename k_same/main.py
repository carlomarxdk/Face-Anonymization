import numpy as np
import cv2
from matplotlib import pyplot as plt
from k_same.load import *
from k_same.anonymizer import *


def main():
    IMAGE_PATH = 'images/'
    shape = (240, 180)  # for FERET database
    gallery = load(IMAGE_PATH, shape=shape)

    A = Anonymizer()
    # KB.setup(gallery,gallery,shape)
    # plt.imshow(gallery[1,: ].reshape(shape), cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.imsave('test.png', gallery[1].reshape(shape), cmap='gray', vmin=0, vmax=255)
    # KB.recognize(gallery[130,:])
    A.k_same_eigen(gallery, 50, shape)


if __name__ == '__main__':
    main()
