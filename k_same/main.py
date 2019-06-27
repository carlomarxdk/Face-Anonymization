import numpy as np
import cv2
from matplotlib import pyplot as plt
from k_same.load import *
from k_same.anonymizer import *

IMAGE_PATH = 'images/'

def main():
    shape = (240, 180)  # for FERET database
    gallery = load(IMAGE_PATH, shape=shape, mask_=False)

    for indx in range(0, gallery.shape[0]):
        img = gallery[indx, :].reshape(shape)
        path = 'results/grey/' + str(indx) + '.png'
        print(path)
        cv2.imwrite(path, img)

    A = Anonymizer()
    # KB.setup(gallery,gallery,shape)
    #plt.imshow(gallery[1,: ].reshape(shape), cmap='gray', vmin=0, vmax=255)
    #plt.show()
    # plt.imsave('test.png', gallery[1].reshape(shape), cmap='gray', vmin=0, vmax=255)
    # KB.recognize(gallery[130,:])
    #A.k_same_looseM(gallery, 10, shape)

    #A.process('pixel', gallery, 100, shape) ## pixel eigen same
    #A.process('eigen', gallery, 2, shape)
    for i in range(1,101):
       A.process('eigen', gallery, i, shape)
    for i in range(1,101):
        A.process('same', gallery, i, shape)


if __name__ == '__main__':
    main()
