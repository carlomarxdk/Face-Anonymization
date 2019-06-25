import numpy as np
from math import sqrt
import cv2
import os
from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as plt

PCA_N = 30
SCALER = 0.45
E = 8e-15


class Anonymizer:
    def __init__(self):
        self.M = None
        self.mean = None
        self.normalized_mean = None
        self.std = None
        self.A = None
        self.pca = None
        self.face_space = None
        self.normalized_gallery = None
        self.normalized_training = None
        self.anonimized = None

    def setup(self, gallery, training, shape):
        # assumed that each image in gallery and training is transformed to 1D np array
        # each face is a row

        self.M = training.shape[0]  # number of images
        self.A = np.zeros(shape=(self.M, training.shape[1]))  # difference matrix
        self.anonimized = np.zeros(shape=(self.M, training.shape[1]))
        self.mean = self.average(training)
        self.std = np.std(training, axis=0)
        self.std = np.nan_to_num(self.std) + E
        self.normalized_training = self.normalize(training,
                                                  self.mean,
                                                  self.std)
        self.normalized_gallery = self.normalize(training,
                                                  self.mean,
                                                  self.std)
        self.normalized_mean = self.mean/(self.std)

        for indx in range(0, self.M):
            self.A[indx, :] = (self.normalized_training[indx, :])

        self.pca = IncrementalPCA(n_components=PCA_N, whiten=False)
        self.pca.fit(self.A)
        # self.img_pca = self.pca.transform(self.img_average.reshape((-1,1)))
        # print(pca.components_.shape)
        # plt.imshow(self.pca.components_[0].reshape(shape),  cmap='gray')
        # plt.show()
        # plt.plot(self.pca.explained_variance_ratio_)
        # plt.show()
        self.face_space = self.pca.transform(self.normalized_gallery)  # faces in the PCA space
                                            #It is the same as A
    def recognize(self, img):
        img = self.normalize_img(img, self.mean, self.std)
        img_ = np.dot(self.pca.components_, img)
        match = np.zeros(shape=(self.M, 2))
        for indx in range(0, self.face_space.shape[0]):
            match[indx, 0] = indx
            match[indx, 1] = np.linalg.norm(img_ - self.face_space[indx, :])
        return match[np.argsort(match[:, -1])]

    def k_same_pixel(self, H, k, shape):
        self.setup(H, H, shape)

        for indx in range(0, self.M):
            match = self.recognize(H[indx, :])
            if H.shape[0] < 2 * k: k = H.shape[0]
            average = np.zeros(shape=(k, H.shape[1]))
            for i in range(0, k):
                average[i, :] = self.normalized_training[int(match[i][0]), :]
                # print(int(match[i][0]))
            average = self.average(average) * self.std + self.mean
            result = average
            self.anonimized[indx, :] = result
            #plt.imshow(H[indx, :].reshape(shape), cmap='gray', vmin=0, vmax=255)
            #plt.show()
            #plt.imshow(result.reshape(shape), cmap='gray')
            #plt.show()

        #plt.imshow(self.mean.reshape(shape), cmap='gray', vmin=0, vmax=255)
        #plt.show()
        return self.anonimized

    def k_same_eigen(self, H, k, shape):
        self.setup(H, H, shape)

        for indx in range(0, self.M):
            match = self.recognize(H[indx, :])
            if H.shape[0] < 2 * k: k = H.shape[0]
            average = np.zeros(shape=(k, PCA_N))
            for i in range(0, k):
                average[i, :] = self.face_space[int(match[i][0]), :]
            average = self.average(average)
            result = self.pca.inverse_transform(average) *self.std + self.mean
            #plt.imshow(H[indx, :].reshape(shape), cmap='gray', vmin=0, vmax=255)
            #plt.show()
            #plt.imshow(result.reshape(shape), cmap='gray')
            #plt.show()
            self.anonimized[indx, :] = result
            return self.anonimized

        #plt.imshow(self.mean.reshape(shape), cmap='gray', vmin=0, vmax=255)
        #plt.show()

    def k_same_looseM(self,H,k,shape):
        self.setup(H, H, shape)

        for indx in range(0, self.M):
            match = self.recognize(H[indx, :])
            if H.shape[0] < 2 * k: k = H.shape[0]
            average = np.zeros(shape=(k, PCA_N))
            for i in range(0, k):
                average[i, :] = self.face_space[int(match[i][0]), :]
            boundary = match[-1][1] * SCALER

            upper_i = match.shape[0]
            random_i = 0
            while True:
                random_i = np.random.randint(low=0, high=upper_i)
                if match[random_i][1] < boundary: break
                else: upper_i = random_i

            #print(match[-1][1], boundary, int(match[random_i][0]), indx)
            match = self.recognize(H[int(match[random_i][0])])
            average = np.zeros(shape=(k, PCA_N))
            for i in range(0, k):
                average[i, :] = self.face_space[int(match[i][0]), :]

            average = self.average(average)
            result = self.pca.inverse_transform(average) * self.std + self.mean
            self.anonimized[indx, :] = result
            #plt.imshow(H[indx, :].reshape(shape), cmap='gray', vmin=0, vmax=255)
            #plt.show()
            #plt.imshow(result.reshape(shape), cmap='gray')
            #plt.show()
        return self.anonimized

    def process(self, method:'str', gallery, k, shape):
        if method == 'pixel':
            output = self.k_same_pixel(gallery, k, shape)
        elif method == 'eigen':
            output = self.k_same_pixel(gallery, k, shape)
        elif method == 'same':
            output = self.k_same_pixel(gallery, k, shape)

        print('Saving images', self.anonimized.shape)
        path = 'results'+ '/' + method + '/'+ str(k)
        try:
            os.makedirs(path)
        except:
            pass

        for indx in range(0, self.M):
            img = output[indx,:].reshape(shape)
            path = 'results'+ '/' + method + '/'+ str(k) + '/' + str(indx) + '.png'
            print(path)
            cv2.imwrite(path, img)


    @staticmethod
    def average(image_set):
        return np.mean(image_set, axis=0)

    def normalize(self, image_set, mean, std):
        normalized = np.zeros(shape=image_set.shape)
        for indx in range(0, self.M):
            img = image_set[indx, :]
            normalized[indx, :] = (img - mean) / std
        return normalized

    def normalize_img(self, img, mean,std):
        return (img - mean) /std
