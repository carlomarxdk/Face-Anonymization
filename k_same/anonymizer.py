import numpy as np
from math import sqrt
from sklearn.decomposition import IncrementalPCA
from matplotlib import pyplot as plt

PCA_N = 25


class Anonymizer:
    def __init__(self):
        self.M = None
        self.mean = None
        self.std = None
        self.A = None
        self.pca = None
        self.facespace = None
        self.normalized_gallery = None

    def setup(self, gallery, training, shape):
        # assumed that each image in gallery and training is transformed to 1D np array
        # each face is a row
        self.M = training.shape[0]

        self.mean = self.average(training)
        self.std = np.std(training, axis=0)

        self.A = np.zeros(shape=(self.M, training.shape[1]))

        self.normalized_training = self.normalize(training)
        self.normalized_gallery = self.normalize(gallery)


        for indx in range(0, self.M):
            self.A[indx, :] = (training[indx, :] - self.mean)

        self.pca = IncrementalPCA(n_components=PCA_N, whiten=True)
        self.pca.fit(self.A)
        # self.img_pca = self.pca.transform(self.img_average.reshape((-1,1)))
        # print(pca.components_.shape)
        # plt.imshow(self.pca.components_[0].reshape(shape),  cmap='gray')
        # plt.show()
        # plt.plot(self.pca.explained_variance_ratio_)
        # plt.show()
        self.facespace = self.pca.transform(self.A)

    def recognize(self, img):
        img = (img - self.mean) / self.std
        img_ = np.dot(self.pca.components_, img)
        match = np.zeros(shape=(self.M, 2))
        for indx in range(0, self.facespace.shape[0]):
            match[indx, 0] = indx
            match[indx, 1] = np.linalg.norm(img_ - self.facespace[indx, :])
        return match[np.argsort(match[:, -1])]

    def k_same_pixel(self, H, k, shape):
        self.setup(H, H, shape)
        average_ = np.dot(self.pca.components_, self.mean / self.std)

        for indx in range(100, 110):
            match = self.recognize(H[indx, :])
            if H.shape[0] < 2 * k: k = H.shape[0]
            average = np.zeros(shape=(k,H.shape[1]))
            for i in range(0, k):
                average[i, :] = H[int(match[i][0]), :]
                # print(int(match[i][0]))
            average = self.average(average)
            result = average
            plt.imshow(H[indx, :].reshape(shape), cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.imshow(result.reshape(shape), cmap='gray')
            plt.show()

        plt.imshow(self.mean.reshape(shape), cmap='gray', vmin=0, vmax=255)
        plt.show()

    def k_same_eigen(self, H, k, shape):
        self.setup(H, H, shape)
        average_ = np.dot(self.pca.components_, self.mean)

        for indx in range(100, 110):
            match = self.recognize(H[indx, :])
            if H.shape[0] < 2 * k: k = H.shape[0]
            average = np.zeros(shape=(k, PCA_N))
            for i in range(0, k):
                average[i, :] = self.facespace[int(match[i][0]), :]
            average = self.average(average)
            #average = np.add(average_, average)
            result = self.pca.inverse_transform(average)
            result = np.add(self.mean, result)
            plt.imshow(H[indx, :].reshape(shape), cmap='gray', vmin=0, vmax=255)
            plt.show()
            plt.imshow(result.reshape(shape), cmap='gray')
            plt.show()

        plt.imshow(self.mean.reshape(shape), cmap='gray', vmin=0, vmax=255)
        plt.show()

    @staticmethod
    def average(image_set):
        return np.mean(image_set, axis=0)

    def normalize(self, image_set):
        normalized = np.zeros(shape=image_set.shape)
        for indx in range(0, self.M):
            img = image_set[indx, :]
            normalized[indx, :] = (img - img.min())/ (img.max()- img.min())
        return normalized