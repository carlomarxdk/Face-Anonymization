import cv2
import numpy as np


class KnowledgeBase:

    def setup(self, gallery, training: np.array):
        #assumed that each image in gallery and training is transformed to 1D np array
        #each face is a row
        self.M = training.shape[1]
        self.img_average = self.average(training)
        self.A = np.zeros(shape=( training.shape[0], self.M))
        for indx in range(0,self.M):
            self.A[:, indx] = training[indx, :] - self.img_average

        self.C = np.matmul(self.A, np.transpose(self.A))



    @staticmethod
    def average(image_set):
        return np.mean(image_set,axis=0)
