import numpy as np


class KNearestNeighbor(object):
    """ A KNN classifier with L2 distance. """

    def __init_(self):
        pass


    def train(self, X_train, Y_train):
        """"""

        self.X_train = X_train
        self.Y_train = Y_train


    def predict(self, distances, k=1):
        """"""

        num_of_test = distances.shape[0]
        Y_test = np.zeros(num_of_test)

        for test_index in xrange(num_of_test):
            
        return Y_test
    def compute_distances_two_loops(self, X_test):
        num_of_test = X_test.shape[0]
        num_of_train = self.X_train.shape[0]
        distances = np.zeros((num_of_test, num_of_train))
        for test_index in xrange(num_of_test):
            for train_index in xrange(num_of_train):
                distances[test_index, train_index] = np.sum(np.abs(X_test[test_index, :] - self.X_train[train_index, :]))
        return distances
                
