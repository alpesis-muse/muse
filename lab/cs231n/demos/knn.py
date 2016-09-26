import numpy as np
import matplotlib.pyplot as plt

import settings
from data.datasets import CIFAR10
from models.classifiers.k_nearest_neighbor import KNearestNeighbor


def visualize(data, classes=settings.CIFAR10_CLASSES, samples_per_class=7):

    total_classes = len(classes)

    for index, cifar10_class in enumerate(classes):
        indexes = np.flatnonzero(data[1] == index)
        sample_indexes = np.random.choice(indexes, samples_per_class, replace=False)
        for i, sample_index in enumerate(sample_indexes):
            plt_index = i * total_classes + index + 1
            plt.subplot(samples_per_class, total_classes, plt_index)
            plt.imshow(data[0][sample_index].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cifar10_class)
    plt.show()
    

def sampling(data, quota):
    mask = range(quota)
    return data[mask]


def main():
    """"""

    # load data
    cifar10 = CIFAR10(settings.CIFAR10)
    data_train = cifar10.load_train('data_batch_%d')
    data_test = cifar10.load_test('test_batch')
    print data_train[0].shape
    print data_train[1].shape
    print data_test[0].shape
    print data_test[1].shape
    # visualize(data=data_train)

    # sampling
    X_train = sampling(data_train[0], 5000)
    Y_train = sampling(data_train[1], 5000)
    X_test = sampling(data_test[0], 500)
    Y_test = sampling(data_test[1], 500)

    # reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print X_train.shape, X_test.shape

    # train
    classifier = KNearestNeighbor()
    classifier.train(X_train, Y_train)
    distances = classifier.compute_distances_two_loops(X_test)
    print distances.shape
    plt.imshow(distances, interpolation='none')
    plt.show()


if __name__ == '__main__':

    main()
