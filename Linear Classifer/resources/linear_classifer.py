import sys

import numpy as np

from resources.config import CIFAR10_PATH, LEARNING_RATE
from resources.config import LAMBDA
from resources.config import DELTA
from resources.load_CIFAR10 import load_CIFAR10


class SVM:
    weight = np.ones((1, 1))
    number_category = 0
    model = 0

    def __init__(self):
        pass

    def training(self, features_vector, labels, delta, lambda_loss):
        """
        This function using for training model SVM.

        This version pre-processing all data input. In case of images, this corresponds to computing mean images across
        the training images and subtracting it from every images to get images where the pixel range from approximately
        [-127... 127]. Then scale that to range from [-1... 1].

        The Loss function is Multi-class Support Vector Machine loss.

        Parameter
        ---------

        features_vector - numpy array size (n x i)

        labels - numpy array size (n x 1)

        Return:
        -------
        :return: model: numpy array
        """

        # Check size data
        if features_vector.shape[0] != labels.shape[0] or features_vector.shape[0] == 0:
            print("Error. The data shape != label shape")
            return
        else:
            # Pre processing features
            features_vector = self.pre_processing_features(features_vector.astype(np.float16))
            print("Features vector shape: ", features_vector.shape)
            # Create weight function
            weight = self.create_weight(features_vector, labels)
            print(weight.shape)
            # For loop calculate loss and change weight
            for i in range(0, features_vector.shape[0] - 1):
                print("Training: ", i, "/", features_vector.shape[0] - 1)
                loss = self.loss_function(features_vector[i], labels[i], weight, delta=delta, lambda_loss=lambda_loss)
                weight = weight - LEARNING_RATE * loss

            self.model = weight
            return weight

    def pre_processing_features(self, features):
        """
        In case of images, this corresponds to computing mean images across
        the training images and subtracting it from every images to get images where the pixel range from approximately
        [-127... 127]. Then scale that to range from [-1... 1].
        Then concatenate one column 1 at least.

        :param features: numpy array
        :return: features: numpy array
        """
        pre_features = features
        for i in range(0, features.shape[0] - 1):
            pre_features[i] = (pre_features[i] - 127) / 127

        pre_features = np.concatenate((pre_features, np.ones((features.shape[0], 1), dtype=np.float16)), axis=1)
        return pre_features

    def loss_function(self, x, label, w, delta=1, lambda_loss=2):
        """
        This is Multi-class Support Vector Machine loss. The expression has two components: the data loss and
        regularization loss. The regularization used L2 norm.

        L=1N∑i∑j≠yi[max(0,f(xi;W)j−f(xi;W)yi+Δ)]+λ∑k∑lW2k,l
        ----------------------------------------------------

        :param x: numpy array size(1 x n)
        :param label: int size 0 to m
        :param w: numpy array size(m x n)
        :param delta: const value
        :param lambda_loss: const value
        :return: loss: numpy array size(m x n)
        """
        # Calculate scores
        score = w.dot(x)
        score = np.maximum(0, score - score[label] + delta)
        score[delta] = 0
        # Calculate hinge loss, sum all distance score
        hinge_loss = np.sum(score)

        # Calculate regularization loss
        regularization_loss = np.sum(np.abs(w))

        # Extra loss
        loss = hinge_loss + lambda_loss * regularization_loss
        return loss

    def create_weight(self, features_vector, labels):
        """
        Create weight function. The function weight return have 2 parameter: weight(w) and bias(b). The features vector
        have extend by addition new column 1 at least.

        :param features_vector: numpy array size (n x i)
            i: the number features
        :param labels: numpy array size (n x 1)

        :return: weight: numpy array (n x number_set_category)
        """
        # Declare 2 dimensions of function weight
        width_w = features_vector.shape[1]
        height_w = self.get_category(labels)
        # Declare weight function
        # Adding bias function at least
        weight = np.ones((height_w, width_w))
        return weight

    def get_category(self, labels):
        """
        This function used to count how many value in numpy array (labels)

        :param labels: numpy array

        :return: number_category - int
            Number set of categories
        """

        # Find max value set of categories, that is number categories - 1
        self.number_category = np.max(labels) + 1
        return self.number_category

    def predict(self, features):
        """
        :param features: feature vector using for test
        :return: predict_labels
        """
        features = self.pre_processing_features(features.astype(np.float16))
        predict_labels = np.ones((features.shape[0], 1), dtype=int)
        for i in range(0, features.shape[0] - 1):
            score = self.model.dot(features[i])
            predict_labels[i] = np.argmax(score)
        return predict_labels

if __name__ == "__main__":
    training_images, training_labels, test_images, test_labels = load_CIFAR10(CIFAR10_PATH)
    model_svm = SVM()
    model_svm.training(training_images, training_labels, DELTA, LAMBDA)
    predict_labels = model_svm.predict(test_images)
    print("Mean: ", np.mean(predict_labels == test_labels))
    sys.exit()
