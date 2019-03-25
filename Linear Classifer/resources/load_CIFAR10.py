import pickle
import os
import numpy as np
import sys
from resources.config import CIFAR10_PATH


def unpickle(file):
    """
    This function use to get data of batch file

    :param

    file: string
        Batch file path

    :return

    dict: dictionary
        Variable store data from batch file
    """
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def load_CIFAR10(directory_path):
    """
    This function used to get all data CIFAR10

        :parameter

        directory_path: string
            Directory path

        :returns

        training_images: narray
            Store 50000 image. Shape = 50000 x 32 x 32 x 3

        training_label: narray
            Corresponding training images. Size = 50000. Hold the training label form 0 to 9

        test_images: narray
            Store 10000 image. Shape = 10000 x 32 x 32 x 3

        test_label: narray
            Corresponding test images. Size = 10000. Hold the test label from 0 to 9
    """

    training_images = np.array([])
    training_labels = np.array([])
    test_images = np.array([])
    test_labels = np.array([])

    # Load data batch file
    for file_name in os.listdir(directory_path):
        # Get data and test path in directory file
        if file_name.rfind("data_batch") == 0:
            data_path = os.path.join(directory_path, file_name)
            # Get data store in batch file
            training_image, training_label = load_CIFAR10_batch(data_path)

            # Store data training and label training.
            # If test_images or label_images empty then assign test_image to it.
            # else concatenate them using np.concatenate
            if training_images.shape[0] == 0:
                training_images = np.array(training_image)
                training_labels = np.array(training_label)
            else:
                training_images = np.concatenate((training_images, training_image), axis=0)
                training_labels = np.concatenate((training_labels, training_label), axis=0)

        elif file_name.rfind("test_batch") == 0:
            test_patch = os.path.join(directory_path, file_name)
            # Get data store in batch file
            test_images, test_labels = load_CIFAR10_batch(test_patch)

    return training_images, training_labels, test_images, test_labels


def load_CIFAR10_batch(batch_file_path: str):
    """
    This function used to get data in batch file of CIFAR10

        :parameter

        batch_file_path: string
            Store batch file path

        :returns

        image: narray
            Store 10000 image. Shape = 10000 x 3072

        label: narray
            Corresponding training images. Size = 10000. Hold the training label form 0 to 9
    """
    # Get data store in batch file
    dict = unpickle(batch_file_path)
    image = dict[b'data']
    label = dict[b'labels']
    image = np.array(image)
    label = np.array(label)

    return image, label


if __name__ == "__main__":

    training_images, training_labels, test_images, test_labels = load_CIFAR10(CIFAR10_PATH)
    print(training_images.shape[0])
    print(training_labels.shape[0])
    sys.exit()
