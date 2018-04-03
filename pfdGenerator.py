# encoding=utf-8
import numpy as np
import pickle
import os

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and
shuffling of the data.
The other source of inspiration is the ImageDataGenerator by @fchollet in the
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I
wrote my own little generator.
"""


class ImageDataGenerator:
    def __init__(self, images, labels, shuffle=False
                 , scale_size=(64, 64), nb_classes=2):  # mean=np.array([127.5]),np.array(,,)

        # Init params

        self.images = images
        self.labels = labels
        self.scale_size = scale_size
        self.shuffle = shuffle
        self.pointer = 0
        self.n_classes = nb_classes

        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        # python 3....
        # images = self.images.copy()
        # labels = self.labels.copy()
        images = self.images
        labels = self.labels
        self.images = []
        self.labels = []

        # create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        # update pointer
        self.pointer += batch_size

        # Read images
        #images = np.ndarray(
        images = np.zeros(
            [batch_size, self.scale_size[0], self.scale_size[1], 3])  # the last parameter is image channel
        for i in range(len(paths)):

            '''
            # rescale image
            try:
                import cv2
                img = cv2.resize(paths[i], (self.scale_size[0], self.scale_size[0]))
            except:
                print(paths[i])
            '''
            #print '*** paths[i][:,:,0]:', paths[i][:,:,0]
            img = paths[i]
            img = img.astype(np.float32)

            images[i,:,:,0] = img.reshape((64,64))
            #print images[i,...].shape
            #print images[i,:,:,1]

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        # return array of images and labels
        return images, one_hot_labels


# if __name__ == '__main__':
#     train_file = "../datasets/pfd_data/FvPs.pkl"
#     val_file = "../datasets/pfd_data/FvPs.pkl"
#     target_file = "../datasets/pfd_data/PulsarFlag.pkl"
#     pg = ImageDataGenerator(train_file, target_file, scale_size=(64, 64), nb_classes=2)

