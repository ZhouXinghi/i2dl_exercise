"""
Definition of image-specific transform classes
"""

# pylint: disable=too-few-public-methods

import numpy as np


class RescaleTransform:
    """Transform class to rescale images to a given range"""
    def __init__(self, out_range=(0, 1), in_range=(0, 255)):
        """
        :param out_range: Value range to which images should be rescaled to
        :param in_range: Old value range of the images
            e.g. (0, 255) for images with raw pixel values
        """
        self.min = out_range[0]
        self.max = out_range[1]
        self._data_min = in_range[0]
        self._data_max = in_range[1]

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # Rescale the given images:                                            #
        #   - from (self._data_min, self._data_max)                            #
        #   - to (self.min, self.max)                                          #
        ########################################################################


        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                for k in range(images.shape[2]):
                    images[i][j][k]  = self.min + (self.max - self.min) * (images[i][j][k] - self._data_min) / (self._data_max - self._data_min)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images
    

def compute_image_mean_and_std(images):
    """
    Calculate the per-channel image mean and standard deviation of given images
    :param images: numpy array of shape NxHxWxC
        (for N images with C channels of spatial size HxW)
    :returns: per-channels mean and std; numpy array of shape C
    """
    mean, std = None, None
    ########################################################################
    # TODO:                                                                #
    # Calculate the per-channel mean and standard deviation of the images  #
    # Hint: You can use numpy to calculate the mean and standard deviation #
    ########################################################################
    # mean = np.zeros(images.shape[3])
    # std = np.zeros(images.shape[3])
    #
    # channel_values = np.array()
    # for n in range(images.shape[0]):
    #     for i in range(images.shape[1]):
    #         for j in range(images.shape[2]):
    #             for k in range(images.shape[3]):
    #                 numpy.append(channel_values[k], images[n][i][j][k])
    # for i in range(images.shape[0]):
    #     mean[i] = np.mean(channel_values[i])
    #     std[i] = np.std(channel_values[i])

    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return mean, std


class NormalizeTransform:
    """
    Transform class to normalize images using mean and std
    Functionality depends on the mean and std provided in __init__():
        - if mean and std are single values, normalize the entire image
        - if mean and std are numpy arrays of size C for C image channels,
            then normalize each image channel separately
    """
    def __init__(self, mean, std):
        """
        :param mean: mean of images to be normalized
            can be a single value, or a numpy array of size C
        :param std: standard deviation of images to be normalized
             can be a single value or a numpy array of size C
        """
        self.mean = mean
        self.std = std

    def __call__(self, images):
        ########################################################################
        # TODO:                                                                #
        # normalize the given images:                                          #
        #   - substract the mean of dataset                                    #
        #   - divide by standard deviation                                     #
        ########################################################################

        n, h, m, c = images.shape
        mean, std = compute_image_mean_and_std(images)
        for i in range(n):
            for j in range(h):
                for k in range(m):
                    for l in range(c):
                        images[i][j][k][l] = (images[i][j][k][l] - mean[l]) / std[l]

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return images


class ComposeTransform:
    """Transform class that combines multiple other transforms into one"""
    def __init__(self, transforms):
        """
        :param transforms: transforms to be combined
        """
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images
