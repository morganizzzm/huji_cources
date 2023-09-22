from imageio import imread
from scipy.signal import convolve2d
import numpy as np
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Read image
    @filename: file name
    @representation: 1 == gray, other=RGB
    """
    im = imread(filename)
    if representation == GRAYSCALE:
        im = rgb2gray(im)
    im = np.divide(im, 255)
    return im


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



def reduce(im, blur_filter):
    """
       Reduces an image by a factor of 2 using the blur filter
       :param im: Original image
       :param blur_filter: Blur filter
       :return: the downsampled image
       """

    # reduces the image by blurring it and then removing every second row
    # and every second element in every row
    return subsample(blur(im, blur_filter))


def expand(im, blur_filter):
    """
       Expand an image by a factor of 2 using the blur filter
       :param im: Original image
       :param blur_filter: Blur filter
       :return: the expanded image
       """
    # expands the image by adding row of zeros every second row
    # and every second element and then blurring
    yy = zero_padding(im)
    return blur(yy,2*blur_filter)


def zero_padding_rows(im):
    # adds zeros every second row
    n, m = im.shape
    new_img = np.zeros((n*2, m))
    new_img[::2] = im
    return new_img


def zero_padding(im):
    # add row of zeros every second row
    new_im = zero_padding_rows(im)
    # add column of zeros every second column
    new_im = zero_padding_rows(new_im.transpose())
    return new_im.transpose()


def subsample(im):
    # remove every second row
    new_im = im[::2]
    # remove every second column
    new_im = new_im.transpose()[::2].transpose()
    return new_im


def create_filter(filter_size):
    # created a row of binomial coeffs of the given size
    # and then normalizes them
    i = 2
    conv_filter = [1, 1]
    while i != filter_size:
        conv_filter = np.convolve([1, 1], conv_filter)
        i += 1
    return conv_filter / sum(conv_filter)


def shape_legal(im):
    # checks if the image is at least of the dims (16*16)
    return (im.shape[0] >= 16) and (im.shape[1] >= 16)



def blur(im, filter):
    # blurs image with the given filter horizontally and vertically
    new_im = np.apply_along_axis(np.convolve, 0, im, filter, mode="same").transpose()
    transposed = np.apply_along_axis(np.convolve, 0, new_im, filter, mode="same").transpose()
    return transposed


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """

    pyr = [im]
    i = 1
    filter_vec = create_filter(filter_size)
    cur_img = im
    while i != max_levels and shape_legal(cur_img):
        new_img = reduce(cur_img, filter_vec)
        if shape_legal(new_img):
            pyr.append(new_img)
        cur_img = new_img
        i += 1
    return pyr, np.reshape(filter_vec, (1, filter_size))
