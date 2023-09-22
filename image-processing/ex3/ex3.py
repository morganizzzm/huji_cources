
import numpy as np
from matplotlib import pyplot as plt

from imageio import imread, imwrite
from scipy import signal
from skimage.color import rgb2gray
from skimage.color import rgb2yiq
from skimage.color import yiq2rgb
from scipy import ndimage, __all__
from scipy.ndimage.interpolation import map_coordinates
import os

GRAYSCALE = 1
RGB = 2


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


def blur(im, filter):
    # blurs image with the given filter horizontally and vertically
    new_im = np.apply_along_axis(np.convolve, 0, im, filter, mode="same").transpose()
    transposed = np.apply_along_axis(np.convolve, 0, new_im, filter, mode="same").transpose()
    return transposed



def blur_all_channels (im, filter):
    # blurs image with the given filter horizontally and vertically
    for i in range(3):
        im[:,:,i] = blur(im[:,:,i], filter)
    return im

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


def shape_legal(im):
    # checks if the image is at least of the dims (16*16)
    return (im.shape[0] >= 16) and (im.shape[1] >= 16)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
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
    pyr = []
    filter_vec = create_filter(filter_size)
    cur_img = im
    next_img = reduce(im, filter_vec)
    i = 0
    while shape_legal(next_img) and i < max_levels - 1:
        pyr.append(cur_img - expand(next_img, filter_vec))
        cur_img = next_img
        next_img = reduce(cur_img, filter_vec)
        i += 1
    pyr.append(cur_img)
    pyr[-1] = pyr[-1]*1.5
    return pyr, np.reshape(filter_vec, (1, filter_size))


def expand_to_original(pyr, filter_vec):
    # expands images in pyr to the original image
    # dimensions
    result = []
    for l in pyr:
        while l.shape != pyr[0].shape:
            l = expand(l, filter_vec)
        result.append(l)
    return result


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

        :param lpyr: Laplacian pyramid
        :param filter_vec: Filter vector
        :param coeff: A python list in the same length as the number of levels in
                the pyramid lpyr.
        :return: Reconstructed image"""
    expanded = expand_to_original(lpyr, filter_vec[0])
    image = np.zeros(expanded[0].shape)
    for i in range(len(lpyr)):
        image += expanded[i] * coeff[i]
    return image


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    n = len(pyr[0])
    stretched_pyr = []
    for i in range(levels):
        p = np.array(pyr[i])
        p.clip(0, 1)
        p = (p - np.min(p)) / (np.max(p - np.min(p)))
        stretched_pyr.append(p)
    ans = stretched_pyr[0]
    i = 1
    while i != levels:
        conc_lvl = np.concatenate((stretched_pyr[i],
                                   np.zeros((n-len(stretched_pyr[i]),
                                             len(stretched_pyr[i][0])))))
        ans = np.concatenate((ans, conc_lvl), axis=1)
        i += 1
    return ans


def display_pyramid(pyr, levels):
    """
    	display the rendered pyramid
    	"""
    plt.imshow(render_pyramid(pyr, levels), cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
         Pyramid blending implementation
        :param im1: input grayscale image
        :param im2: input grayscale image
        :param mask: a boolean mask
        :param max_levels: max_levels for the pyramids
        :param filter_size_im: is the size of the Gaussian filter (an odd
                scalar that represents a squared filter)
        :param filter_size_mask: size of the Gaussian filter(an odd scalar
                that represents a squared filter) which defining the filter used
                in the construction of the Gaussian pyramid of mask
        :return: the blended image
        """


    La, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    Lb = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    Gm = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    Lc = compute_lapl_pyr_blended(Gm, La, Lb)
    return laplacian_to_image(Lc, filter_vec, [1 for _ in range(La[0].shape[0])])


def compute_lapl_pyr_blended(Gm, La, Lb):
    # computes Lc according to the formula
    return [Gm[i] * La[i] + (1 - Gm[i]) * Lb[i] for i in range(len(La))]



def blending_example1():
    """
        Perform pyramid blending on two images RGB and a mask
        :return: image_1, image_2 the input images, mask the mask
            and out the blended image
        """
    im1 = read_image(relpath("externals/fei_fei_li_original.png"), RGB)
    im2 = read_image(relpath("externals/fei_fei_li_inpainting_mask.png"), RGB)
    mask_or = read_image(relpath("externals/fei_fei_li_inpainting_mask.png"), RGB)
    mask = mask_or
    max_levels = 3
    filter_size = 5
    im_blend = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask[:,:, 0], max_levels,
                                filter_size, filter_size)
    for i in range(1, 3):
        curr_blend = pyramid_blending(im1[:, :, i], im2[:, :, i], mask[:,:, 0],
                                      max_levels, filter_size, filter_size)
        im_blend = np.dstack((im_blend, curr_blend))


    plt.imshow(im_blend)
    plt.show()

    return im1, im2, mask, im_blend


def blending_example2():
    """
        Perform pyramid blending on two images RGB and a mask
        :return: image_1, image_2 the input images, mask the mask
            and out the blended image
        """
    im1 = read_image(relpath("externals/02im1.jpg"), RGB)
    im2 = read_image(relpath("externals/02im2.jpg"), RGB)
    mask_or = read_image(relpath("externals/02mask.jpg"), RGB)
    mask = mask_or.astype(np.bool)
    max_levels = 4
    filter_size = 4
    im_blend = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask[:, :, 0],
                                max_levels,
                                filter_size, filter_size)
    for i in range(1, 3):
        curr_blend = pyramid_blending(im1[:, :, i], im2[:, :, i],
                                      mask[:, :, 0],
                                      max_levels, filter_size, filter_size)
        im_blend = np.dstack((im_blend, curr_blend))

    plt.figure()
    plt.imshow(im1)
    plt.figure()
    plt.imshow(im2)
    plt.figure()
    plt.imshow(mask_or, cmap="gray")
    plt.figure()
    plt.imshow(im_blend)
    plt.show()

    return im1, im2, mask, im_blend


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


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def gaussian(M, std, sym=True):

    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = np.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = np.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def create_figure(images, phrases, name_of_fig, rows=2, columns=3):
    fig = plt.figure(figsize=(18, 9))

    for i in range(len(images)):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(phrases[i])
    plt.savefig("/Users/home/PycharmProjects/imp3/externals/"+name_of_fig)
    plt.show()


def create_masked(original_png, mask_png, path_to_save):
    im1 = read_image(
        original_png,
        RGB)
    im2 = read_image(
        mask_png,
        RGB)
    ans = np.ones((1024, 1024))
    i1 = 325
    # j1 = random.randint(1, synth_images.shape[3])
    j1 = 509
    # i2 = random.randint(i1, synth_images.shape[2])
    i2 = 696
    # j2 = random.randint(j1, synth_images.shape[3])
    j2 = 751
    zeros = np.zeros((i2 - i1, j2 - j1))
    ans[i1:i2, j1:j2] = zeros

    im2 = np.array(im2[:, :, 0:3])
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            if ans[i][j] == 0 and ans[i][j]==0:
                print(i)
                print(j)



    # plt.imsave(path_to_save, im_numpy)


if __name__ == '__main__':
    # pass
    # print(create_filter(5))
    # blending_example2()
    image1 = np.asarray(read_image(
        "/Users/home/PycharmProjects/imp3/externals/fei_fei_li_original.png",
        RGB))
    image2 = np.asarray(read_image(
        "/Users/home/PycharmProjects/imp3/externals/fei_fei_li_inpainting_mask.png",
        RGB))
    create_masked("/Users/home/PycharmProjects/imp3/externals/fei_fei_li_original.png",
                  "/Users/home/PycharmProjects/imp3/externals/fei_fei_li_inpainting_mask.png",
                  "/Users/home/PycharmProjects/imp3/externals/fei_masked.png")
    # arr = []
    # for i in range(image1.shape[0]):
    #     row=[]
    #     for j in range(image1.shape[1]):
    #         k=0.299*image1[i][j][0]+ 0.587 * image1[i][j][1] + \
    #                0.114* image1[i][j][2]
    #         row.append(k)
    #     arr.append(row)
    # arr = np.array(arr)
    # plt.imshow(arr, cmap="gray")
    # print(arr)
    # plt.show()
    # kernel = create_filter(30)
    # blurred = blur_all_channels(image1, kernel)
    # plt.imshow(blurred)
    # plt.show()
    # ans = gaussian(image1, std=125)
    # plt.imshow(ans)
    # plt.show()
    # image2 = read_image(
    #     "/Users/home/PycharmProjects/imp3/externals/fei_fei_li_inpainting_mask.png",
    #     RGB)
    # image3 = read_image(
    #     "/Users/home/PycharmProjects/imp3/externals/fei_1.jpeg",
    #     RGB)
    #
    # image4 = read_image(
    #     "/Users/home/PycharmProjects/imp3/externals/fei_masked.png",
    #     RGB)
    # image5 = read_image(
    #     "/Users/home/PycharmProjects/imp3/externals/fei_masked.jpeg",
    #     RGB)
    # create_figure([image1, image2, image3, image4, image5], ["First", "Second", "Third", "Blalal", "Iosdfio"]
    #               ,"new_fig.jpeg")
    #
    # create_masked("/Users/home/PycharmProjects/imp3/externals/fei_fei_li_original.png",
    #  "/Users/home/PycharmProjects/imp3/externals/fei_fei_li_inpainting_mask.png" ,)

    # # pyr, f = build_gaussian_pyramid(im, 5, 5)
    # # # # l = expand(im, create_filter(5).reshape(1, 5))
    # # # # print(im.shape)
    # # # # # # # #
    # # #
    # plt.figure()
    # plt.imshow(im, cmap="gray")
    # pyr, f = build_laplacian_pyramid(im, 5, 3)
    # # # display_pyramid(pyr, 4)
    # l = laplacian_to_image(pyr, f, [1,1,1,1,1])
    # # # # blending_example1()
    # # # print(l.shape)
    # # # # imn = laplacian_to_image(pyr,f, [0.5,1,0.2,1,1])
    # # # # # imn = expand_to_original([len(pyr[0]), len(pyr[0][0])], pyr[4], f)
    # # # # # for p in pyr:
    # # # # #     plt.figure()
    # # # # plt.figure()
    # # # # plt.imshow(red, cmap="gray")
    # # #
    # plt.figure()
    # plt.imshow(l, cmap="gray")
    # plt.show()
    # # sim = subsample(im)
    # # conv_im = blur(im, [0.25, 0.5, 0.25])
