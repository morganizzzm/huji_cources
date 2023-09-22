
import numpy as np
from imageio import imread, imwrite
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """

    if representation < GRAYSCALE or representation > RGB:
        raise Exception("Wrong input!")

    im = imread(filename)
    if representation == GRAYSCALE:
        im = rgb2gray(im)

    else:
        im = im.astype(np.float64)
        im /= 255
    return im


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """

    im = read_image(filename, representation)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    r = imRGB[:, :, 0]
    g = imRGB[:, :, 1]
    b = imRGB[:, :, 2]
    imYIQ = np.zeros(shape=imRGB.shape)
    imYIQ[:, :, 0] = 0.299 * r + 0.587 * g + 0.114 * b
    imYIQ[:, :, 1] = 0.586 * r - 0.275 * g - 0.321 * b
    imYIQ[:, :, 2] = 0.212 * r - 0.523 * g + 0.311 * b
    return imYIQ


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    y = imYIQ[:, :, 0]
    i = imYIQ[:, :, 1]
    q = imYIQ[:, :, 2]
    imRGB = np.zeros(shape=imYIQ.shape)
    imRGB[:, :, 0] = y + 0.956 * i + 0.621 * q
    imRGB[:, :, 1] = y - 0.272 * i - 0.647 * q
    imRGB[:, :, 2] = y - 1.106 * i + 1.703 * q
    return imRGB


def create_eq_img(map, img):
    im_eq = map[img.astype(np.int64)].astype(np.float64)
    return im_eq


def histogram_equalize_grayscale(im_orig):
    # translate image from fl oat to int
    im_orig_int = (im_orig * 255).astype(np.int64)
    # create histogram
    hist_orig, bins = np.histogram(im_orig_int, bins=range(257))
    # create cumulative histogram
    c = np.cumsum(hist_orig)
    #  first gray level for which C(m)!=0
    if len(np.nonzero(hist_orig)[0]) == 1:
        return [im_orig, im_orig, im_orig]
    m = np.nonzero(hist_orig)[0][0]
    # compute look up table T
    T = generate_T(c, m)
    # apply look up table
    im_eq = create_eq_img(T, im_orig_int)
    hist_eq, bins_eq = np.histogram(im_eq, bins=range(257))
    # transform back to [0..1]
    im_eq /= 255
    return [im_eq, hist_orig, hist_eq]


def generate_T(c, m):
    """
    :param c: normalized cumulative histogram
    :param m: first gray level for which C(m)!=0
    :return: look up table for  equaized image
    """
    T = np.zeros(len(c))
    for k in range(len(c)):
        T[k] = int(255 * ((c[k] - c[m]) / (c[255] - c[m])))
    return T


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """

    if len(im_orig.shape) == 3:
        imYIQ = rgb2yiq(im_orig)
        info = histogram_equalize_grayscale(imYIQ[:, :, 0])
        imYIQ[:, :, 0] = info[0]
        info[0] = yiq2rgb(imYIQ)
        return info

    return histogram_equalize_grayscale(im_orig)


######################################################################
######################################################################
######################################################################
######################################################################

def guess_z_sabir(n_quant, h):
    """
    :param n_quant:
    :param h:
    :param resolution: pixels in screen
    :return:
    """
    # avg number of pixels in single sector
    c = np.cumsum(h)
    arr = np.searchsorted(c, np.int64(np.linspace(0, c[-1], n_quant+1)))
    return [i for i in arr]


def round_value_to_int(val):
    return np.floor(val).astype(int)


def round_value_to_int64(val):
    return np.floor(val).astype(np.int64)


def compute_error(z_i_s_cur, q_i_s_cur, h):
    """
    computes error according to the formula given in the ex1
    :param z_i_s_cur: an array of all z's
    :param q_i_s_cur: an array of all q's
    :param h: histogram
    :return: error
    """
    error = 0
    # from i=0 to i=k-1
    for i in range(len(z_i_s_cur) - 1):
        # from g = [z_i]+1 to [z_i+1]
        for g in range(round_to_int_initial(i, z_i_s_cur),
                       round_to_int_last(i, z_i_s_cur) + 1):
            error += ((q_i_s_cur[i] - g) ** 2) * (h[g])
    return error


def round_to_int_last(i, z_i_s_cur):
    return int(np.floor(z_i_s_cur[i + 1]))


def round_to_int_initial(i, z_i_s_cur):
    return int(np.floor(z_i_s_cur[i]) + 1)


def compute_q(initial_g, last_g, h):
    """
    computes q according to the formula in the ex 1
    :param initial_g:  rounded z[i]+1
    :param last_g: rounded z[i+1]
    :param h: histogram
    :return: the value of q_i
    """
    mone, mechane = 0, 0
    for g in range(initial_g, last_g + 1):
        mone += g * h[g]
        mechane += h[g]
    return mone / mechane


def compute_z_i_s(q_i_s_cur):
    """
    z_i = (q_i + q_i-1) / 2
    :param q_i_s_cur: array of current q_i's
    :return: array of corresponding z_i's
    """
    all_z_new = []
    for i in range(len(q_i_s_cur) - 1):
        all_z_new.append((q_i_s_cur[i] + q_i_s_cur[i + 1]) / 2)
    # add 0 at the beginning and 255 at the end
    return [0] + all_z_new + [255]


def create_im_quant(im_orig, z_i_s_cur, q_i_s_cur):
    """
    :param im_orig: original image
    :param z_i_s_cur: an array of all z's
    :param q_i_s_cur: an array of all q's
    :return: returns quantized version of the original image
    """
    quant_map = np.zeros(256)
    for i in range(len(q_i_s_cur)):
        quant_map[round_value_to_int64(z_i_s_cur[i] + 1):
                  round_value_to_int(z_i_s_cur[i + 1]) + 1] = \
            np.floor(q_i_s_cur[i])
    quant_map[0] = np.floor(q_i_s_cur[0])
    quant_map /= 255
    return quant_map[im_orig]


def compute_q_i_s(h, z_i_s_cur):
    """
    :param h: histogram
    :param z_i_s_cur: an array of all z's
    :return: an array of q's where q[i] corresponds z[i]
    """
    q = []
    for i in range(len(z_i_s_cur) - 1):
        initial_g = round_to_int_initial(i, z_i_s_cur)
        last_g = round_to_int_last(i, z_i_s_cur)
        if initial_g == last_g: continue
        q.append(compute_q(initial_g, last_g, h))
    return q


def quantize_grayscale(im_orig, n_quant, n_iter):
    """
    Perorms quantization algorithm of gray scale
    :param im_orig:  is the input grayscale or RGB image to be quantized
    (float64 image with values in [0, 1])
    :param n_quant: is the number of intensities your output im_quant image
    should have
    :param n_iter: is the maximum number of iterations of the optimization
     procedure (may converge earlier.)
    :return:
    """
    # transform img from [0..255] to [0..1]
    im_orig_int = (im_orig * 255).astype(np.int64)
    h, bins = np.histogram(im_orig_int, bins=range(257))
    # first guess of z where we divide pixels in sectors equally
    z_i_s_prev = guess_z_sabir(n_quant, h)
    z_i_s_cur = z_i_s_prev
    errors, q_i_s_cur = [], []
    for i in range(n_iter):
        # compute q_i_s
        q_i_s_cur = compute_q_i_s(h, z_i_s_cur)
        # compute error for current q_i_s and z_i_s
        errors.append(compute_error(z_i_s_cur, q_i_s_cur, h))
        # compute z_i_s
        z_i_s_cur = compute_z_i_s(q_i_s_cur)
        if z_i_s_prev == z_i_s_cur:
            errors = np.delete(errors, np.s_[i:])
            return [create_im_quant(im_orig_int, z_i_s_cur, q_i_s_cur),
                    np.array(errors)]
        z_i_s_prev = z_i_s_cur
    return [create_im_quant(im_orig_int, z_i_s_cur, q_i_s_cur),
            np.array(errors)]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    if n_quant <= 0:
        raise Exception("Wrong input!")

    # the input is rgb
    if len(im_orig.shape) == 3:
        # translate to YIQ
        imYIQ = rgb2yiq(im_orig)
        # perform quantization on Y
        info = quantize_grayscale(imYIQ[:, :, 0], n_quant, n_iter)
        imYIQ[:, :, 0] = info[0]
        # translate back to RGB
        info[0] = yiq2rgb(imYIQ)
        return info

    return quantize_grayscale(im_orig, n_quant, n_iter)


def guess_z_sabir_rgb(n_quant, im_orig, color):
    """
    divide pixels approximately in n_quant sectors
    """
    c = im_orig[:, :, color]
    h, bins = np.histogram(c, bins=range(257))
    return guess_z_sabir(n_quant, h)


def get_sector_idx(sectors, color):
    """
    decide to which sector  out of given the pixel belongs
    """
    for i in range(1, len(sectors)):
        if sectors[i - 1] <= color <= sectors[i]:
            return i
    return 0


def calculate_avg_sector_color(sectors_color_n_quant):
    """
    calculate avg color values for each sector
    """
    sectors_avg_c = np.zeros(len(sectors_color_n_quant))
    for i in range(1, len(sectors_color_n_quant)):
        sectors_avg_c[i] = np.mean(sectors_color_n_quant[i]).astype(int)
    return sectors_avg_c


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    im_orig_int = (im_orig * 255).astype(np.int64)
    im_orig_int_copy = np.copy(im_orig_int)
    sectors_red = guess_z_sabir_rgb(n_quant, im_orig_int, 0)
    sectors_green = guess_z_sabir_rgb(n_quant, im_orig_int, 1)
    sectors_blue = guess_z_sabir_rgb(n_quant, im_orig_int, 2)

    sectors_red_n_quant = [[] for _ in range(len(sectors_red) + 1)]
    sectors_green_n_quant = [[] for _ in range(len(sectors_red) + 1)]
    sectors_blue_n_quant = [[] for _ in range(len(sectors_red) + 1)]

    for row in im_orig_int:
        for pixel in row:
            r, g, b = pixel[0], pixel[1], pixel[2]
            sectors_red_n_quant[get_sector_idx(sectors_red, r)].append(r)
            sectors_green_n_quant[get_sector_idx(sectors_green, g)].append(g)
            sectors_blue_n_quant[get_sector_idx(sectors_blue, b)].append(b)

    sectors_avg_red = calculate_avg_sector_color(sectors_red_n_quant)
    sectors_avg_green = calculate_avg_sector_color(sectors_green_n_quant)
    sectors_avg_blue = calculate_avg_sector_color(sectors_blue_n_quant)

    for i, row in enumerate(im_orig_int):
        for j, pixel in enumerate(row):
            r, g, b = pixel[0], pixel[1], pixel[2]
            im_orig_int_copy[i, j][0] = \
                sectors_avg_red[get_sector_idx(sectors_red, r)]
            im_orig_int_copy[i, j][1] = \
                sectors_avg_green[get_sector_idx(sectors_green, g)]
            im_orig_int_copy[i, j][2] = \
                sectors_avg_blue[get_sector_idx(sectors_blue, b)]
    return im_orig_int_copy / 255


# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :],
#                    np.array([255] * 6)[None, :]])
#     grad = np.tile(x, (256, 1))
#     # ing= histogram_equalize(grad/255)[0]
#     ing = histogram_equalize(read_image("/Users/home/PycharmProjects/imp1/venv/3096491.jpg", RGB))[0]
#     # # grad = np.hstack(grad, [0])
#     ing = quantize_rgb(ing, 3)
#     plt.imshow(ing)
#     plt.show()

# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
