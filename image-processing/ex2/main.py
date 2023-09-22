# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from matplotlib import pyplot as plt

from imageio import imread, imwrite
from scipy.io import wavfile
from skimage.color import rgb2gray
from skimage.color import rgb2yiq
from skimage.color import yiq2rgb
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates


GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def IDFT(fourier_signal):
    N = fourier_signal.shape[0]
    # our w's [frequencies] from 0 to N-1
    x = np.arange(N)
    # u[i] = [ x[i] ]
    u = x.reshape(N, 1)
    # it's a matrix where [i,j] contains i*j
    T = (x * u)
    change_basis_matrix = np.exp(2 * np.pi * T * 1j / N)
    # we are allowed to assume that the result is going to be real
    return np.real(change_basis_matrix.dot(fourier_signal) / N)


def DFT(signal):
    N = signal.shape[0]
    # our sample points from 0 to N-1
    x = np.arange(N)
    # u[i] = [ x[i] ]
    u = x.reshape(N, 1)
    # it's a matrix where [i,j] contains i*j
    T = (x * u)
    # change basis matrix from the tirgul -- with e^(-2pi*i/N)
    change_basis_matrix = np.exp((-2) * np.pi * T * 1j / N)
    return change_basis_matrix.dot(signal).astype(np.complex128)


def IDFT2(fourier_image):
    N, M = fourier_image.shape
    idft2 = np.zeros([N, M], dtype=np.complex128)
    # computer fourier in y direction
    ans = compute_Fourier_in_dir(M, idft2, fourier_image, IDFT, 0)
    # compute it in x direction
    return np.real(compute_Fourier_in_dir(N, ans, ans, IDFT, 1))


def compute_Fourier_in_dir(dim, ans, image, fourier, coeff):
    for i in range(dim):
        image_i = image[:, i] if coeff == 0 else image[i, :]
        fourier_image_i = fourier(image_i)
        if coeff == 0:
            ans[:, i] = fourier_image_i
        else:
            ans[i, :] = fourier_image_i
    return ans


def DFT2(image):
    N, M = image.shape
    dft2 = np.zeros([N, M], dtype=np.complex128)
    # compute fourier transform for each column
    ans = compute_Fourier_in_dir(M, dft2, image, DFT, 0)
    return compute_Fourier_in_dir(N, ans, ans, DFT, 1)


def change_rate(filename, ratio):
    rate, signal = wavfile.read(filename)
    wavfile.write("change_samples.wav", int(rate*ratio), signal)


def resize_with_foo(foo, signal, ratio):
    return foo(signal, ratio)


def resize_spectrogram(data, ratio):
    # we slow down
    spectogram = stft(data)
    spectogram_resized = np.apply_along_axis(resize, 1, spectogram, ratio)
    return istft(spectogram_resized)


def magnitude(dx, dy):
    # TESTED
    # computes magnitude according to the formula
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)


def resize_vocoder(data, ratio):
    # TESTED
    return istft(phase_vocoder(stft(data), ratio))


def resize(data, ratio):
    # TESTED
    if ratio < 1:
        # we need to add zeros from left and from the right of the signal
        freq_to_leave = int(np.floor(len(data) * ((1/ratio)-1)))
        start, end = get_pad_start_end(freq_to_leave)
        # add zeros
        dft_padded = np.pad(DFT(data), (start, end))
        return IDFT(dft_padded)
    # decide where to trim initial signal
    freq_to_leave = int(np.ceil(len(data) * (1 - 1 / ratio)))
    start, end = get_pad_start_end(freq_to_leave)
    # compute DFT
    dft = DFT(data)
    # shift zero-frequency component
    dft_shifted = np.fft.fftshift(dft)
    # clipping the high frequencies
    dft_shifted_clipped = dft_shifted[start: len(dft_shifted)-end]
    # shift zero-frequency component back
    dft_clipped = np.fft.ifftshift(dft_shifted_clipped)
    return IDFT(dft_clipped)


def get_pad_start_end(freq_to_leave):
    # divides given freq_to_leave in 2 numbers
    # in case freq_to_leave is even the first num returned = the second num
    # returned, else first = second - 1
    if freq_to_leave % 2 == 0:
        return int(freq_to_leave/2), int(freq_to_leave/2)
    return int((freq_to_leave-1)/2), int((freq_to_leave-1)/2)+1


def change_samples(filename, ratio):
    """
    reducing the number of samples using Fourier
    :param filename:
    :param ratio:
    :return:
    """
    rate, signal = wavfile.read(filename)
    new_signal = resize_with_foo(resize, signal, ratio)
    wavfile.write("change_samples.wav", rate, new_signal)


def conv_der_single(im, reshape_i, reshape_j):
    # derives with convolution in single direction
    kernel = (np.array([0.5, 0, -0.5]).reshape(reshape_i, reshape_j))
    return signal.convolve2d(im, kernel, "same")


def conv_der(im):
    # compute dx with convolution
    dx = conv_der_single(im, 3, 1)
    # compute dy with convolution
    dy = conv_der_single(im, 1, 3)
    return magnitude(dx, dy)


def compute_dx(im, dft):
    M = im.shape[0]
    weighted_der = np.multiply((2*np.pi*1j/M) * np.arange(-M/2, M/2), dft.T).T
    shifted_weighted_der = np.fft.ifftshift(weighted_der)
    return IDFT2(shifted_weighted_der)


def compute_dy(im, dft):
    N = im.shape[1]
    weighted_der = np.multiply((2*np.pi*1j/N) * np.arange(-N/2, N/2), dft)
    shifted_weighted_der = np.fft.ifftshift(weighted_der)
    return IDFT2(shifted_weighted_der)


def fourier_der(im):
    dft = np.fft.fftshift(DFT2(im))
    return magnitude(compute_dx(im, dft), compute_dy(im, dft))


def compute_mag_spectrum(im):
    return np.log(1 + np.abs(im))



#functions from ex2_helper and ex1

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]

    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)

    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


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

def histogram_equalize_grayscale(im_orig):
    # translate image from float to int
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

def create_eq_img(map, img):
    im_eq = map[img.astype(np.int64)].astype(np.float64)
    return im_eq


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
    # change_samples("/Users/home/PycharmProjects/imp2/aria_4kHz.wav", 2)

    # im = read_image("/Users/home/PycharmProjects/imp2/monkey.jpg", GRAYSCALE)
    # im_ft = np.fft.fft2(im)
    # plt.imshow(im_ft, cmap="gray")
    # plt.show()
    # der = conv_der(im)
    # plt.imshow(der, cmap="gray")
    # plt.show()
    # a = [1, 2, 3, 4, 5, 6]
    # print(np.pad(a, (2, 4)))
    x = np.arange(256)
    # y = np.cos(2 * np.pi * x / 60)
    # y += max(y)
    img = np.array([[np.cos(2 * np.pi*((4*i)/256)) for j in range(256)] for i in range(256)],
                   dtype=np.uint8)
    # print(img)
    # iii = histogram_equalize(img)
    # print(iii[1])
    # print("nexr")
    # print(iii[2])
    dft = np.fft.fft2(img)
    #
    # kernel = (np.array([1, -2, 1]).reshape(1, 3))
    # conv_im = signal.convolve2d(img, kernel, "same")

    # conv_im = conv_der(img)
    # dft_conv= np.fft.fft2(conv_im)
    # #
    # # idft = np.real(IDFT2(dft))
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = np.real(dft)
    plt.imshow(img, cmap="gray")
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
