# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates
import shutil
from imageio import imwrite, imread
import sol4_utils


def compute_image_dx(im):
    # TESTED
    # computes Ix
    return scipy.signal.convolve2d(im, np.array([1, 0, -1]).reshape((1, 3)),
                                   "same")


def compute_image_dy(im):
    # TESTED
    # computes Iy
    return scipy.signal.convolve2d(im, np.array([1, 0, -1]).reshape((3, 1)),
                                   "same")


def compute_corner_response(im_dx_square, im_dy_square, im_dx_dy, im_dy_dx):
    # TESTED
    # computes corner response
    M_det = im_dx_square * im_dy_square - im_dx_dy * im_dy_dx
    M_trace = im_dx_square + im_dy_square
    return M_det - 0.04 * M_trace ** 2


def generate_corners_array(R_threshold):
    # TESTED
    # creates array of pixels cords whose eigenvalues satisfied the threshold
    corners_after_threshold = np.argwhere(R_threshold)
    corners_after_threshold = np.flip(corners_after_threshold, axis=1)
    return corners_after_threshold


def harris_corner_detector(im):
    # TESTED
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    # print("shape of im %d",im.shape)
    im_dx = compute_image_dx(im)
    im_dy = compute_image_dy(im)
    im_dx_square = sol4_utils.blur_spatial(np.multiply(im_dx, im_dx), 3)
    im_dy_square = sol4_utils.blur_spatial(np.multiply(im_dy, im_dy), 3)
    im_dx_dy = sol4_utils.blur_spatial(np.multiply(im_dx, im_dy), 3)
    im_dy_dx = sol4_utils.blur_spatial(np.multiply(im_dy, im_dx), 3)

    R = compute_corner_response(im_dx_square, im_dy_square, im_dx_dy, im_dy_dx)
    R_threshold = non_maximum_suppression(R)
    corners = generate_corners_array(R_threshold)
    # print("shape of corners %d",corners.shape)
    return corners


def normalize_single_patch(patch):
    # TESTED
    mu = np.mean(patch)
    norm = np.linalg.norm(patch - mu)
    if norm == 0:
        return np.zeros(patch.shape)
    return (patch - mu)/norm


def normalize_patches(patches):
    # TESTED
    return np.apply_along_axis(normalize_single_patch, 1, patches)


def create_window(axis, dim, array):
    # TESTED
    return np.apply_along_axis(lambda x: np.full(dim, x), axis, array)


def sample_descriptor(im, corners, desc_rad):
    # TESTED
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param corners: An array with shape (N,2), where corners[i,:] are the [x,y] coordinates of the ith corner points.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = 2*desc_rad + 1
    N = corners.shape[0]
    corners_gauss3 = 2**(-2) * corners
    possible_window_cords = np.outer(np.full(N, K), np.linspace(0, 1, K))
    tmp = corners_gauss3 - desc_rad
    single_row = possible_window_cords + tmp[:, 1][:, np.newaxis]
    single_col = possible_window_cords + tmp[:, 0][:, np.newaxis]
    patches = map_coordinates(im, [create_window(2, K, single_row.reshape((N, K, 1))),
                                   create_window(1, (K, K), single_col)], order=1, prefilter=False)
    return normalize_patches(patches)


def compute_match_score(Dij, Di1k):
    return np.einsum('ij, kj->ik', Dij, Di1k)


def compute_desc1_desc2_mult(desc1, desc2, N1, N2):
    K = desc1.shape[1]
    desc1_flat = np.reshape(desc1, (N1, K ** 2))
    desc2_flat = np.reshape(desc2, (N2, K ** 2))
    # now we can compute dot product
    return compute_match_score(desc1_flat,
                               desc2_flat)


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
    """
    corners = spread_out_corners(pyr[0], 7, 7, 9)
    desc = sample_descriptor(pyr[2], corners, 3)
    return corners, desc


def filter_points(desc1_2_best_match, desc2_2_best_match, min_score,
                  S_jk_matrix):
    # finds matches in S_jk_matrix . by matches i mean indices of S_jk_matrix
    # for which all 3 conditions hold [ 1) S[i,j] >= min_score 2) S[i,j]
    # >= 2nd best match for Di+1;  3) S[i,j] >= 2nd best match for Di]
    matches = np.argwhere(
        np.logical_and(np.logical_and(S_jk_matrix >= desc1_2_best_match,
                                      S_jk_matrix >= desc2_2_best_match),
                       S_jk_matrix >= min_score))

    return matches[:, 1], matches[:, 0]


def match_features(desc1, desc2, min_score):
    # TESTED
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    N1, N2 = desc1.shape[0], desc2.shape[0]
    # it's a matrix where S_jk_matrix[j][k] = Di,j *Di+1,k
    S_jk_matrix = compute_desc1_desc2_mult(desc1, desc2, N1, N2)
    # it's the second best match in desc2 to desc1. we reshape so
    # the function argwhere could broadcast then together
    desc1_2_best_match = (np.partition(S_jk_matrix, -2, axis=0)[-2]). \
        reshape((1, N2))
    # it's the second best match in desc1 to desc2. we reshape so
    #  the function argwhere could broadcast then together
    desc2_2_best_match = (
    np.partition(S_jk_matrix.transpose(), -2, axis=0)[-2]). \
        reshape((N1, 1))
    return filter_points(desc1_2_best_match, desc2_2_best_match, min_score,
                         S_jk_matrix)


def apply_homography(pos1, H12):
    # TESTED
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    pos1_homo = np.hstack(
        (pos1, np.ones((pos1.shape[0], 1), dtype=pos1.dtype)))
    pos1_homo_mult = np.apply_along_axis(lambda x: np.dot(H12, x.T), 1,
                                         pos1_homo)
    ans = np.apply_along_axis(lambda x: (x / x[-1])[:2], 1, pos1_homo_mult)
    return ans


def ransac_homography(points1, points2, num_iter, inlier_tol,
                      translation_only=False):
    # TESTED
    """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    max_inliers_set = []
    i = 0
    samples_num = 2
    while i != num_iter:
        # we assume that we need only 2 points
        if translation_only:
            samples_num = 1
        sample_points_idx = np.random.permutation(points1.shape[0])[
                            :samples_num]
        points1_sample = points1[sample_points_idx, :]
        points2_sample = points2[sample_points_idx, :]
        H12 = estimate_rigid_transform(points1_sample, points2_sample)
        points2_tag = apply_homography(points1, H12)
        Ej = np.linalg.norm(points2_tag - points2, axis=1) ** 2
        thresheld = np.argwhere(Ej < inlier_tol)
        inliers = thresheld.reshape(thresheld.shape[0])
        if len(max_inliers_set) < len(inliers):
            max_inliers_set = inliers
        i += 1
    H12 = estimate_rigid_transform(points1[max_inliers_set],
                                   points2[max_inliers_set])
    return H12, max_inliers_set





def run_the_shit():
    # get_range(np.array([1, 4, 8]), 3)
    im1 = sol4_utils.read_image(
        "/Users/home/PycharmProjects/imp4/external/oxford1.jpg",
        sol4_utils.GRAYSCALE)
    # print(sample_descriptor(im1, np.array([[0, 0], [42, 8], [37, 44], [100, 200], [11,91]]), 3))
    # homography= np.array([ [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # print(warp_channel(im1, homography))
    im2 = sol4_utils.read_image(
        "/Users/home/PycharmProjects/imp4/external/oxford2.jpg",
        sol4_utils.GRAYSCALE)
    pyr1 = sol4_utils.build_gaussian_pyramid(im1, 3, 5)[0]
    pyr2 = sol4_utils.build_gaussian_pyramid(im2, 3, 5)[0]
    cons1, desc1 = find_features(pyr1)
    cons2, desc2 = find_features(pyr2)
    pos1, pos2 = match_features(desc1, desc2, 0.5)
    H12, inliers = ransac_homography(cons1[pos2,:], cons2[pos1,:], 97, 100)
    display_matches(im1, im2, cons1[pos2], cons2[pos1], inliers)


def display_matches(im1, im2, pos1, pos2, inliers):
    # TESTED
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    im = np.hstack((im1, im2))
    plt.imshow(im, 'gray')
    offset = im1.shape[1]
    for i in range(pos1.shape[0]):
        x1 = pos1[i][0]
        x2 = offset + pos2[i][0]
        y1 = pos1[i][1]
        y2 = pos2[i][1]
        plt.plot([x1, x2], [y1, y2], mfc='r', c='b', lw=.2, ms=4, marker='o')
    for i in range(inliers.shape[0]):
        x1 = pos1[inliers[i]][0]
        x2 = offset + pos2[inliers[i]][0]
        y1 = pos1[inliers[i]][1]
        y2 = pos2[inliers][i][1]
        plt.plot([x1, x2], [y1, y2], mfc='r', c='y', lw=.5, ms=4, marker='o')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    H_succesive_l = len(H_succesive)
    H2m = [0] * (H_succesive_l + 1)
    H2m[m] = np.eye(3)
    for i in range(m):
        H_i = H2m[i + 1] * H_succesive[i]
        H2m[i] = H_i / (H_i[2, 2])
    for i in range(m + 1, H_succesive_l + 1):
        H_i = H2m[i - 1] * np.linalg.inv(H_succesive[i - 1])
        H2m[i] = H_i / (H_i[2, 2])
    return H2m


def compute_bounding_box(homography, w, h):
    # TESTED
    """ computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """

    new_corners = np.sort((apply_homography(
        np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]]),
        homography)).astype(np.int64), axis=0)
    return np.array([new_corners[0], new_corners[-1]])


def warp_channel(image, homography):
    # TESTED
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    tl_corner, br_corner = compute_bounding_box(homography, image.shape[1],
                                                image.shape[0])
    x_min_max_range = np.arange(tl_corner[0], br_corner[0]+1)
    y_min_max_range = np.arange(tl_corner[1], br_corner[1]+1)
    x_i, y_i = np.meshgrid(x_min_max_range, y_min_max_range)
    x_y = np.stack((x_i, y_i), axis=-1)
    x_y_reshaped = x_y.reshape((x_y.shape[0]*x_y.shape[1], 2))
    # denoted Xi and Yi should be transformed by the inverse homography H ̄−1
    # using apply_homography back
    x_y_frame_i = apply_homography(x_y_reshaped, np.linalg.inv(homography)).T
    # These back-warped coordinates can now be used to interpolate the image
    # with map_coordinates.
    tmp = np.flip(x_y_frame_i, 0).reshape((2, x_y.shape[0], x_y.shape[1]))
    return map_coordinates(image, tmp, order=1, prefilter=False)


def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack(
        [warp_channel(image[..., channel], homography) for channel in
         range(3)])



def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images, bonus=False):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.bonus = bonus
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]

  def generate_panoramic_images(self, number_of_panoramas):
      """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
      if self.bonus:
        self.generate_panoramic_images_bonus(number_of_panoramas)
      else:
        self.generate_panoramic_images_normal(number_of_panoramas)

  def generate_panoramic_images_normal(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

  def generate_panoramic_images_bonus(self, number_of_panoramas):
    """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
    pass

  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_the_shit()
    # homography = [[0.0002, 0.034, 0.1219], [0.783, 0.3555, 0.90],
    #               [0.567, 0.89, 0.742]]
    # print(compute_bounding_box(homography, 600, 400))
    # want the second argument as a column vector (or a transposed row)
    # see on some points of the grid:

    # x = np.arange(121)
    # x = np.resize(x, (11,11))
    # print(x)
    # x= np.pad(x, (3,3))
    # print(x)
    # points = np.array([[0,0], [2,3], [10, 78]])
    # points_tag = np.array([[5,5], [2, 4], [4,6], [9,8], [1, 3], [0,0], [90, 2]])

    # T = np.array([np.array(x[p[0]:p[0]+7, p[1]:p[1]+7]) for p in points])

    # plt.imshow(im2, cmap="gray")
    # plt.plot(cons2, mfc='r', c='b', lw=.1, ms=4, marker='o')
    # plt.show()
    # cons2, desc2 = find_features(pyr2)
    # pos1, pos2 = match_features(desc1, desc2, 0.5)
    # print(match_features(desc1, desc2, 0.5))
    # match_1, match_2 = cons1[pos1], cons2[pos2]

    # M = np.array([ [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    #                [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
    #                [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]])
    #
    # N = np.array([ [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
    #                [[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
    #                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    #                [[6, 6,6, 6], [6, 6,6, 6], [6, 6,6, 6], [6, 6,6, 6]]])

    # print(match_features(N, M, 0.05))
    # z = np.zeros((3, 1))
    # b = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
    # # print(b)
    # H12 = np.array([[1, 1, 1],
    #                 [2, 1, 2],
    #                 [31, 9,5]])
    # dist = np.linalg.norm(points-points_tag, axis=1)
    # print(ransac_homography(points, points_tag, 20, 3))
    # print(apply_homography(points, H12))
    # print(np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b)))
    # print(apply_h(points, H12))
    # print(np.dot(H12,np.array([0,0, 1])))
    # print(match_features_my(N, M, 0.05))

    # M = M.reshape((3, 16))
    # print(M)
    # N = N.reshape((4, 16))
    # print(N)
    # all_scores = np.einsum('ij, kj->ik', M, N)
    # #
    # print(all_scores)
    # print(np.argpartition(all_scores.transpose(), -2, axis=-1))
    # N = np.array([9,9, -1])
    # print(N*M)
    # print(harris_corner_detector(im))
    # # print(np.linalg.det(M))
    # corners = sample_descriptor_my(im, harris_corner_detector(im), 3)
    # print(corners.shape)
    # print("///////////////////////////////////////////////////////////////")
    # flat_desc1 = np.reshape(corners, (-1, corners.shape[-1]))
    # print(flat_desc1.shape)
    # print(sample_descriptor(im, harris_corner_detector(im), 3))
    # print(im.shape)
    # pos_in_gaussian = scipy.ndimage.map_coordinates(im,
    #                                                 np.array([[0.5, 2], [1, 2], [2, 3], [1,3]]).T,
    #                                                 order=1, prefilter=False)
    # print(pos_in_gaussian)
    # print(sample_descriptor_my(im, corners, 7))
    # dx = np.array([[1, 2],[3, 4], [5, 6]])
    # dy = np.array([[10, 20],[30, 40], [50, 60]])
    # dx_dy = np.multiply(dx, dy )
    # print(compute_corner_response(create_m(dx, dy, dx_dy)))
    # print(compute_corner_response(m))
    # iml = compute_image_dy(im)
    # plt.imshow(iml, cmap="gray")
    # plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
