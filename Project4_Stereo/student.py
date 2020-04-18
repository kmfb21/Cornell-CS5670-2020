import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # raise NotImplementedError()
    n = len(images)
    row, col, cha = images[0].shape

    i = np.array(images).reshape(n, -1)

    lt = lights.T
    l_inv = np.linalg.inv(np.dot(lt, lights))
    g_matrix = np.dot(np.dot(l_inv, lt), i)
    # print(g_matrix)

    # albedo
    rgb_g = np.reshape(g_matrix.T, (row, col, cha, 3))
    albedo = np.linalg.norm(rgb_g, axis=3)

    # normals
    grey_g = np.mean(rgb_g, axis=2)
    normals = grey_g / np.maximum(1e-7, np.linalg.norm(grey_g, axis=2)[:, :, np.newaxis])
    normals[np.linalg.norm(grey_g, axis=2) < 1e-7] = 0

    return albedo, normals


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # raise NotImplementedError()
    proj_matrix = np.dot(K, Rt)
    height, weight, dim = np.shape(points)
    homo = np.concatenate((points, np.ones((height, weight, 1))), axis=2)
    proj = np.tensordot(homo, proj_matrix.T, axes=1)
    norm_proj = proj / (proj[:, :, 2])[:, :, np.newaxis]
    return norm_proj[:, :, :2]

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    # raise NotImplementedError()
    height, width, channels = image.shape
    ans = np.zeros((height, width, channels * ncc_size ** 2))
    mid = ncc_size // 2

    for row in range(mid, height - mid):
        for col in range(mid, width - mid):
            patches = image[row - mid: row + mid + 1, col - mid: col + mid + 1, :]
            patches = patches - np.mean(patches, axis=(0, 1))
            patches = [patches[:, :, c].flatten().T for c in range(channels)]
            flattened = np.array(patches).flatten(order='C')
            normalized = np.linalg.norm(flattened)
            if normalized < 1e-6:
                flattened.fill(0)
            else:
                flattened = flattened / normalized
            ans[row, col] = flattened
    return ans


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # raise NotImplementedError()
    return np.sum(np.multiply(image1, image2), axis=2)