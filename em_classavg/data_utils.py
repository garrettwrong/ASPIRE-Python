from scipy.io import loadmat
import numpy as np
import pyfftw
from scipy.special import erf

np.set_string_function(lambda a: str(a.shape), repr=False)

def mat_to_npy(file_name):
    return loadmat(file_name + '.mat')[file_name]


def mat_to_npy_vec(file_name):
    a = mat_to_npy(file_name)
    return a.reshape(a.shape[0] * a.shape[1])


def estimate_snr(images, prewhiten=False):
    # Prewhiten the image if needed.
    if prewhiten:
        raise NotImplementedError

    if len(images.shape) == 2:
        images = images[np.newaxis, :, :]

    images = np.transpose(images, axes=(1, 2, 0))

    n = images.shape[2]

    # Compute radius outside which the pixels are noise
    p = images.shape[1]
    radius_of_mask = np.floor(p / 2.0) - 1.0

    # Compute indices of noise samples
    r = cart2rad(p)
    points_inside_circle = r < radius_of_mask
    num_noise_points = p * p - np.count_nonzero(points_inside_circle)

    var_n = np.sum(np.var(images[~points_inside_circle], axis=0)) * num_noise_points / (num_noise_points * n - 1)
    var_s = np.sum(np.var(images, axis=(0, 1))) * p * p / (images.size - 1) - var_n
    snr = var_s / var_n

    return float(snr), float(var_s), float(var_n)


def cart2rad(n):
    # Compute the radii corresponding of the points of a cartesian grid of size NxN points
    # XXX This is a name for this function.
    n = np.floor(n)
    x, y = image_grid(n)
    r = np.sqrt(np.square(x) + np.square(y))
    return r


def image_grid(n):
    # Return the coordinates of Cartesian points in an NxN grid centered around the origin.
    # The origin of the grid is always in the center, for both odd and even N.
    p = (n - 1.0) / 2.0
    x, y = np.meshgrid(np.linspace(-p, p, n), np.linspace(-p, p, n))
    return x, y


def normalize_background(stack):
    # Normalizes background to mean 0 and std 1.
    #
    # stack = normalize_background(stack)
    #   Estimate the mean and std of each image in the stack using pixels
    #   outside radius r (=half the image size in pixels), and normalize the image such that the
    #   background has mean 0 and std 1. Each image in the stack is corrected
    #   separately.
    #
    # Example:
    # stack2 = normalize_background(stack)
    n_images = len(stack)
    m = np.shape(stack)[1]
    n = np.shape(stack)[2]

    if m != n:
        ValueError('Images in the stack must be square.')

    r = np.floor(n / 2)

    # Find indices of backgruond pixels in the images
    ctr = (n + 1) / 2

    xv, yv = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))

    radii_sq = (xv - ctr) ** 2 + (yv - ctr) ** 2
    background_pixels_mask = (radii_sq > r * r)

    sd_bg = np.zeros(n_images)
    mean_bg = np.zeros(n_images)
    for kk in np.arange(n_images):
        proj = stack[kk]
        background_pixels = proj[background_pixels_mask]

        # Compute mean and standard deviation of background pixels
        mm = np.mean(background_pixels)
        sd = np.std(background_pixels, ddof=1)

        proj = (proj - mm) / sd
        stack[kk] = proj

        sd_bg[kk] = sd
        mean_bg[kk] = mm

    return stack, mean_bg, sd_bg

# TODO:decorator function since 1. Itay's mask function expexts input in matlab style and 2. doesn't support a single image
def mask_decorator(images, is_stack=False, r=None, rise_time=None):

    do_alter = images.ndim == 2 and is_stack == True
    if do_alter:
        images = images[:, :, np.newaxis]

    images_masked = mask(images, is_stack, r, rise_time)

    if do_alter:
        images_masked = images_masked[:, :, 0]

    return images_masked


def mask(images, is_stack=False, r=None, rise_time=None):

    num_dims = images.ndim
    if num_dims < 2 or num_dims > 3:
        pass  # raise error

    if is_stack and num_dims == 2:
        pass  # raise error

    if is_stack:
        num_dims = 2

    shape = images.shape[:num_dims]
    if num_dims == 2:
        if shape[0] != shape[1]:
            pass  # raise error

    if num_dims == 3:
        if shape[0] != shape[1] or shape[0] != shape[2] or shape[1] != shape[2]:
            pass  # raise error

    n = shape[0]
    if r is None:
        r = int(np.floor(0.45 * n))

    if rise_time is None:
        rise_time = int(np.floor(0.05 * n))

    m = fuzzymask(n, num_dims, r, rise_time)
    out = (images.transpose(2, 0, 1) * m).transpose(1, 2, 0)
    return out


def fuzzymask(n, dims, r0, risetime, origin=None):
    if isinstance(n, int):
        n = np.array([n])

    if isinstance(r0, int):
        r0 = np.array([r0])

    center = (n + 1.0) / 2
    k = 1.782 / risetime

    if dims == 1:
        if origin is None:
            origin = center
            origin = origin.astype('int')
        r = np.abs(np.arange(1 - origin[0], n - origin[0] + 1))

    elif dims == 2:
        if origin is None:
            origin = np.floor(n / 2) + 1
            origin = origin.astype('int')
        if len(n) == 1:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1]

        if len(r0) < 2:
            r = np.sqrt(np.square(x) + np.square(y))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]))

    elif dims == 3:
        if origin is None:
            origin = center
            origin = origin.astype('int')
        if len(n) == 1:
            x, y, z = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[0]:n[0] - origin[0] + 1]
        else:
            x, y, z = np.mgrid[1 - origin[0]:n[0] - origin[0] + 1, 1 - origin[1]:n[1] - origin[1] + 1, 1 - origin[2]:n[2] - origin[2] + 1]

        if len(r0) < 3:
            r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        else:
            r = np.sqrt(np.square(x) + np.square(y * r0[0] / r0[1]) + np.square(z * r0[0] / r0[2]))
    else:
        return 0  # raise error

    m = 0.5 * (1 - erf(k * (r - r0[0])))

    return m