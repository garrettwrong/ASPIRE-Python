import mrcfile
import numpy as np
from scipy.fftpack import fft2, ifft2, ifftshift
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import lstsq

from aspire.utils.matrix import anorm
from aspire.utils import ensure
from aspire.utils.coor_trans import grid_2d
from aspire.utils.fft import centered_fft2_C, centered_ifft2_C


# TODO: The implementation of these functions should move directly inside the appropriate Image methods that call them.

def _im_translate2(im, shifts):
    """
    Translate image by shifts
    :param im: An array of size n-by-L-by-L-by-n containing images to be translated.
    :param shifts: An array of size n-by-2 specifying the shifts in pixels.
        Alternatively, it can be a row vector of length 2, in which case the same shifts is applied to each image.
    :return: The images translated by the shifts

    TODO: This implementation has been moved here from aspire.aspire.abinitio and is faster than _im_translate.
    """
    if shifts.ndim == 1:
        shifts = shifts[np.newaxis, :]
    n_im = im.shape[0]
    n_shifts = shifts.shape[0]

    if shifts.shape[1] != 2:
        raise ValueError('Input `shifts` must be of size n-by-2')

    if n_shifts != 1 and n_shifts != n_im:
        raise ValueError('The number of shifts must be 1 or match the number of images')

    if im.shape[1] != im.shape[2]:
        raise ValueError('Images must be square')

    resolution = im.shape[1]
    grid = np.fft.ifftshift(np.ceil(np.arange(-resolution / 2, resolution / 2)))
    om_y, om_x = np.meshgrid(grid, grid)
    phase_shifts = (np.einsum('ij, k -> ijk', om_x, shifts[:,0]) +
                    np.einsum('ij, k -> ijk', om_y, shifts[:,1]))
    # TODO: figure out how why the result of einsum requires reshape
    phase_shifts = phase_shifts.reshape(n_shifts, resolution, resolution)
    phase_shifts /= resolution

    mult_f = np.exp(-2 * np.pi * 1j * phase_shifts)
    im_f = np.fft.fft2(im)
    im_translated_f = im_f * mult_f
    im_translated = np.real(np.fft.ifft2(im_translated_f))

    return im_translated


def normalize_bg(imgs, bg_radius=1.0, do_ramp=True):
    """
    Normalize backgrounds and apply to a stack of images

    :param imgs: A stack of images in L-by-L-by-N array
    :param bg_radius: Radius cutoff to be considered as background (in image size)
    :param do_ramp: When it is `True`, fit a ramping background to the data
            and subtract. Namely perform normalization based on values from each image.
            Otherwise, a constant background level from all images is used.
    :return: The modified images
    """
    L = imgs.shape[0]
    grid = grid_2d(L)
    mask = (grid['r'] > bg_radius)

    if do_ramp:
        # Create matrices and reshape the background mask
        # for fitting a ramping background
        ramp_mask = np.vstack((grid['x'][mask].flatten(),
                           grid['y'][mask].flatten(),
                           np.ones(grid['y'][mask].flatten().size))).T
        ramp_all = np.vstack((grid['x'].flatten(), grid['y'].flatten(),
                          np.ones(L*L))).T
        mask_reshape = mask.reshape((L*L))
        imgs = imgs.reshape((L*L, -1))

        # Fit a ramping background and apply to images
        coeff = lstsq(ramp_mask, imgs[mask_reshape])[0]
        imgs = imgs - ramp_all @ coeff
        imgs = imgs.reshape((L, L, -1))

    # Apply mask images and calculate mean and std values of background
    imgs_masked = (imgs * np.expand_dims(mask, 2))
    denominator = np.sum(mask)
    first_moment = np.sum(imgs_masked, axis=(0, 1))/denominator
    second_moment = np.sum(imgs_masked ** 2, axis=(0, 1))/denominator
    mean = first_moment
    variance = second_moment - mean**2
    std = np.sqrt(variance)

    return (imgs-mean)/std


class Image:
    def __init__(self, data):

        assert isinstance(data, np.ndarray), "Image should be instantiated with an ndarray"

        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        ensure(data.shape[1] == data.shape[2], 'Only square ndarrays are supported.')

        self.data = data
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.n_images = self.shape[0]
        self.res = self.shape[1]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        if isinstance(other, Image):
            other = other.data

        return Image(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Image):
            other = other.data

        return Image(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Image):
            other = other.data

        return Image(self.data * other)

    def sqrt(self):
        return np.sqrt(self.data)

    def __repr__(self):
        return f'{self.n_images} images of size {self.res}x{self.res}'

    def asnumpy(self):
        return self.data

    def copy(self):
        return Image(self.data.copy())

    def shift(self, shifts):
        """
        Translate image by shifts. This method returns a new Image.

        :param shifts: An array of size n-by-2 specifying the shifts in pixels.
            Alternatively, it can be a column vector of length 2, in which case
            the same shifts is applied to each image.
        :return: The Image translated by the shifts, with periodic boundaries.
        """
        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]

        im_translated = self._im_translate(shifts)
        return Image(im_translated)

    def downsample(self, ds_res):
        """
        Downsample Image to a specific resolution. This method returns a new Image.

        :param ds_res: int - new resolution, should be <= the current resolution
            of this Image
        :return: The downsampled Image object.
        """
        grid = grid_2d(self.res)
        grid_ds = grid_2d(ds_res)

        im_ds = np.zeros((self.n_images, ds_res, ds_res), dtype=self.dtype)

        # x, y values corresponding to 'grid'. This is what scipy interpolator needs to function.
        res_by_2 = self.res / 2
        x = y = np.ceil(np.arange(-res_by_2, res_by_2)) / res_by_2

        mask = (np.abs(grid['x']) < ds_res / self.res) & (np.abs(grid['y']) < ds_res / self.res)
        im = np.real(centered_ifft2_C(centered_fft2_C(self.data) *
                                    np.expand_dims(mask, 0)))

        for s in range(im_ds.shape[0]):
            interpolator = RegularGridInterpolator(
                (x, y),
                im[s],
                bounds_error=False,
                fill_value=0
            )
            im_ds[s] = interpolator(np.dstack([grid_ds['x'], grid_ds['y']]))

        return Image(im_ds)

    def filter(self, filter):
        """
        Apply a `Filter` object to the Image and returns a new Image.

        :param filter: An object of type `Filter`.
        :return: A new filtered `Image` object.
        """
        filter_values = filter.evaluate_grid(self.res)

        im_f = centered_fft2_C(self.data)
        if im_f.ndim > filter_values.ndim:
            im_f = np.expand_dims(filter_values, 0) * im_f
        else:
            im_f = filter_values * im_f
        im = centered_ifft2_C(im_f)
        im = np.real(im)

        return Image(im)

    def rotate(self):
        raise NotImplementedError

    def save(self, mrcs_filepath, overwrite=False):
        with mrcfile.new(mrcs_filepath, overwrite=overwrite) as mrc:
            # original input format (the image index first)
            mrc.set_data(self.data.astype('float32'))

    def _im_translate(self, shifts):
        """
        Translate image by shifts
        :param im: An array of size n-by-L-by-L containing images to be translated.
        :param shifts: An array of size n-by-2 specifying the shifts in pixels.
            Alternatively, it can be a row vector of length 2, in which case the same shifts is applied to each image.
        :return: The images translated by the shifts, with periodic boundaries.

        TODO: This implementation is slower than _im_translate2
        """
        im = self.data

        if shifts.ndim == 1:
            shifts = shifts[np.newaxis, :]
        n_shifts = shifts.shape[0]

        ensure(shifts.shape[-1] == 2, "shifts must be nx2")

        ensure(n_shifts == 1 or n_shifts == self.n_images, "number of shifts must be 1 or match the number of images")

        L = self.res
        im_f = fft2(im, axes=(1, 2))
        grid_1d = ifftshift(np.ceil(np.arange(-L/2, L/2))) * 2 * np.pi / L
        om_x, om_y = np.meshgrid(grid_1d, grid_1d, indexing='ij')

        #phase_shifts_x = np.broadcast_to(-shifts[:, 0], (n_shifts, L, L))
        #phase_shifts_y = np.broadcast_to(-shifts[:, 1], (n_shifts, L, L))
        phase_shifts_x = -shifts[:, 0].reshape((n_shifts, 1, 1))
        phase_shifts_y = -shifts[:, 1].reshape((n_shifts, 1, 1))

        phase_shifts = (om_x[np.newaxis, :, :] * phase_shifts_x) + (om_y[np.newaxis, :, :] * phase_shifts_y)

        mult_f = np.exp(-1j * phase_shifts)
        im_translated_f = im_f * mult_f
        im_translated = ifft2(im_translated_f, axes=(1, 2))
        im_translated = np.real(im_translated)

        return im_translated

    def norm(self):
        return anorm(self.data)

    @property
    def size(self):
        # probably not needed, transition
        return np.size(self.data)



class CartesianImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class PolarImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class BispecImage(Image):
    def expand(self, basis):
        return BasisImage(basis)


class BasisImage(Image):
    def __init__(self, basis):
        self.basis = basis

    def evaluate(self):
        return CartesianImage()


class FBBasisImage(BasisImage):
    pass
