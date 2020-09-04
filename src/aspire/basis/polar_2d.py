import logging
import numpy as np

from aspire.basis import Basis
from aspire.image import Image
from aspire.nufft import anufft, nufft
from aspire.utils import ensure
from aspire.utils.matlab_compat import m_reshape
from aspire.utils.matrix import roll_dim, unroll_dim
from aspire.utils.misc import real_type, complex_type

logger = logging.getLogger(__name__)


class PolarBasis2D(Basis):
    """
    Define a derived class for polar Fourier representation for 2D images
    """

    def __init__(self, size, nrad=None, ntheta=None):
        """
        Initialize an object for the 2D polar Fourier grid class

        :param size: The shape of the vectors for which to define the grid.
            Currently only square images are supported.
        :param nrad: The number of points in the radial dimension.
        :param ntheta: The number of points in the angular dimension.
        """

        ndim = len(size)
        ensure(ndim == 2, 'Only two-dimensional grids are supported.')
        ensure(len(set(size)) == 1, 'Only square domains are supported.')

        self.nrad = nrad
        if nrad is None:
            self.nrad = self.nres // 2

        self.ntheta = ntheta
        if ntheta is None:
            # try to use the same number as Fast FB basis
            self.ntheta = 8 * self.nrad

        super().__init__(size)

    def _build(self):
        """
        Build the internal data structure to 2D polar Fourier grid
        """
        logger.info('Represent 2D image in a polar Fourier grid')

        self.count = self.nrad * self.ntheta
        self._sz_prod = self.sz[0] * self.sz[1]

        # precompute the basis functions in 2D grids
        self.freqs = self._precomp()

    def _precomp(self):
        """
        Precomute the polar Fourier grid
        """
        omega0 = 2 * np.pi / (2 * self.nrad - 1)
        dtheta = 2 * np.pi / self.ntheta

        # only need half size of ntheta
        freqs = np.zeros((2, self.nrad * self.ntheta // 2))
        for i in range(self.ntheta // 2):
            freqs[0, i * self.nrad: (i + 1) * self.nrad] = np.arange(self.nrad) * np.sin(i * dtheta)
            freqs[1, i * self.nrad: (i + 1) * self.nrad] = np.arange(self.nrad) * np.cos(i * dtheta)

        freqs *= omega0
        return freqs

    def evaluate(self, v):
        """
        Evaluate coefficients in standard 2D coordinate basis from those in polar Fourier basis

        :param v: A coefficient vector (or an array of coefficient vectors)
            in polar Fourier basis to be evaluated. The first dimension must equal to
            `self.count`.
        :return x: Image instance in standard 2D coordinate basis with \
        resolution of `self.sz`.
        """
        if self.dtype != real_type(v.dtype):
            logger.error(f'Input data type, {v.dtype}, is not consistent with'
                         f' the defined in the class.')

        v, sz_roll = unroll_dim(v, 2)
        nimgs = v.shape[1]

        half_size = self.ntheta // 2

        v = m_reshape(v, (self.nrad, self.ntheta, nimgs))

        v = (v[:, :half_size, :]
             + v[:, half_size:, :].conj())

        v = m_reshape(v, (self.nrad*half_size, nimgs))
        x = np.empty((nimgs, self.sz[0], self.sz[1]), dtype=self.dtype)
        # TODO: need to include the implementation of the many framework in Finufft.
        for isample in range(0, nimgs):
            x[isample, ... ] = np.real(anufft(v[:, isample], self.freqs, self.sz))

        return Image(x)

    def evaluate_t(self, x):
        """
        Evaluate coefficient in polar Fourier grid from those in standard 2D coordinate basis

        :param x: The Image instance representing coefficient array in the \
        standard 2D coordinate basis to be evaluated.
        :return v: The evaluation of the coefficient array `v` in the polar \
        Fourier grid. This is an array of vectors whose first dimension \
        corresponds to x.n_images, and last dimension equals `self.count`.
        """

        assert isinstance(x, Image)

        if self.dtype != x.dtype:
            logger.error(f' Input data type, {x.dtype}, is not consistent with'
                         f' the defined in the class.')

        nimgs = x.n_images

        half_size = self.ntheta // 2

        # get consistent complex type from the real type of x
        out_type = complex_type(x.dtype)
        pf = np.empty((nimgs, self.nrad * half_size), dtype=out_type)
        # TODO: need to include the implementation of the many framework in Finufft.
        for isample in range(0, nimgs):
            pf[isample] = nufft(x[isample], self.freqs)

        pf = np.reshape(pf, (nimgs, self.nrad, half_size))
        v = np.concatenate((pf, pf.conj()), axis=1)

        # return v coefficients with the last dimension size of self.count
        v = v.reshape(nimgs, -1)
        return v
