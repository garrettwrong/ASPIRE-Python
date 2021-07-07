import logging

import numpy as np

from aspire.reconstruction.kernel import FourierKernel
from aspire.volume import Volume

logger = logging.getLogger(__name__)


class Estimator:
    def __init__(self, src, basis, batch_size=512, preconditioner="circulant"):
        self.src = src
        self.basis = basis
        self.dtype = self.src.dtype
        self.batch_size = batch_size
        self.preconditioner = preconditioner

        self.L = src.L
        self.n = src.n

        if not self.dtype == self.basis.dtype:
            logger.warning(
                f"Inconsistent types in {self.dtype} Estimator."
                f" basis: {self.basis.dtype}"
            )

        if src.L != basis.nres:
            raise ValueError(
                "Currently require 2D source and 3D volume resolution to be the same."
                f" Given src.L={src.L} != {basis.nres}"
            )

        """
        An object representing a 2*L-by-2*L-by-2*L array containing the non-centered Fourier transform of the mean
        least-squares estimator kernel.
        Convolving a volume with this kernel is equal to projecting and backproject-ing that volume in each of the
        projection directions (with the appropriate amplitude multipliers and CTFs) and averaging over the whole
        dataset.
        Note that this is a non-centered Fourier transform, so the zero frequency is found at index 1.
        """

    def __getattr__(self, name):
        """Lazy attributes instantiated on first-access"""

        if name == "kernel":
            logger.info("Computing kernel")
            kernel = self.kernel = self.compute_kernel()
            return kernel

        elif name == "precond_kernel":
            if self.preconditioner == "circulant":
                logger.info("Computing Preconditioner kernel")
                precond_kernel = self.precond_kernel = FourierKernel(
                    1.0 / self.kernel.circularize(), centered=True
                )
            else:
                precond_kernel = self.precond_kernel = None
            return precond_kernel

        else:
            raise AttributeError(name)

    def compute_kernel(self):
        raise NotImplementedError("Subclasses must implement the compute_kernel method")

    def estimate(self, b_coeff=None, tol=None):
        """Return an estimate as a Volume instance."""
        if b_coeff is None:
            b_coeff = self.src_backward()
        est_coeff = self.conj_grad(b_coeff, tol=tol)
        est = np.transpose(self.basis.evaluate(est_coeff), (0, 3, 2, 1))

        return Volume(est)
