import numpy as np
from pynufft import NUFFT, helper

from aspire.nufft import Plan


class pynufftPlan(Plan):
    def __init__(self, sz, fourier_pts, epsilon=None, ntransforms=1, **kwargs):
        self.ntransforms = ntransforms

        self.sz = sz
        self.dim = len(sz)

        self.dtype = fourier_pts.dtype

        self.fourier_pts = np.ascontiguousarray(
            np.mod(fourier_pts + np.pi, 2 * np.pi) - np.pi
        )

        self.num_pts = fourier_pts.shape[1]

        # epsilon and Jd?
        Jd = (6, 6)
        Kd = 2 * np.array(self.sz)

        self.NUFFT = NUFFT(helper.device_list()[0])
        self.plan = self.NUFFT(self.fourier_pts, self.sz, Kd, Jd)
        breakpoint()

    def transform(self, signal):
        breakpoint()
        return self.plan.forward(signal)

    def adjoint(self, signal):
        breakpoint()
        return self.plan.adjoint(signal)
