""" Down/up sample projections.
    converted (and adjusted) from MATLAB module/function "cryo_compare_stacks.m".
"""

import numpy
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2, fftn, ifftn

from aspire.preprocessor.crop import crop
from aspire.common.exceptions import DimensionsIncompatible
from aspire.utils.helpers import TupleCompare, f_flatten


def downsample(img, side, compute_fx=False, stack=False, mask=None):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input image 'img'. If the optional argument
        stack is set to True, then the *first* dimension of 'img' is interpreted as the index of each
        image in the stack. The size argument side is an integer, the size of the
        output images.  Let the size of a stack
        of 2D images 'img' be n1 x n1 x k.  The size of the output will be side x side x k.

        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size. For example for downsampling an
        n0 x n0 image with a 0.9 x nyquist filter, do the following:
            msk = fuzzymask(n,2,.45*n,.05*n)
            out = downsample(img, n, 0, msk)
            The size of the mask must be the size of output. The optional fx output
            argument is the padded or cropped, masked, FT of in, with zero
            frequency at the origin.
    """

    try:
        side = int(side)
    except ValueError:
        raise ValueError("side should be an integer!")

    if not isinstance(stack, bool):
        raise TypeError("stack should be a bool! set it to either True/False.")

    if mask is not None and mask.shape != img.shape:
        raise DimensionsIncompatible(f'Dimensions incompatible! mask shape={mask.shape}, img shape={img.shape}.')

    ndim = sum([True for i in img.shape if i > 1])  # number of non-singleton dimensions
    if ndim not in [1, 2, 3]:
        raise DimensionsIncompatible(f"Can't downsample image with {ndim} dimensions!")

    if ndim == 1:
        szout = (1, side)  # this is the shape of the final vector
    elif ndim == 2 or ndim == 3 and stack:
        szout = (side, side)  # this is the shape of the final mat
    else:  # ndim == 3 and not stack
        szout = numpy.array([side, side, side])  # this is the shape of the final cube

    if ndim == 1:
        # force input img into row vector with the shape (1, img.size)
        img = numpy.asmatrix(f_flatten(img))

    # check sizes of input and output
    szin = img[0, :, :].shape if stack else img.shape

    if TupleCompare.eq(szout, szin):  # no change in shape
        if not compute_fx:
            return img

    # adjust mask to be the size of desired output
    mask = crop(mask, side) if mask else 1

    if ndim == 1:
        # return a vector scaled from the original vector
        x = fftshift(fft(img))
        fx = crop(x, side) * mask
        out = ifft(ifftshift(fx), axis=0) * (numpy.prod(szout) / numpy.prod(szin))

    elif ndim == 2:
        # return a 2D image scaled from the original image
        fx = crop(fftshift(fft2(img)), side) * mask
        out = ifft2(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

    elif ndim == 3 and stack:
        # return a stack of 2D images where each one of them is downsampled
        num_images = img.shape[0]
        out = numpy.zeros([num_images, side, side], dtype=complex)
        for i in range(num_images):
            fx = crop(fftshift(fft2(img[i, :, :])), side) * mask
            out[i, :, :] = ifft2(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

    else:  # ndim == 3 and not stack
        # return a 3D object scaled from the input 3D cube
        fx = crop(fftshift(fftn(img)), side) * mask
        out = ifftn(ifftshift(fx)) * (numpy.prod(szout) / numpy.prod(szin))

    if numpy.all(numpy.isreal(img)):
        out = numpy.real(out)

    if compute_fx:
        fx = numpy.fft.ifftshift(fx)
        return out, fx

    return out.astype('float32')