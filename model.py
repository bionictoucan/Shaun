import numpy as np
from scipy.signal import convolve2d
from specutils.utils.wcs_utils import vac_to_air
from tqdm import tqdm
import astropy.units as u
from hankel import HankelTransform

class SeeingApertureMTF:
    """
    This is the class that generates the effective aperture for a given Fried parameter and can also populate it using the model assuming that the Earth's atmosphere can be modelled as a medium with smoothly varying turbulence.

    Parameters
    ----------
    wavel : float
        The wavelength of light in Angstroms.
    r0 : float
        The Fried parameter in m.
    pxScale : float
        The size of one detector pixel in arcseconds on the observed object.
    air : bool, optional
        Whether or not the provided wavelength is air wavelength. If not then the wavelength is converted from vacuum value. Default is False.
    """

    def __init__(self, wavel, r0, pxScale, air=False):
        if air:
            self.wavel = wavel * 1e-10
        else:
            self.wavel = vac_to_air(wavel<<u.Angstrom).value * 1e-10

        self.r0 = r0
        self.pxScale = pxScale

        self.resolution = 0.98 * self.wavel / self.r0 * 206265 # resolution of the image after being imaged through seeing
        self.diameter = int(self.resolution / self.pxScale) # diameter of PSF in pixels
        self.psf = np.zeros(shape=(int(self.diameter), int(self.diameter)))
        self.pxm = np.linspace(1.75, 880, int(880/1.75)+1) # 1.75 is the size of one pixel in metres and 880 is the size of 840 pixels in metres and the int(880/1.75)+1 is the number of pixels in the field-of-view of 840 pixels, this is an average field-of-view size in pixels with higher field-of-views not adding much to the terms
        self.modtf = self.mtf(self.wavel, self.r0)
        self.ht = HankelTransform(nu=0, h=0.05, N=62)
        self.psf1d = self.ht.transform(self.modtf, self.pxm, ret_err=False)

        for j in range(self.psf.shape[0]):
            for i in range(self.psf.shape[1]):
                idx = int(np.linalg.norm((j-self.psf.shape[0]//2, i-self.psf.shape[1]//2)))
                self.psf[j, i] = self.psf1d[idx]
                
        self.psf /= self.psf.sum()

    @staticmethod
    def mtf(wavel, r0):
        return lambda x: np.exp(-(6.88 / 2.0) * (wavel * x / (2 * np.pi * r0))**(5 / 3))

def gaussian_noise(img_shape, diameter):
    """
    This function will generate the Gaussian noise in image space where every pixel in the image will have a different realisation of Gaussian noise. This can be hypothetically done for every separate set of observations too since turbulence is random.

    Parameters
    ----------
    img_shape : list or tuple
        The shape of the image to generate the random Gaussian noise for.
    diameter : int
        The diameter of the seeing aperture.

    Returns
    -------
    gn : numpy.ndarray
        The Gaussian noise.
    """

    N = img_shape[0] // diameter
    M = img_shape[1] // diameter

    if N*diameter != img_shape[0] and M*diameter != img_shape[1]:
        gd = np.random.normal(scale=1, size=(N+1, M+1))
    elif N*diameter != img_shape[0] and M*diameter == img_shape[1]:
        gd = np.random.normal(scale=1, size=(N+1,M))
    elif N*diameter == img_shape[0] and M*diameter != img_shape[1]:
        gd = np.random.normal(scale=1, size=(N,M+1))
    else:
        gd = np.random.normal(scale=1, size=(N,M))

    gn = np.zeros(img_shape, dtype=np.float32)
    for j in range(gd.shape[0]):
        for i in range(gd.shape[1]):
            gn[j*diameter:(j+1)*diameter,i*diameter:(i+1)*diameter] = gd[j,i]

    return gn / gn.sum()

def synth_seeing(img, aper, gn):
    """
    This function will apply the synthetic seeing aperture to the diffraction-limited images using the equation:

    .. math::
       S = I \\ast P + N
    """

    return convolve2d(img, aper, mode="same", boundary="wrap") + gn
