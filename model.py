import numpy as np
from scipy.signal import convolve2d
import scipy.fftpack as sf
from specutils.utils.wcs_utils import vac_to_air

class SeeingAperture:
    """
    This is the class that generates the effective aperture for a given Fried parameter and can also populate it using the model assuming that the Earth's atmosphere can be modelled as a medium with smoothly varying turbulence.
    """

    def __init__(self, wavel, r_0, px_scale, air=False):
        if air:
            self.wavel = wavel
        else:
            self.wavel = vac_to_air(wavel)

        self.r_0 = r_0
        self.px_scale = px_scale
        self.m_scale = 1852 / 60 #this comes from the accepted size of a nautical mile

    def aperture_init(self):
        """
        This class method works out the size in pixels of the effective aperture and thus the size in which to make the point-spread function to convolve with the image.
        """

        self.resolution = (0.98 * self.wavel / self.r_0) * 206265 #this calculates the angular resolution of the effective aperture in arcseconds
        self.diameter = int(self.resolution / self.px_scale) #this calculates the number of detector pixels the point-spread function takes up
        self.fov = np.zeros((int(self.diameter), int(self.diameter))) #generates a blank point-spread function sized array to be filled with the seeing model values

        return self

    def structure_function(self, coord):
        """
        This class method works out the structure function for a particular location within the point-spread function.

        Parameters
        ----------
        coord : numpy.ndarray
            The coordinate in pixel number to calculate the structure function for.
        """
        dist = np.linalg.norm(coord, axis=0) * self.px_scale * self.m_scale #this calculates the distance from the origin of the point-spread function in metres in the sky
        return (6.88 * (dist / self.r_0)**(5/3))

    def fill_aperture(self):
        """
        This class method fills the aperture using the seeing model.
        """
        x_range = np.linspace(-self.diameter/2, self.diameter/2, num=int(self.diameter))
        y_range = np.linspace(-self.diameter/2, self.diameter/2, num=int(self.diameter))
        mg = np.meshgrid(x_range, y_range)
        self.fov = np.exp(-0.5 * self.structure_function(mg) / self.structure_function(mg).sum())
        if self.diameter % 2 != 0:
            self.fov[self.diameter//2, self.diameter//2] = 1
        self.fov /= self.fov.sum() #we are convolving with a kernel which will be the normalised seeing aperture since the intensity is not lost but redistributed over the kernel which we achieve by the normalisation of the kernel

        return self

def gaussian_noise(img_shape):
    """
    This function will generate the Gaussian noise in image space where every pixel in the image will have a different realisation of Gaussian noise. This can be hypothetically done for every separate set of observations too since turbulence is random.

    Parameters
    ----------
    img_shape : list or tuple
        The shape of the image to generate the random Gaussian noise for.
    """

    gd = np.random.normal(scale=np.sqrt(2)/2, size=(*img_shape,2)).view(np.complex128)
    return np.absolute(sf.ifft2(gd)).squeeze()/np.absolute(sf.ifft2(gd)).squeeze().sum()

def synth_seeing(img, aper, gn):
    """
    This function will apply the synthetic seeing aperture to the diffraction-limited images using the equation:

    .. math::
       S = I \\ast P + N
    """

    return convolve2d(img, aper, mode="same", boundary="wrap") + gn

def segmentation(img, n):
    '''
    This is a preprocessing function that will segment the images into segments with dimensions n x n.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be segmented.
    n : int
        The dimension of the segments.

    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image.
    '''

    N = img.shape[0] // n #the number of whole segments in the y-axis
    M = img.shape[1] // n #the number of whole segments in the x-axis

    ####
    # there are 4 cases
    #+------------+------------+------------+
    #| *n         | y segments | x segments |
    #+------------+------------+------------+
    #| N !=, M != | N+1        | M+1        |
    #+------------+------------+------------+
    #| N !=, M =  | N+1        | M          |
    #+------------+------------+------------+
    #| N =, M !=  | N          | M+1        |
    #+------------+------------+------------+
    #| N =, M =   | N          | M          |
    #+------------+------------+------------+
    ####
    if N*n != img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N+1, M+1, n, n), dtype=np.float32)
    elif N*n != img.shape[0] and M*n == img.shape[1]:
        segments = np.zeros((N+1, M, n, n), dtype=np.float32)
    elif N*n == img.shape[0] and M*n != img.shape[1]:
        segments = np.zeros((N, M+1, n, n), dtype=np.float32)
    else:
        segments = np.zeros((N, M, n, n), dtype=np.float32)

    y_range = range(segments.shape[0])
    x_range = range(segments.shape[1])

    for j in y_range:
        for i in x_range:
            if i != x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,i*n:(i+1)*n]
            elif i == x_range[-1] and j != y_range[-1]:
                segments[j, i] = img[j*n:(j+1)*n,-n:]
            elif i != x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,i*n:(i+1)*n]
            elif i == x_range[-1] and j == y_range[-1]:
                segments[j, i] = img[-n:,-n:]

    segments = np.reshape(segments, newshape=((segments.shape[0]*segments.shape[1]), n, n))

    return segments

def segment_cube(img_cube, n):
    '''
    A function to segment a three-dimensional datacube.

    Parameters
    ----------
    img_cube : numpy.ndarray
        The image cube to be segmented.
    n : int
        The dimension of the segments.

    Returns
    -------
    segments : numpy.ndarray
        A numpy array of the segments of the image cube.
    '''

    for j, img in enumerate(tqdm(img_cube, desc="Segmenting image cube: ")):
        if j == 0:
            segments = segmentation(img, n=n)
            #we expand the segments arrays to be four-dimensional where one dimension will be the image positiion within the cube so it will be (lambda point, segments axis, y, x)
            segments = np.expand_dims(segments, axis=0)
        else:
            tmp_s = segmentation(img, n=n)
            tmp_s = np.expand_dims(tmp_s, axis=0)
            #we then add each subsequent segmented image along the wavelength axis
            segments = np.append(segments, tmp_s, axis=0)
    segments = np.swapaxes(segments, 0, 1) #this puts the segment dimension first, wavelength second to make it easier for data loaders

    return segments

def find_corners(img, rows=False, reverse=False):
    '''
    This is a function to find the corners of a CRISP observation since the images are rotated in the image plane.

    Parameters
    ----------
    img : numpy.ndarray
        The image to be rotated. Since CRISP observations usually have multiple images, all of the images will be rotated the same amount in the image plane for a single observation so the corners only need to be found for one -- this is usually taken as the core wavelength (for a waveband measurement).
    rows : bool, optional
        Whether or not to search along the rows. Default searches down the columns.
    reverse : bool, optional
        Determines which direction the search takes place. This also depends on the rows parameter. Default searches from top to bottom if rows is False and left to right if rows is True.

    Returns
    -------
    corner : numpy.ndarray
        An array containing the image plane coordinates of the corner that is to be found. The coordinate is returned in (y,x) format.

    Since CRISP images are rectangular, only 3 of 4 corners are required to be able to obtain the whole image.

    If rows and reverse are both False, then the algorithm finds the top-left corner. If rows is True but reverse is False, the algorithm finds the top-right corner. If rows is False and reverse is True, the algorithm find the bottom-right corner. If rows and reverse are both True, then the algorithm finds the bottom-left corner.
    '''

    if reverse and not rows:
        y_range = range(img.shape[0])
        x_range = reversed(range(img.shape[1]))
    elif reverse and rows:
        y_range = reversed(range(img.shape[0]))
        x_range = range(img.shape[1])
    else:
        y_range = range(img.shape[0])
        x_range = range(img.shape[1])

    if not rows:
        for i in x_range:
            for j in y_range:
                if img[j, i] != img[0, 0]:
                    if img[j, i] == img[0, 0] + 1 or img[j, i] == img[0, 0] - 1:
                        pass
                    else:
                        corner = np.array([j, i])
                        return corner
    else:
        for j in y_range:
            for i in x_range:
                if img[j, i] != img[0, 0]:
                    if img[j, i] == img[0, 0] + 1 or img[j, i] == img[0, 0] - 1:
                        pass
                    else:
                        corner = np.array([j, i])
                        return corner

def im_rotate(img_cube):
    '''
    This is a function that will find the corners of the image and rotate it with respect to the x-axis and crop so only the map is left in the array.

    Parameters
    ----------
    img_cube : numpy.ndarray
        The image cube to be rotated.

    Returns
    -------
    img_cube : numpy.ndarray
        The rotated image cube.
    '''

    mid_wvl = img_cube.shape[0] // 2
    #we need to find three corners to be able to rotate and crop properly and two of these need to be the bottom corners
    bl_corner = find_corners(img_cube[mid_wvl], rows=True, reverse=True)
    br_corner = find_corners(img_cube[mid_wvl], reverse=True)
    tr_corner = find_corners(img_cube[mid_wvl], rows=True)
    unit_vec = (br_corner - bl_corner) / np.linalg.norm(br_corner - bl_corner)
    angle = np.arccos(np.vdot(unit_vec, np.array([0, 1]))) #finds the angle between the image edge and the x-axis
    angle_d = np.rad2deg(angle)

    #find_corners function finds corners in the frame where the origin is the natural origin of the image i.e. (y, x) = (0, 0). However, the rotation is done with respect to the centre of the image so we must change the corner coordinates to the frame where the origin of the image is the centre of the image. Furthermore, as we will be performing an affine transformation on the corner coordinates to obtain our crop ranges we need to add a faux z-axis for the rotation to occur around e.g. add a third dimension. Also as the rotation requires interpolation, the corners are not easily identifiable after the rotation by the find_corners method so find their transformation rotation directly is the best way to do it
    bl_corner = np.array([bl_corner[1]-(img_cube.shape[-1]//2), bl_corner[0] - (img_cube.shape[-2]//2), 1])
    br_corner = np.array([br_corner[1]-(img_cube.shape[-1]//2), br_corner[0] - (img_cube.shape[-2]//2), 1])
    tr_corner = np.array([tr_corner[1]-(img_cube.shape[-1]//2), tr_corner[0] - (img_cube.shape[-2]//2), 1])
    rot_matrix = np.array([[np.cos(-angle), np.sin(-angle), img_cube.shape[-1]//2], [-np.sin(-angle), np.cos(-angle), img_cube.shape[-2]//2], [0,0,1]])
    new_bl_corner = np.rint(np.matmul(rot_matrix, bl_corner))
    new_br_corner = np.rint(np.matmul(rot_matrix, br_corner))
    new_tr_corner = np.rint(np.matmul(rot_matrix, tr_corner))

    for j in range(img_cube.shape[0]):
        img_cube[j] = rotate(img_cube[j], -angle_d, reshape=False, output=np.int16, cval=img_cube[j,0,0])
    img_cube = img_cube[:, int(new_tr_corner[1]):int(new_bl_corner[1]), int(new_bl_corner[0]):int(new_br_corner[0])]

    return img_cube

def add_segmentc(segments_1, segments_2):
    '''
    A function to add two groups of segments together.

    Parameters
    ----------
    segments_1 : numpy.ndarray
        A numpy array of the segments from one image/image cube.
    segments_2 : numpy.ndarray
        A numpy array of the segments to be added to the first lot of segments.
    '''

    return np.append(segments_1, segments_2, axis=0)
