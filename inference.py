import torch
import numpy as np
from .shaun import Shaun
from crisPy.utils import segmentation, mosaic, segment_cube, mosaic_cube

class Corrector:
    """
    This is the object to correct for seeing in observations.

    Parameters
    ----------
    in_channels : int
        The number of channels of the input images.
    out_channels : int
        The number of channels of the output images.
    nef : int
        The number of base feature maps used in the first convolutional layer.
    model_path : str
        The path to the trained model for the network.
    error : float, optional
        The error on the estimates from the network. Default is None which takes the last training error from the model file.
    """

    def __init__(self, in_channels, out_channels, nef, model_path, error=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Shaun(in_channels=in_channels, out_channels=out_channels, nef=nef).to(self.device)

        if os.path.isfile(model_path):
            print(f"loading model {model_path}")
            ckp = torch.load(model_path)
            self.model.load_state_dict(ckp["model_state_dict"])
            if error is None:
                self.error = ckp["losses"]["train_l"][-1]
            else:
                self.error = error
            print("=> model loaded.")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

        self.model.path()

    def mysticalman(self, img):
        """
        This class method does the correctino on the images.

        Parameters
        ----------
        img : numpy.ndarray
            The image to be corrected by the network.
        """
        with torch.no_grad():
            img_shape = img.shape[-2:]
            if img.ndim == 2:
                img = segmentation(img, n=256)
                img = torch.from_numpy(img).unsqueeze(1).float().to(self.device)
                out = self.model(img).squeeze().cpu().numpy()

                return mosaic(out, img_shape, n=256)
            elif img.ndim == 3:
                img = segment_cube(img, n=256)
                out = np.zeros_like(img)
                for j, im in enumerate(np.rollaxis(img, 1)):
                    im = torch.from_numpy(im).unsqueeze(1).float().to(self.device)
                    out[:,j] = self.model(im).squeeze().cpu().numpy()

                return mosaic_cube(out, img_shape, n=256)

class SpeedyCorrector(Corrector):
    """
    This class is to do the same corrections but using the torchscript model. This is ~4x faster but limits the batch size to 16.

    Parameters
    ----------
    model_path : str
        The path to the torchscript model.
    error : float
        The error to apply to the estimates.
    """

    def __init__(self, model_path, error):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path).to(self.device)
        self.error = error

        self.model.eval()
