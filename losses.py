import torch
import torch.nn as nn
import sys

class PerceptualLoss:
    """
    This is a class for defining the perceptual loss between two images. The perceptual loss is the distance between two images that have been passed throug a pre-trained deep convolutional neural network until a defined deep layer which is trained to extract complex features.
    The idea behind this is that images that are sufficiently similar will activate the same.
    There are two types of perceptual loss:
    1) content loss which measures the L2 between the activated images
    2) pix2pix loss which measures the L1 between the activated images

    Parameters
    ----------
    loss : torch.nn.Module or torch.nn.Functional
        The distance function for the distance between the activated images.
    slic_pth : str
        The path to the Slic module which contains the architecture for the pretrained network.
    model_pth : str
        The path to the weights of the pretrained model.
    """
    def __init__(self, loss, slic_pth, model_pth, layer):
        self.criterion = loss
        self.slic = slic_pth
        self.model = model_pth
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.content = self.content_func(layer)

    def content_func(self, layer):
        '''
        This is a function that turns the pretrained deep convolutional neural network into a function to be applied to our images.
        We stop at the layer before the second to last maxpool with the assumption that this layer has learned complex representations about the Sun.

        Returns
        -------
        model : torch.nn.Sequential
            The pretrained network loaded with the weights up to a certain cutoff layer.
        '''
        sys.path.append(self.slic)
        from model import SolarClassifier

        conv_layer = layer
        cnn = SolarClassifier().to(self.device)
        cnn.load_state_dict(torch.load(self.model,map_location=self.device))
        cnn = nn.Sequential(cnn.layer1,cnn.layer2,cnn.max_pool,cnn.layer3,cnn.layer4,cnn.max_pool,cnn.layer5,cnn.layer6,cnn.max_pool,cnn.layer7,cnn.layer8,cnn.max_pool,cnn.layer8,cnn.layer8,cnn.max_pool)
        model = nn.Sequential().to(self.device)
        for j, layer in enumerate(list(cnn)):
            model.add_module(str(j),layer)
            if j == conv_layer:
                break

        return model

    def get_loss(self,fake_im,real_im):
        '''
        This is the function that works out the perceptual loss between two images.
        We pass both images through our pretrained network to see how close they are to each other.

        Parameters
        ----------
        fake_im : torch.FloatTensor
            The image with bad seeing.
        real_im : torch.FloatTensor
            The image with good seeing.

        Returns
        -------
        loss : torch.FloatTensor
            The perceptual loss between the two images.
        '''

        f_fake = self.content.forward(fake_im)
        f_real = self.content.forward(real_im)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake,f_real_no_grad)
        return loss