# A Machine Learning Approach to Correcting Atmospheric Seeing in Solar Flare Observations

The following is the repository of the Seeing AUtoeNcoder (Shaun) which is a trained fully-convolutional autoencoder with the purpose of correcting for atmospheric seeing in ground-based solar observations. Atmospheric seeing is a problem in all of ground-based astronomy and affects the resolution of observations by refracting incoming light slightly due to inhomogeneities in the density and temperature structure of the Earth's atmosphere. Adaptive optics systems are used in all modern facilities but when imaging an extended object (e.g. a layer of the Sun's atmosphere), the adaptive optics cannot correct the wavefronts over the whole field-of-view. As a result, post-processing techniques are required to correct for seeing with robust statistical methods e.g. [MOMFBD](https://link.springer.com/article/10.1007%2Fs11207-005-5782-z) or Speckle interferometry developed for this purpose.

However, for solar flare observations these post-processing techniques are not the best-suited:

1. MOMFBD requires a wideband context image to aid in its restoration of features which we propose is actually detrimental to solar flare reconstructions since strong spectral line enhancements do not always have a corresponding continuum enhancement leading to the features appearing different when imaged in the line as opposed to the continuum.

2. Speckle interferometry requires $\mathcal{O}(10^{2})$ raw frames to get a good estimate of the atmospheric parameters to perform a sound reconstruction. Estimates of the atmospheric parameters require a quasi-static solar background which cannot be guaranteed during flares for as many frames due to their dynamics. This means that Speckle reconstructions of flares ideally work using a smaller number of frames which increases the uncertainty in the estimated parameters.

Therefore, we have developed a dedicated flare seeing correction tool based on deep learning. Following [Racine 1996](adsabs.harvard.edu/full/1996PASP..108..699R) we generate synthetic seeing point-spread functions (PSFs) where the PSF is the 0th order Hankel transform of the modulation transfer function (MTF) and the MTF relies on the structure function of the atmosphere:

$$ P_{\mathrm{atmos}}(\rho)=\int_{0}^{\infty} J_{0}(\rho v) \exp \left\{-0.5 D_{S}(\nu)\right\} \nu d \nu $$

where $\nu$ is spatial frequency and $J_{0}(\rho \nu)$ is the zeroth order Bessel function.

We then take imaging spectroscopy data from the Swedish 1-m Solar Telescope's (SST) CRisp Imaging SpectroPolarimeter (CRISP) instrument obtained in H&alpha; and Ca II 8542&#8491; of flares with good seeing and apply bad seeing to them. This forms the basis of our training dataset. The two active regions we use data from are AR12157 with M1.1 flare SOL20140906T17:09 (data avaiable from [F-CHROMA](https://star.pst.qub.ac.uk/wiki/doku.php/public/solarflares/start)), and AR12673 with X2.2 flare SOL20170906T09:10 and X9.3 flare SOL20170906T12:02. The model is then trained to correct for this synthetic seeing and this is then applied to data with no ground truth.

![](gifs/x22.gif)

![](gifs/x93.gif)

The above are exampels of two cases with no ground truth. The top image is during the decay phase of the X2.2 solar flare SOL20170906T09:10 and we show the correction to the red wing and just blue of the line core. The bottom image is the peak of the X9.3 (the most energetic solar flare of Solar Cycle 24) solar flare SOL20170906T12:02 and the far red wing and line core corrections are shown. Each of the field-of-views shown is around $60^{\prime \prime} \times 60^{\prime \prime}$.

## Requirements
For training:

* `Python` 3.6+
* `PyTorch` 1.4+ [here](https://pytorch.org)
* `matplotlib` 3.0+
* `crisPy2` 1.0+ [here](https://github.com/rhero12/crisPy2)
* `specutils` 1.0+ [here](https://specutils.readthedocs.io/en/stable/)
* `hankel` 1.0+ [here](https://hankel.readthedocs.io/en/latest/)

For inference:

* Either:
  - `Python` 3.6+
  - `PyTorch` 1.4+
  - `crisPy2` 1.0+
* or:
  - `C++` (in development)

## Usage
Please see the `example.ipynb` notebook for how to use Shaun in Python.


## Release
Release 1.0.0 comes with trained models for H&alpha; and Ca II 8542&#8491; along with the traced torchscript models for faster inference but less flexibility!

To get this release:
```
git clone https://github.com/rhero12/Shaun
cd Shaun
git checkout tags/v1.0
```

## Publications
"A Machine Learning Approach to Correcting Atmospheric Seeing in Solar Flare Observations", **J. A. Armstrong** and L. Fletcher, MNRAS, *accepted*. [arXiv](https://arxiv.org/abs/2011.12814) [Advanced Article](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/staa3742/6019896?searchresult=1)