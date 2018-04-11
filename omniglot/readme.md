### Poom's ConvNet for One-Shot Learning on the Omniglob Dataset

I follow the approach from [1], which reported a whopping 93.8% accuracy on 20-way one-shot character matching with some slight modifications: 

* I ditched batch normalization.
* rather than downsizing the image, a thin plate spline spatial transformer network is used to crop input images [2]
* the elementwise complex norm of the fourier transform of the image is fed through a parallel network of the same architecture whose prediction is combined with the original one using learned weights to obtain the final prediction.
* kernels of the first convolutional layer is larger to account for the intuition that larger image means larger features
* Munkres assignment algorithm is used to find the maximum probability matching [3]
* elastic distortions, small random rotations and small random shearing is used to augment the dataset

These weird tricks and some more minor ones allow me to achieve a __prediction acccuracy of 98.6%__ on the test set which means an error rate of __smaller than a fourth of [1] using a training set that's 3/4th of the size__. The intuition behind adding the fourier transform is that the elementwise norm of the fourier transform is translation invariant, which is useful when we want features that are independent of the position of the character in the image. (However, rotations commute with fourier transform, so we still need the spatial transformer network.)

Disclaimer: the thin-plate spline spatial transfomer network (`crop.py`) is `iwyoo`'s implementation [4].

[1] https://arxiv.org/abs/1606.04080  
[2] https://arxiv.org/abs/1506.02025  
[3] https://en.wikipedia.org/wiki/Hungarian_algorithm  
[4] https://github.com/iwyoo/TPS_STN-tensorflow