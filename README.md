# Strided-CNN

A Python implementation of Strided-CNN.

## Background
### Global Average Pooling
Conventional convolutional neural networks perform convolution in the lower layers of the network. For classification, the feature maps of the last convolutional layer are vectorized and fed into fully connected layers followed by a softmax logistic regression layer [4] [8] [11]. This structure bridges the convolutional structure with traditional neural network classifiers. It treats the convolutional layers as feature extractors, and the resulting feature is classified in a traditional way.

However, the fully connected layers are prone to overfitting, thus hampering the generalization ability of the overall network. 

The paper [x] proposed a global average pooling to replace the traditional fully connected layers in CNN. The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer. 
![](https://www.researchgate.net/profile/Pantelis_Kaplanoglou/publication/318277197/figure/fig10/AS:513490781966337@1499437150580/Figure-16-Global-average-pooling-layer-replacing-the-fully-connected-layers-The-output.ppm)

For example, in the case of classification with 10 categories (CIFAR10, MNIST). It means that if you have a 3D 8,8,128 tensor at the end of your last convolution, in the traditional method, you flatten it into a 1D vector of size 8x8x128. And you then add one or several fully connected layers and then at the end, a softmax layer that reduces the size to 10 classification categories and applies the softmax operator.

The global average pooling means that you have a 3D 8,8,10 tensor and compute the average over the 8,8 slices, you end up with a 3D tensor of shape 1,1,10 that you reshape into a 1D vector of shape 10. And then you add a softmax operator without any operation in between. The tensor before the average pooling is supposed to have as many channels as your model has classification categories.

## Model Description
The CNN model used here differs from standard CNN in several key aspects:
1. The pooling layers are replaced with convolutional layers with stride two.
2. Small convolutional layers with k < 5 are used, which can greatly reduce the number of parameters in a network and thus serve as a form of regularization.
3. If the image area covered by units in the topmost convolutional layer covers a portion of the image large enough to recognize its content (i.e. the object we want to recognize) then fully connected layers can also be replaced by simple 1-by-1 convolutions.
4. A global average pooling is used.

## Derived Model
![](https://leanote.com/api/file/getImage?fileId=59dbf4b1ab64415777000403)
### Strided-CNN-C
A model in which max-pooling is removed and the stride of the convolution layers preceding the max-pool layers is increased by 1 (to ensure that the next layer covers the same spatial region of the input image as before). 

### All-CNN-C
A model in which max-pooling is replaced by a convolution layer.

## Optimization
### Layer-sequential unit-variance (LSUV) initialization
After the success of CNNs in IVSRC 2012 (Krizhevsky et al. (2012)), initialization with Gaussian noise with mean equal to zero and standard deviation set to 0.01 and adding bias equal to one for some layers become very popular. However, it is not possible to train very deep network from scratch with it (Simonyan & Zisserman (2015)). The problem is caused by the activation (and/or) gradient magnitude in final layers (He et al. (2015)). If each layer, not properly initialized, scales input by k, the final scale would be kL, where L is a number of layers. Values of k > 1 lead to extremely large values of output layers, k < 1 leads to a diminishing signal and gradient.

Glorot & Bengio (2010) proposed a formula for estimating the standard deviation on the basis of the number of input and output channels of the layers under assumption of no non-linearity between layers. 

Layer-sequential unit-variance (LSUV) initialization is a data-driven weights initialization that extends the orthonormal initialization Saxe et al. (2014) to an iterative procedure. The proposed scheme can be viewed as an orthonormal initialization combined with batch normalization performed only on the first mini-batch. There are two main steps:

1. First, fill the weights with Gaussian noise with unit variance. 
2. Second, decompose them to orthonormal basis with QR or SVD-decomposition and replace weights with one of the components.
3. Third, estimate output variance of each convolution and inner product layer and scale the weight to make variance equal to one.

![](https://leanote.com/api/file/getImage?fileId=59dbf6c1ab6441555200040c)
