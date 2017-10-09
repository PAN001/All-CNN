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
4. A global average pooling is used. For normal CNN, 

## Derived Model
### Strided-CNN-C
A model in which max-pooling is removed and the stride of the convolution layers preceding the max-pool layers is increased by 1 (to ensure that the next layer covers the same spatial region of the input image as before). 

### All-CNN-C
A model in which max-pooling is replaced by a convolution layer.
