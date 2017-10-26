# Introduction
For the homework, the two netwrok architectures (i.e. `Strided-CNN,` and `All-CNN`) are both implemented. Additionally, the `Layer-sequential unit-variance (LSUV)` initialization is also implemented. I implemented them both in `Tensorflow` and `Keras`. 

The implementations of Strided-CNN and LSUV in Tensorflow are mainly for the purpose of demonstration of knowledge of Tensorflow. For later training and optimization, they are done using Keras (as Prof. Bhiksha says it is ok to use Keras in this HW). The summary of each file is as follows:

- `strided_CNN_tf_LSUV.py`
    - implementation of Strided-CNN, All-CNN and LSUV
    - no further optimizations
    
- `strided_CNN_keras.py`
    - implementation of Strided-CNN and All-CNN
    - actual code for training including data agumentation, parameters choice, and initializations

- `LSUV.py`

# Running the code


# Network Architecture
My implementation of Strided CNN and All CNN follows the architecture of Strided-CNN-C and All-CNN-C in the paper. The summary of the architecture is shown in the table below:

|           | Strided-CNN-C                                                                       | All-CNN-C                                                                           |
|-----------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| Conv1     | 3 input channel, 3*3 filter, 96 ReLU, stride = 1 (?, 32, 32, 96)                    | 3 input channel, 3*3 filter, 96 ReLU, stride = 1 (?, 32, 32, 96)                    |
| Conv2     | 96 input channel, 3*3 filter, 96 ReLU, stride = 2 (?, 16, 16, 96)                   | 96 input channel, 3*3 filter, 96 ReLU, stride = 1 (?, 32, 32, 96)                   |
| Conv3     | 96 input channel, 3*3 filter, 192 ReLU, stride = 1 (?, 16, 16, 192)                 | 96 input channel, 3*3 filter, 192 ReLU, stride = 2 (?, 16, 16, 192)                 |
| Droupout1 | P(droupout) = 0.5                                                                   | P(droupout) = 0.5                                                                   |
| Conv4     | 192 input channel, 3*3 filter, 192 ReLU, stride = 2 (?, 8, 8, 192)                  | 192 input channel, 3*3 filter, 192 ReLU, stride = 1 (?, 16, 16, 192)                |
| Conv5     | 192 input channel, 3*3 filter, 192 ReLU, stride = 2, padding = valid (?, 6, 6, 192) | 192 input channel, 3*3 filter, 192 ReLU, stride = 1 (?, 16, 16, 192)                |
| Conv6     | 192 input channel, 1*1 filter, 192 ReLU, stride = 1, padding = valid (?, 6, 6, 192) | 192 input channel, 3*3 filter, 192 ReLU, stride = 2 (?, 8, 8, 192)                  |
| Droupout2 | P(droupout) = 0.5                                                                   | P(droupout) = 0.5                                                                   |
| Conv7     | 192 input channel, 1*1 filter, 10 ReLU, stride = 1 (?, 6, 6, 10)                    | 192 input channel, 3*3 filter, 192 ReLU, stride = 1, padding = valid (?, 6, 6, 192) |
| Conv8     |                                                                                     | 192 input channel, 1*1 filter, 192 ReLU, stride = 1 (?, 6, 6, 192)                  |
| Conv9     |                                                                                     | 192 input channel, 1*1 filter, 10 ReLU, stride = 1 (?, 6, 6, 10)                    |
| GAP       | 10 input channel, 6*6 filter, stride = 1, padding = same (?, 1, 1, 10)              | 10 input channel, 6*6 filter, stride = 1, padding = same (?, 1, 1, 10)              |
| Softmax   | 10-way softmax of flat representation (?, 1, 10) of GAP output                      | 10-way softmax of flat representation (?, 1, 10) of GAP output                      |
     
- (?, n, n, n) represents the output of each layer where ? is the number of input images

# Weight Initialization
When working with deep neural networks, initializing the network with the right weights can be the difference between the network converging in a reasonable amount of time and the network loss function not going anywhere even after hundreds of thousands of iterations.

In short, the weight initialization is of great important for training a network:

- If the weights in a network start too small, then the signal shrinks as it passes through each layer until it’s too tiny to be useful.
- If the weights in a network start too large, then the signal grows as it passes through each layer until it’s too massive to be useful.

## Based on Gaussian Distribution
Without knowing about the training data, one good way of initialization is to assign the weights from a Gaussian distribution which has zero mean and some finite variance. 

## Glorot normal Initialization (Xavier Initialization)
The motivation for Xavier initialization in neural networks is to initialize the weights of the network so that the neuron activation functions are not starting out in saturated or dead regions. In other words, we want to initialize the weights with random values that are not "too small" and not "too large". Specifically, it draws samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))` where `fan_in is` the number of input units in the weight tensor and `fan_out` is the number of output units in the weight tensor.

## LSUV Initialization
Layer-sequential unit-variance (LSUV) initialization is an extension of  orthonormal initialization Saxe et al. (2014) to an iterative procedure proposed by Mishkin et al. (2015)[1]. First, it fills the weights with Gaussian noise with unit variance. Second, decompose them to orthonormal basis with QR or SVD-decomposition and replace weights with one of the components. The LSUV process then estimates output variance of each convolution and inner product layer and scales the weight to make variance equal to one. The proposed scheme can be viewed as an orthonormal initialization combined with batch normal- ization performed only on the first mini-batch.

The idea of a data-driven weight initialization, rather than theoretical computation for all layer types, is very attractive: as ever more complex nonlinearities and network architectures are devised, it is more and more difficult to obtain clear theoretical results on the best initialization. This paper elegantly sidesteps the question by numerically rescaling each layer of weights until the output is approximately unit variance. The simplicity of the method makes it likely to be used in practice, although the absolute performance improvements from the method are quite small.

## He Uniform Initialization
It draws samples from a uniform distribution within `[-limit, limit]` where `limit` is `sqrt(6 / fan_in)` where `fan_in` is the number of input units in the weight[2].

# Data Augmentation
## ZCA Whitening


# Experiments
Due to the lack of GPU resources and long training process, I have no choice but to only train fewer than 10 epoches for each different parameter setting for evaluation. The evaluation metrics include the accuracy, loss, and the speed of convergency on test set. The experiments are mainly focued on the following three parameters:

1. Different ways of weight initialization
2. Different ways of data augmentation   
3. Different training strategies
4. Different training optimizers

## Experiment1: weight initialization

| Parameter    | Setting                                                                                       |
|--------------|-----------------------------------------------------------------------------------------------|
| Training     | SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov                                              |
| Optimizer   | dropout with 0.5 after each "pooling" layer"     |
| Data Augmentation | horizontal and vertical shift within the range of 10%, horizontal flipping, zca whitening |

In the first experiment, I compared the effectiveness of different initialization strategies. Specifically, LSUV initialization, Glorot normal initialization, He uniform initialization, together with simple Gaussian distribution initialization are compared. As figures shown below, LSUV achieves best performance in first 3000 batches.

![Experiment1: model accuracy on training set](exp1_acc.png?raw=true "Experiment1: model accuracy on training set")
![Experiment1: model loss on training set](exp1_loss.png?raw=true "Experiment1: model loss on training set")

## Experiment2: Data Augmentation

| Parameter   | Setting                                          |
|-------------|--------------------------------------------------|
| Training    | SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov |
| Optimizer   | dropout with 0.5 after each "pooling" layer"     |
| Initializer | LSUV                                             |

In the second experiment, I compared the effectiveness of different image preprocessing. Specifically, shift, flipping, normalization and zca whitening are compared. As figures shown below, augmentation by shifting, flipping, normalization and zca whitening helps the model coverge more quickly in first 3000 batches. 

![Experiment2: model accuracy on training set](exp2_acc.png?raw=true "Experiment1: model accuracy on training set")
![Experiment2: model loss on training set](exp2_loss.png?raw=true "Experiment1: model loss on training set")

- experiment#1: LSUV, SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), horizontally and vertically shift within the range of 10%, horizontal flipping
- experiment#2: zca_whitening

# Conclusion
The final model reveals `90.88%` accuracy on test set at epoch 339 with loss of `0.4994`. It is a typical `All-CNN` architecture summaried as follows:
```
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 32, 32, 96)        2688      
    _________________________________________________________________
    activation_1 (Activation)    (None, 32, 32, 96)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 96)        83040     
    _________________________________________________________________
    activation_2 (Activation)    (None, 32, 32, 96)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 16, 16, 96)        83040     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 16, 96)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 16, 16, 192)       166080    
    _________________________________________________________________
    activation_3 (Activation)    (None, 16, 16, 192)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 192)       331968    
    _________________________________________________________________
    activation_4 (Activation)    (None, 16, 16, 192)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 8, 8, 192)         331968    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 8, 8, 192)         0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 8, 8, 192)         331968    
    _________________________________________________________________
    activation_5 (Activation)    (None, 8, 8, 192)         0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 8, 8, 192)         37056     
    _________________________________________________________________
    activation_6 (Activation)    (None, 8, 8, 192)         0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 8, 8, 10)          1930      
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 10)                0         
    _________________________________________________________________
    activation_7 (Activation)    (None, 10)                0         
    =================================================================
    Total params: 1,369,738
    Trainable params: 1,369,738
    Non-trainable params: 0
```

The parameter setting is as follows:

| Parameter   | Setting                                          |
|-------------|--------------------------------------------------|
| Training    | SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov |
| Optimizer   | dropout with 0.5 after each "pooling" layer"     |
| Initializer | LSUV                                             |

Notice:
This may not be the best parameter setting since in the later experiments, it is found that LSUV may help the model converge more quickly, and thus it may result in a better model. Moreover, more data augmentation may be helpfule as well. This is just a model that I get with the limited training time. 

# Future Work
## Use of Scheduler
In the original paper learning rate of 'γ' and scheduler S = "e1 ,e2 , e3" were used in which γ is multiplied by a fixed multiplier of 0.1 after e1. e2 and e3 epochs respectively. (where e1 = 200, e2 = 250, e3 = 300). 

But in my experiments, due to the long training time, I only experimented with a learning rate of 0.1, decay of 1e-6 and momentum of 0.9. 

## Data Augmentation
In the original paper very extensive data augmentation were used such as placing the cifar10 images of size 32 × 32 into larger 126 × 126 pixel images and can hence be heavily scaled, rotated and color augmented. 

But in my experiments, due to the long training time, I only experimented with mild data augmentation.

# Reference
[1] Mishkin D, Matas J. All you need is a good init[J]. arXiv preprint arXiv:1511.06422, 2015.

[2] He K, Zhang X, Ren S, et al. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification[C]//Proceedings of the IEEE international conference on computer vision. 2015: 1026-1034.
MLA 
