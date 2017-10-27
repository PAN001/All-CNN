# Introduction
For the homework, the two netwrok architectures (i.e. `Strided-CNN,` and `All-CNN`) are both implemented. Additionally, the `Layer-sequential unit-variance (LSUV)` initialization is also implemented. I implemented them both in `Tensorflow` and `Keras`. 

The implementations of Strided-CNN/All-CNN and LSUV in Tensorflow are mainly for the purpose of demonstration of knowledge of Tensorflow. For later training and optimization, they are done using Keras (as Prof. Bhiksha says it is ok to use Keras in this HW). The summary of each file is as follows:

- Tensorflow:
    + `strided_CNN_tf_LSUV.py`
        + implementation of Strided-CNN, All-CNN and LSUV
        + no further optimizations
    + `read_cifar10`
        + read cifar10 images from dataset downloaded from the official website
- Keras:
    + `strided_all_CNN_keras.py`
        + implementation of Strided-CNN and All-CNN
        + actual code for training including data agumentation, parameters choice, and initializations

    + `LSUV.py`
        + Keras implementation of LSUV 

# Code
For the sake of convenience, a test shell script `test.sh` is provided for testing the pretrained model on Cifar10 test set. Simply run the following, and it will help set up the environment and fit the model automatically:
```shell
sh test.sh
```

## Environment Set Up
The running environment is set up in Python 2.7.10.
By running `virtualenv`, it could help set up the environment based on the `requirements.txt` easily:
```
# Create and activate new virtual environment
virtualenv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Code Description
The All-CNN is encapsulated as a class All-CNN based on Keras. The script `strided_all_CNN_keras.py` allows to train the model from the beginning or pretrained model, fit the data based on pretrained model, and plot figures. The detailed usage is as follows:
```
python strided_all_CNN_keras.py --help

usage: strided_all_CNN_keras.py [-h] [-id ID] [-batchsize BATCH_SIZE]
                                [-epoches EPOCHES] [-init INITIALIZER]
                                [-retrain RETRAIN] [-weightspath WEIGHTS_PATH]
                                [-train] [-bn] [-dropout] [-f]

optional arguments:
  -h, --help            show this help message and exit
  -id ID                the running instance id
  -batchsize BATCH_SIZE
                        batch size
  -epoches EPOCHES      the numer of epoches
  -init INITIALIZER     the weight initializer
  -retrain RETRAIN      whether to train from the benginning or read weights
                        from the pretrained model
  -weightspath WEIGHTS_PATH
                        the path of the pretrained model/weights
  -train                whether to train or test
  -bn                   whether to perform batch normalization
  -dropout              whether to perform dropout with 0.5
  -f                    whether to plot the figure
```

## Run the Code: Test Model
The best model with `90.88%` accuracy on Cifar10 test set at epoch 339 with loss of `0.4994` are stored as `all_cnn_weights_0.9088_0.4994.hdf5`. The following code loads the pretrained model and then fits the Cifar10 test data:
```
# load the pretrained model and then fit the Cifar10 test data:
python all_CNN_keras_oo -weightspath all_cnn_weights_0.9088_0.4994.hdf5 -f
```


# Network Architecture
My implementation of Strided CNN and All CNN follows the architecture of Strided-CNN-C and All-CNN-C in the paper:

- Strided-CNN:
    A model in which max-pooling is removed and the stride of the convolution layers preceding the max-pool layers is increased by 1 (to ensure that the next layer covers the same spatial region of the input image as before). 

- All-CNN:
    A model in which max-pooling is replaced by a convolution layer.

 The summary of the architecture is shown in the table below:

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

## Global Average Pooling
Conventional convolutional neural networks perform convolution in the lower layers of the network. For classification, the feature maps of the last convolutional layer are vectorized and fed into fully connected layers followed by a softmax logistic regression layer. This structure bridges the convolutional structure with traditional neural network classifiers. It treats the convolutional layers as feature extractors, and the resulting feature is classified in a traditional way.

However, the fully connected layers are prone to overfitting, thus hampering the generalization ability of the overall network. 

The paper [3] proposed a global average pooling to replace the traditional fully connected layers in CNN. The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer. 
![Global Average Pooling](https://github.com/PAN001/Strided-CNN/blob/master/gap.png?raw=true "Global Average Pooling")

For example, in the case of classification with 10 categories (CIFAR10, MNIST), and Tensorflow environment, if the output from the end of last convolution is a `3D 8,8,128` tensor, in the traditional method, it is flattened into a 1D vector of size `8x8x128`. And then one or several fully connected layers are added and at the end, a softmax layer that reduces the size to 10 classification categories and applies the softmax operator.

The global average pooling is to compute the average over the 8,8 slices for a 3D 8,8,10 tensor, which ends up with a 3D tensor of shape 1,1,10. Then, reshape it into a 1D vector of shape 10. Finally, add a softmax operator without any operation in between. The tensor before the average pooling is supposed to have as many channels as the model has classification categories.

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
After the success of CNNs in IVSRC 2012 (Krizhevsky et al. (2012)), initialization with Gaussian noise with mean equal to zero and standard deviation set to 0.01 and adding bias equal to one for some layers become very popular. However, it is not possible to train very deep network from scratch with it (Simonyan & Zisserman (2015)). The problem is caused by the activation (and/or) gradient magnitude in final layers (He et al. (2015)). If each layer, not properly initialized, scales input by k, the final scale would be kL, where L is a number of layers. Values of k > 1 lead to extremely large values of output layers, k < 1 leads to a diminishing signal and gradient.

Glorot & Bengio (2010) proposed a formula for estimating the standard deviation on the basis of the number of input and output channels of the layers under assumption of no non-linearity between layers. 

Layer-sequential unit-variance (LSUV) initialization is a data-driven weights initialization that extends the orthonormal initialization Saxe et al. (2014) to an iterative procedure. The proposed scheme can be viewed as an orthonormal initialization combined with batch normalization performed only on the first mini-batch. There are two main steps:

1. First, fill the weights with Gaussian noise with unit variance. 
2. Second, decompose them to orthonormal basis with QR or SVD-decomposition and replace weights with one of the components.
3. Third, estimate output variance of each convolution and inner product layer and scale the weight to make variance equal to one.

![](https://leanote.com/api/file/getImage?fileId=59dbf6c1ab6441555200040c)

In `LSUV.py`, it is implemented as follows:
```python
def LSUV_init(model, batch_xs, layers_to_init = (Dense, Convolution2D)):
    margin = 1e-6
    max_iter = 10
    layers_cnt = 0
    for layer in model.layers:
        if not any([type(layer) is class_name for class_name in layers_to_init]):
            continue
        print("cur layer is: ", layer.name)

        # as layers with few weights tend to have a zero variance, only do LSUV for complicated layers
        if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
            print(layer.name, 'with output shape fewer than 32, not inited with LSUV')
            continue

        print('LSUV initializing', layer.name)
        layers_cnt += 1
        weights_all = layer.get_weights();
        weights = np.array(weights_all[0])

        # pre-initialize with orthonormal matrices
        weights = svd_orthonormal(weights.shape)
        biases = np.array(weights_all[1])
        weights_all_new = [weights, biases]
        layer.set_weights(weights_all_new)

        iter = 0
        target_var = 1.0 # the targeted variance

        layer_output = forward_prop_layer(model, layer, batch_xs)
        var = np.var(layer_output)
        print("cur var is: ", var)

        while (abs(target_var - var) > margin):
            # update weights based on the variance of the output
            weights_all = layer.get_weights()
            weights = np.array(weights_all[0])
            # print(weights)
            biases = np.array(weights_all[1])
            # if np.abs(np.sqrt(var1)) < 1e-7: break  # avoid zero division

            weights = weights / np.sqrt(var) # try to scale the variance to the target
            weights_all_new = [weights, biases]
            layer.set_weights(weights_all_new) # update new weights

            layer_output = forward_prop_layer(model, layer, batch_xs)
            var = np.var(layer_output)
            print("cur var is: ", var)

            iter = iter + 1
            if iter > max_iter:
                break

    print('LSUV: total layers initialized', layers_cnt)
    return model
```

The idea of a data-driven weight initialization, rather than theoretical computation for all layer types, is very attractive: as ever more complex nonlinearities and network architectures are devised, it is more and more difficult to obtain clear theoretical results on the best initialization. This paper elegantly sidesteps the question by numerically rescaling each layer of weights until the output is approximately unit variance. The simplicity of the method makes it likely to be used in practice, although the absolute performance improvements from the method are quite small.

## He Uniform Initialization
It draws samples from a uniform distribution within `[-limit, limit]` where `limit` is `sqrt(6 / fan_in)` where `fan_in` is the number of input units in the weight[2].

# Data Augmentation
## ZCA Whitening
Whitening is a transformation of data in such a way that its covariance matrix is the identity matrix. Hence whitening decorrelates features. It is used as a preprocessing method.

It is implemented in Python as follows:
```python
# Calculate principal components
sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
u, s, _ = linalg.svd(sigma)
principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)

# Apply ZCA whitening
whitex = np.dot(flat_x, principal_components)
```

# Experiments
Due to the lack of GPU resources and long training process, I have no choice but to only train fewer than 10 epoches for each different parameter setting for evaluation. The evaluation metrics include the accuracy, loss, and the speed of convergency on test set. The experiments are mainly focued on the following four:

1. Different ways of weight initialization
2. Different ways of data augmentation   
3. Different training optimizers
4. Different training modes

## Experiment1: weight initialization

| Parameter    | Setting                                                                                       |
|--------------|-----------------------------------------------------------------------------------------------|
| Training     | batchsize = 32, SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov                                              |
| Optimizer   | dropout with 0.5 after each "pooling" layer"     |
| Data Augmentation | horizontal and vertical shift within the range of 10%, horizontal flipping, zca whitening |

In the first experiment, I compared the effectiveness of different initialization strategies. Specifically, LSUV initialization, Glorot normal initialization, He uniform initialization, together with simple Gaussian distribution initialization are compared. As figures shown below, LSUV (blue) achieves best performance in first 3000 batches.

![Experiment1: model accuracy on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp1_acc.png?raw=true "Experiment1: model accuracy on training set")
![Experiment1: model loss on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp1_loss.png?raw=true "Experiment1: model loss on training set")

## Experiment2: Data Augmentation

| Parameter   | Setting                                          |
|-------------|--------------------------------------------------|
| Training    | batchsize = 32, SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov |
| Optimizer   | dropout with 0.5 after each "pooling" layer"     |
| Initializer | LSUV                                             |

In the second experiment, I compared the effectiveness of different image preprocessing. Specifically, shift, flipping, normalization and zca whitening are compared. As figures shown below, augmentation by shifting, flipping, normalization and zca whitening (green) helps the model coverge more quickly in first 3000 batches. 

![Experiment2: model accuracy on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp2_acc.png?raw=true "Experiment2: model accuracy on training set")
![Experiment2: model loss on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp2_loss.png?raw=true "Experiment2: model loss on training set")


## Experiment3: Optimizer

| Parameter         | Setting                                               |
|-------------------|-------------------------------------------------------|
| Optimizer         | None                                                  |
| Data Augmentation | horizontal and vertical shift within the range of 10% |
| Initializer       | LSUV                                                  |

In the third experiment, I compared the effectiveness of different optimizations. Specifically, dropout and batch normalization are compared. As figures shown below, the model with batch normalization (green) performs better in first 3000 batches. However, since the 3000 batches is a very small number, the power of dropout in overcoming overfitting can not be revealed in this early training stage.

![Experiment3: model accuracy on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp3_acc.png?raw=true "Experiment3: model accuracy on training set")
![Experiment3: model loss on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp3_loss.png?raw=true "Experiment3: model loss on training set")


## Experiment4: Training Mode

| Parameter         | Setting                                               |
|-------------------|-------------------------------------------------------|
| Training          | batchsize = 32, SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov      |
| Data Augmentation | horizontal and vertical shift within the range of 10% |
| Initializer       | LSUV                                                  |

In the last experiment, I compared the effectiveness of different training modes. Specifically, SGD, RMSProp and Adam are compared:

- SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov
- RMSprop: lr=0.001, rho=0.0, epsilon=1e-08, decay=0.001
- Adam: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0

As figures shown below, the training process with RMSProp (orange) is very unstable and converge much slower than the other two.

![Experiment4: model accuracy on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp4_acc.png?raw=true "Experiment4: model accuracy on training set")
![Experiment4: model loss on training set](https://github.com/PAN001/Strided-CNN/blob/master/exp4_loss.png?raw=true "Experiment4: model loss on training set")

<!-- - experiment#1: LSUV, SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), horizontally and vertically shift within the range of 10%, horizontal flipping
- experiment#2: zca_whitening -->

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
| Training    | batchsize = 32, SGD: lr=0.01, decay=1e-6, momentum=0.9, nesterov |
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

[3] Lin M, Chen Q, Yan S. Network in network[J]. arXiv preprint arXiv:1312.4400, 2013.
