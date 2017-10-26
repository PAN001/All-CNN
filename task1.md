# Introduction
For the homework, the two netwrok architectures (i.e. Strided-CNN, and All-CNN) are both implemented. Additionally, the Layer-sequential unit-variance (LSUV) initialization is also implemented. I implemented them both in Tensorflow and Keras. 

The implementations of Strided-CNN and LSUV in Tensorflow are mainly for the purpose of demonstration of knowledge of Tensorflow. For later training and optimization, they are done using Keras (as Prof. Bhiksha says it is ok to use Keras in this HW). The summary of each file is as follows:

- strided_CNN_tf_LSUV.py
    - implementation of Strided-CNN, All-CNN and LSUV
    - no further optimizations
    
- strided_CNN_keras.py
    - implementation of Strided-CNN and All-CNN
    - actual configurations for training including data agumentation, parameters choice, and 

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

# LSUV

# Experiments

- experiment#1: LSUV, SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
