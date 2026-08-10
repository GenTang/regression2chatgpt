## Overview

[Chapter 8](../ch08_mlp) examined the classic multilayer perceptron in depth and demonstrated its remarkable versatility in classification. In practice, an MLP often performs well on datasets that already have effective feature representations. Other machine-learning models can perform just as well on such data, however. A neural network may offer no additional improvement while sacrificing interpretability. Neural networks must therefore prove their unique value in situations where the objects being modeled cannot readily be represented as vectors.

Such modeling challenges are common in real life because high-quality feature representations are relatively rare. Humans can recognize and digitize images with ease, yet converting an image into a useful vector is extremely difficult. We do not fully understand how the human body performs this apparently simple task or which elements of an image influence our perception. As a result, computers can calculate the gradient of a complex loss function yet struggle with the seemingly simple task of recognizing an image.

As discussed in [Chapter 8](../ch08_mlp), the hidden layers of a neural network can be viewed as tools for automatic feature extraction. Can this capability reach or surpass the human level? In other words, when humans cannot extract effective features, can a neural network do so automatically and complete the modeling task? The answer is yes. This chapter explains how neural networks can extract image features automatically and thereby give computers the ability to see.

The **convolutional neural network** (CNN) discussed in this chapter is an important milestone in deep learning. It demonstrated the potential of depth: increasing the number of network layers can produce astonishing gains in performance. Its pioneers were early adopters of GPUs rather than traditional CPUs for model training, greatly accelerating computation. Their success drove the widespread adoption of GPUs for neural networks, made training deep networks practical, and accelerated the development of deep learning. It is no exaggeration to say that CNNs opened the deep-learning era and brought AI out of the laboratory and into the real world. In project after project, artificial intelligence surpassed human performance in remarkably little time. Perhaps the words "I came, I saw, I conquered" have echoed within it all along.

## Code Overview

| Code | Description |
|---|---|
| [mnist.ipynb](mnist.ipynb) | Display the training data used in this chapter |
| [mlp.ipynb](mlp.ipynb) | Recognize images with a multilayer perceptron |
| [conv_example.ipynb](conv_example.ipynb) | Implement two-dimensional convolution |
| [cnn.ipynb](cnn.ipynb) | Recognize images with a convolutional neural network |
| [res_nets.ipynb](res_nets.ipynb) | Recognize images with residual networks |
