## Overview

As the preceding chapters explained, building a model begins with the practical application: we first analyze the characteristics of the data to gain inspiration and intuition. We then use mathematical abstraction and transformation to select an appropriate model architecture. Finally, we implement the model with an open-source Python library, which estimates the model parameters for us.

From a software-design perspective, open-source Python libraries excel at **abstraction**. They hide the low-level details of model construction and training, allowing us to focus on high-level concepts and operations through a set of application programming interfaces (APIs). These interfaces often make it possible to build and train a model in only a few dozen lines of code. We need not think too much about the complex mathematics behind the model, and implementing the algorithms that estimate its parameters is no longer an obstacle. Ideally, all this complexity is perfectly abstracted away, making a data scientist's work easier and more convenient. The other side of the coin, of course, is that this lower barrier to entry may affect the number and compensation of related jobs. Unfortunately—or perhaps fortunately—models involve such complex mathematical abstractions and computations that even excellent software design cannot hide everything. Some details inevitably leak through and affect how users understand and operate the system. This is known as a **leaky abstraction**.

For example, certain datasets can cause an open-source library to fail while estimating the parameters of a logistic regression model. Such leaks occur less often with classic or simple models. In more complex models, including deep neural networks and large language models, they can be pervasive. Without understanding the underlying implementation details, it is difficult to make progress in these fields. Theoretically, one cannot grasp the essence of a model or optimize it effectively enough to achieve the desired result. In practice, program failures become difficult to fix, training can take too long, libraries become hard to use beyond their examples, and model architectures cannot be adjusted flexibly to meet specific needs.

This chapter therefore examines the core details of open-source libraries and explores how model parameters are estimated from mathematical formulas. In more academic terms, it studies algorithms for solving optimization problems. Many methods are available, and different algorithms suit different models and excel at different kinds of problems. Given the space available, the chapter focuses on the most fundamental and widely used techniques: gradient descent, stochastic gradient descent, and their variants.

## Code Overview

| Code | Description |
|---|---|
| [pytorch_tutorial.ipynb](pytorch_tutorial.ipynb) | Tensor operations and fundamentals |
| [gradient_descent.ipynb](gradient_descent.ipynb) | Implement gradient descent with PyTorch |
| [stochastic\_gradient_descent.ipynb](stochastic_gradient_descent.ipynb) | Implement stochastic gradient descent with PyTorch |
