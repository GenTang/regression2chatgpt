## Overview

Neural networks have remained a brilliant and enduring star in artificial intelligence. Over the past decade, the industry has made astonishing progress—from convolutional neural networks that demonstrated the potential of deep learning, to AlphaGo defeating human Go champions, to the remarkable emergence of ChatGPT. Neural networks have repeatedly surprised the world. Yet alongside this excitement, they have also inspired confusion, unease, and awe. Their conception can be traced to bionics: using computers and mathematical models to simulate the human nervous system. We still understand relatively little about the models themselves, however, and their operating principles remain shrouded in mystery. We know little not only about how they work but also about their ultimate limits.

Academia and industry are sharply divided over the future of neural networks. One camp argues that the current AI boom may be a bubble and that the field has yet to make a substantive breakthrough. The distinguished scholar Professor Michael I. Jordan holds this view, arguing that artificial intelligence remains far from human-level capabilities. Neural networks may be able to "simulate" intelligence in certain fields, but rigorously speaking, this is not the same as genuine intelligence. The other camp believes that AI is on the eve of a breakthrough that will bring enormous benefits to humanity. Entrepreneurs such as Mark Zuckerberg and Pony Ma hold this view. They are confident that AI will drive a new industrial revolution, reshaping industry and human life much as electricity once did.

Many people also worry that AI could create enormous risks because we are creating a new kind of intelligent agent that, in some sense, resembles an immortal form of **silicon-based life**. Elon Musk in industry and Geoffrey Hinton in academia have expressed this view. Artificial intelligence may not yet pose a direct threat, but given its rapid development, we could face substantial risks within the next five or ten years. These risks may not resemble the final war of judgment portrayed in science-fiction films, in which AI destroys humanity. They may instead take the form of large-scale unemployment as AI becomes capable of performing many jobs and displaces human workers.

Whatever position one takes, the future has already arrived. We may admire, quote, refute, question, celebrate, or criticize artificial intelligence, but we cannot ignore it. Understanding and mastering neural networks has become an essential skill for a new generation of data scientists. With the groundwork of the preceding chapters in place, this chapter begins an in-depth study of neural-network models. We start with the foundational **multilayer perceptron** (MLP), proceed to the [convolutional neural network (CNN)](../ch09_cnn) that demonstrated the potential of deep learning, then introduce the [recurrent neural network (RNN)](../ch10_rnn) for sequential data, and finally examine how advanced conversational AI systems such as ChatGPT are trained and constructed with [large language models](../ch11_llm).

## Code Overview

| Code | Description |
|---|---|
| [utils.py](utils.py) | Define MLP components such as linear models and sigmoid functions |
| [perceptron.ipynb](perceptron.ipynb) | Show the computation graph of a perceptron |
| [logit_regression.ipynb](logit_regression.ipynb) | Rebuild and train logistic regression as a neural network |
| [mlp.ipynb](mlp.ipynb) | Build a multilayer perceptron and demonstrate its versatility |
| [saturated\_activation_function.ipynb](saturated_activation_function.ipynb) | Use computation graphs to illustrate dead neurons |
| [activation_monitoring.ipynb](activation_monitoring.ipynb) | Monitor model training |
| [activation_functions.ipynb](activation_functions.ipynb) | Common activation functions |
| [initialization.ipynb](initialization.ipynb) | Improved parameter-initialization methods |
| [normalization.ipynb](normalization.ipynb) | Normalization layers |
