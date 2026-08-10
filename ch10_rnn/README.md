## Overview

[Chapter 8](../ch08_mlp) and [Chapter 9](ch09_cnn) examined multilayer perceptrons and convolutional neural networks in depth. Although their architectures differ substantially, they share a basic assumption about data: individual observations are independent, and the model focuses only on the relationship between the features and label of the current observation. Such models are commonly called **vanilla neural networks**. In image recognition, for example, a CNN classifies each image independently without considering possible relationships between images. CNNs can also classify text. A sentiment-analysis system might label a sentence as positive ("After extra time, China came from behind to win") or negative ("My tears would not stop falling"). Here again, the model processes each sentence independently and ignores dependencies between them.

Not all data satisfies this independence assumption. If the sentences in a sentiment-analysis task come from the same article, understanding one sentence requires its context because the same sentence can convey different emotions in different settings. Consider the passage: "After extra time, China came from behind to win. My tears would not stop falling." In this context, the second sentence expresses a positive emotion. Data with such dependencies is called **sequential data** or **sequence data**. Typical examples include financial-market prices (time series), text (character or token sequences), and video (image sequences).

Vanilla neural networks usually perform poorly on sequential data because their architecture limits their ability to learn dependencies. Clever designs can strengthen this ability, but the improvement is often limited and may introduce other modeling problems. [char_mlp.ipynb](char_mlp.ipynb) uses a concrete example to explain how vanilla neural networks can learn sequential data, together with the advantages and disadvantages of this approach.

To overcome these limitations, researchers introduced the **recurrent neural network** (RNN), a fundamentally different architecture that has produced remarkable results in many settings. Its performance in **natural language processing** (NLP) has been especially impressive. Large language models—the systems that have amazed and even frightened the world—were built on foundations established by recurrent neural networks. Beginning with this chapter, we focus on NLP and RNNs and explore how this emerging form of intelligence can understand human language and acquire the knowledge encoded within it.

## Code Overview

| Code | Description |
|---|---|
| [tokenizer.ipynb](tokenizer.ipynb) | How tokenizers process different languages |
| [char_mlp.ipynb](char_mlp.ipynb) | Use an MLP for autoregressive language learning by predicting the next character from context |
| [embedding_example.ipynb](embedding_example.ipynb) | Explain text embeddings with a simple implementation |
| [char\_rnn.ipynb](char_rnn.ipynb) | Use an RNN for autoregressive language learning; the implementation is inefficient but easy to understand |
| [char\_rnn_batch.ipynb](char_rnn_batch.ipynb) | Use an RNN for autoregressive language learning with batch computation |
| [bptt_example.ipynb](bptt_example.ipynb) | Visualize the details of backpropagation through time with a computation graph |
| [lstm.ipynb](lstm.ipynb) | Use an LSTM network for autoregressive language learning |
