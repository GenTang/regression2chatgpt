## Overview

Neural networks receive enormous attention in artificial intelligence, but they are not the field's only important models. AI includes too many classic models to cover each one in detail. This chapter therefore examines several especially instructive models. Some are closely related to neural networks, while others work well alongside them: **decision trees** and their derivatives, **hidden Markov models**, and **unsupervised learning**.

1. A decision tree is intuitive and easy to understand, making it an outstanding example of connectionist modeling. In practice, decision trees are often combined with other models. They can extract key features, and their clear structure can improve the interpretability of an entire system. Like neural networks, they can also be assembled into more powerful derived models, including random forests and gradient-boosted decision trees.
2. Hidden Markov models were once extremely popular in fields such as speech recognition and financial markets. In finance, the Medallion Fund—often described as the most profitable quantitative fund in history—has used hidden Markov models. The model can be viewed as a special case of a recurrent neural network, which is why it is included here.
3. Every model discussed so far, from simple linear regression to complex large language models, belongs to supervised learning. Such models require a label variable in the data. Many real-world datasets have no labels, and unsupervised learning is needed in these situations. This chapter introduces three major categories of unsupervised models: clustering, dimensionality reduction, and singular value decomposition.

The chapter is somewhat self-contained. It broadens our perspective and helps us understand the origins and meaning of several techniques used in neural networks.

## Code Overview

| Code | Description |
|---|---|
| [dt_example.ipynb](dt_example.ipynb) | Decision-tree models |
| [dt_logit.ipynb](dt_logit.ipynb) | Combine decision trees with logistic regression, using trees for feature extraction |
| [gbts.ipynb](gbts.ipynb) | Gradient-boosted decision trees |
| [viterbipy.py](viterbipy.py) | Implement the Viterbi algorithm |
| [stock_analysis.ipynb](stock_analysis.ipynb) | Analyze Chinese A-share data with a hidden Markov model |
| [kmeans.ipynb](kmeans.ipynb) | K-means clustering |
| [kmeans\_choose_k.ipynb](kmeans_choose_k.ipynb) | Choose the number of clusters |
| [pca.ipynb](pca.ipynb) | Principal component analysis for dimensionality reduction |
