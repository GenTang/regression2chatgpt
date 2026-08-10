## Overview

**Econometrics** is an important branch of economics. Based on mathematical statistics, it uses real-world data and mathematical models to test economic theories. Like a ruler for economic research, it advances the field from qualitative discussion to quantitative analysis. [Linear regression](../ch03_linear) and [logistic regression](../ch04_logit) are its core models.

The economist John Maynard Keynes once observed:

> The ideas of economists and political philosophers exert far more power than people usually imagine. Even practical people often remain servants of some long-dead economist.

This observation highlights the enormous influence economic theory has on the world. Ensuring that such theories are accurate is therefore crucial. As the primary tool for validating economic theory, econometrics places demanding requirements on its core models. Although their structures are relatively simple, the field has accumulated many techniques for refining model details. These techniques have two main goals. The first is to process features so models can use them more effectively—what artificial intelligence calls feature engineering. The second is to make models as valid, stable, and interpretable as possible.

The first group of techniques does not depend on model structure and can therefore benefit any model. The second depends heavily on structure, making the analysis harder as models become more complex. As later chapters explain, however, a complex model can usually be decomposed into a feature-extraction model followed by either a linear model for regression or a logistic regression model for classification. In other words, the outermost layer of a complex model is often one of econometrics' core models. Econometric analysis can therefore help assess the validity and stability of the overall system and provide a degree of interpretability. Despite some theoretical limitations, it remains a useful solution.

This chapter introduces no new model architecture, but its topics are essential to most modeling scenarios.

## Code Overview

| Code | Description |
|---|---|
| [categorical_variable.ipynb](categorical_variable.ipynb) | Processing categorical features |
| [continuous_variable.ipynb](continuous_variable.ipynb) | Processing continuous features |
| [multicollinearity.ipynb](multicollinearity.ipynb) | The multicollinearity problem |
| [one\_way_anova.ipynb](one_way_anova.ipynb) | Use one-way ANOVA to detect multicollinearity between continuous and categorical features |
