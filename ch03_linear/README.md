## Overview

Linear regression models are highly intuitive and concise, which often leads people to overlook their importance. In fact, linear regression is a cornerstone of artificial intelligence and provides strong support for building many complex models. The following joke about a mathematician offers a vivid analogy that illustrates the importance of linear regression in AI.

One day, a mathematician grew tired of mathematics and suddenly went to a fire station to become a firefighter. The fire chief said, "You look promising, but first you have to pass a test." They went to the alley behind the station, where there was a warehouse, a fire hydrant, and a hose. The chief asked, "What would you do if the warehouse caught fire?" The mathematician replied, "I would connect the hose to the hydrant, turn on the water, and put out the fire." The chief nodded. "Exactly right! But what if you walked into the alley and the warehouse was not on fire? What would you do?" The mathematician thought for a moment and answered, "I would set the warehouse on fire." Startled, the chief asked, "Why? That is dangerous! Why would you set the warehouse on fire?" The mathematician replied:

> Because that would reduce the problem I need to solve to one I already know and have solved before.

The way we use models to solve real-world problems is similar to the mathematician's approach. When faced with an unfamiliar problem, we always try to transform it mathematically into a problem that an existing model can solve. Even in today's most complex deep neural network models, such as large language models, a closer examination reveals that they are densely packed with what are essentially linear regression models. Moreover, from a physiological perspective, the human brain struggles to handle nonlinear relationships and tends to process simpler linear ones instead. This suggests that linear models align with both our intuitive preferences and our innate capabilities. Therefore, no matter how complex or sophisticated a model may be, linear models remain an indispensable part of it. Mastering linear models is the foundation for understanding and working with complex models.

Even when used on their own, linear regression models play an important role in many scenarios. For example, they are widely used in economics. In fact, many of the economic policies we encounter are products of linear regression models. Building a model does not mean perfectly simulating the real world; it means constructing an ongoing approximation. This is why data scientists often say:

> All models are wrong, but some are useful.

A "useful" model filters out unimportant details in the data, captures the main underlying relationships, and helps us better understand and explain the data. In many cases, a linear model is exactly this kind of "useful" model: it is concise, efficient, and easy to understand. If a linear model is already useful enough, why struggle to build a complex model that is difficult to understand and may introduce new problems?

Linear models are not only the starting point of artificial intelligence but also important tools for solving real-world problems. Their clear mathematical foundations and broad range of applications make them an excellent entry point for learning and mastering AI.

## Code Overview

| Code | Description |
|---|---|
| [linear_stat.ipynb](linear_stat.ipynb) | Understand linear regression models from a statistical perspective |
| [linear_ml.ipynb](linear_ml.ipynb) | Understand linear regression models from a machine-learning perspective |
| [linear_overfitting.ipynb](linear_overfitting.ipynb) | The overfitting problem |
| [linear\_illusion_ci.ipynb](linear_illusion_ci.ipynb) | Address overfitting with confidence intervals |
| [linear\_illusion_reg.ipynb](linear_illusion_reg.ipynb) | Address overfitting with penalty terms |
