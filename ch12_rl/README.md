## Overview

Beginning with [Chapter 3](../ch03_linear), we have studied models ranging from simple to complex. Although they differ significantly in architecture and performance, their training and use follow a similar pattern: the training data must be collected and prepared in advance, and the model must be thoroughly trained and optimized before deployment. To use a slightly exaggerated but vivid analogy, producing a model resembles nurturing a baby in the womb. This artificial form of "life" is still fragile and cannot yet interact deeply with the outside world, so it needs a relatively enclosed environment in which to grow. The next stage in the evolution of any living organism is continual adaptation to new environments and challenges. Model training must likewise enter a new phase in which the model participates in society and learns through continuous interaction.

This chapter discusses **reinforcement learning** (RL). Reinforcement learning is not a new model architecture but an entirely new way to train models. Its central problem is how to train a model in an uncertain environment, before all the training data has been collected. To cope with this uncertainty, reinforcement learning adopts an unusual strategy: it begins using an incompletely prepared model to assist with its own training. This resembles how people learn in the real world—for example, improving at riding a bicycle through repeated attempts and practice.

Reinforcement learning covers enough material to constitute a complete discipline. Because it operates in uncertain environments, it involves extensive probabilistic analysis and complex mathematical derivations. A comprehensive treatment would require a very substantial monograph. This chapter therefore follows only the path relevant to large language models. Specifically, it follows ChatGPT's approach and explores how **Proximal Policy Optimization** (PPO) can optimize a model. The techniques used by ChatGPT are close to the frontier of reinforcement learning, so the chapter still covers most of the field's key concepts.

## Code Overview

| Code | Description |
|---|---|
| [intuition_model.ipynb](intuition_model.ipynb) | Build an intuitive connection between a large language model and a reward model |
| [utils.py](utils.py) | Define the game and its visualization utilities |
| [value_learning.ipynb](value_learning.ipynb) | Value-function learning |
| [policy_learning.ipynb](policy_learning.ipynb) | Policy learning |
| [a2c.ipynb](a2c.ipynb) | Baselines and the A2C model |
| [llm_ppo.ipynb](llm_ppo.ipynb) | Optimize a large language model with PPO so the fine-tuned model receives higher scores |
| [llm\_ppo\_correct\_dropout.ipynb](llm_ppo_correct_dropout.ipynb) | Pursue the same goal as [llm_ppo.ipynb](llm_ppo.ipynb), with emphasis on using dropout correctly in PPO |
