# 《概率之舞：从模型到语言的人工智能征程》配套代码

本书正在编辑之中，不久就会面市。

## 简要说明

对于人工智能的经典模型，第三方开源工具都提供了封装良好的实现，使用它们并不复杂。然而，这些开源工具出于工程化的考虑，在代码中引入了过多的封装和细节，使得理解模型的核心结构变得困难。为帮助读者更好地掌握模型原理，本书特别投入较大精力重新实现了模型的核心部分，并附有详细注释。有时候，用人类的语言描述一些精妙的算法处理需要较大篇幅，而且效果也不尽如人意。相比之下，阅读代码则变得直观清晰。

这份代码依赖于多个第三方库，相关的安装命令已经在相应脚本的开头提供。按照给定的顺序运行这些脚本即可。由于涉及随机数，重新运行可能会得到稍有不同的结果，但整体影响不大。值得注意的是，与大语言模型相关的代码需要在GPU上运行，否则计算时间将显著增加。

## 代码目录

- [ch03_linear](ch03_linear): 线性回归
- [ch04_logit](ch04_logit): 逻辑回归
- [ch05_econometrics](ch05_econometrics): 计量经济学的启示
- [ch06_optimizer](ch06_optimizer): 最优化算法
- [ch07_autograd](ch07_autograd): 反向传播
- [ch08_mlp](ch08_mlp): 多层感知器
- [ch09_cnn](ch09_cnn): 卷积神经网络
- [ch10_rnn](ch10_rnn): 循环神经网络
- [ch11_llm](ch11_llm): 大语言模型
- [ch12_rl](ch12_rl): 强化学习
- [ch13_others](ch13_others): 其他经典模型