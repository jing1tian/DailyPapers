---
type: concept
aliases: [Adversarial Skill Embeddings, 对抗技能嵌入]
---

# ASE（Adversarial Skill Embeddings）

## 定义
AMP 的扩展方法，在对抗运动先验基础上引入潜在技能变量，学习多样化的运动技能库，支持高层策略通过技能编码控制低层运动风格。

## 数学形式

$$
r = r_{\text{task}} + r_{\text{style}}(z), \quad z \sim \mathcal{Z}
$$

其中 $z$ 为潜在技能编码，判别器以 $(z, \bm{s}_t, \bm{s}_{t+1})$ 为输入区分真实与生成运动。

## 核心要点

1. 引入技能潜变量 $z$，使同一策略网络支持多种运动风格
2. 高层规划器选择技能编码，低层执行器产生对应运动
3. 继承 AMP 的 mode collapse 问题，大规模应用受限

## 代表工作

- [[SONIC]]: 对比方法，ASE 对比密集监督的运动追踪在多样性扩展上存在局限

## 相关概念

- [[AMP]]: ASE 的基础方法
- [[运动捕捉]]: 训练参考数据
- [[有限标量量化]]: SONIC 采用的替代方案，避免 mode collapse
