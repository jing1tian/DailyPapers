---
type: concept
aliases: [Adversarial Motion Prior, 对抗运动先验]
---

# AMP（Adversarial Motion Prior）

## 定义
一种基于对抗学习的运动风格化方法，通过判别器区分策略生成运动与参考运动捕捉数据，将运动真实性作为隐式奖励信号引导策略学习。

## 数学形式

$$
r_{\text{style}} = -\log(1 - D(\bm{s}_t, \bm{s}_{t+1}))
$$

判别器 $D$ 区分策略转换与 MoCap 数据转换，风格奖励来自判别器输出。

## 核心要点

1. 无需手工设计运动奖励，由判别器自动提供风格信号
2. 与任务奖励组合：$r = w_{\text{task}} r_{\text{task}} + w_{\text{style}} r_{\text{style}}$
3. **局限**：判别器训练不稳定，大规模多样运动时容易 mode collapse

## 代表工作

- [[SONIC]]: 对比方法，AMP 在大规模多样运动追踪中不如密集监督方法

## 相关概念

- [[ASE]]: AMP 的扩展，增加技能潜变量
- [[运动捕捉]]: AMP 使用的参考数据来源
- [[近端策略优化]]: 常与 AMP 配合使用的 RL 算法
