---
type: concept
aliases: [世界隐空间, 世界状态空间, World State Space]
---

# World Latent Space

## 定义

世界模型中用于表示环境状态的隐向量空间，不与特定感知模态（像素/token/动作）绑定，而是对世界状态的抽象内部表示，下游任务通过轻量 Readout 头将其解码为具体输出。

## 数学形式

$$
S_t \in \mathbb{R}^{d_s}, \quad S_t = f_\Theta(v_t,\, z_t)
$$

其中 $v_t$ 为视觉观测，$z_t$ 为隐式动态，$d_s$ 为隐空间维度。

## 核心要点

1. 不针对单一任务设计，作为多任务的共享表示瓶颈
2. 通过 Next-State-Prediction 学习而非像素重建，更高效
3. 理想情况下应捕获物理规律、因果关系、物体属性等世界知识
4. 当前实现（如 Orca）仍以 ViT 隐空间为对齐目标，并非完全自由的世界表示

## 代表工作

- [[Orca]]: 通过双路预训练（无意识+有意识）构建通用世界隐空间
- [[V-JEPA]]: 早期在 ViT 隐空间做预测编码的世界模型

## 相关概念

- [[Next-State Prediction]]
- [[World State Transition]]
- [[Unconscious Learning]]
- [[Conscious Learning]]
