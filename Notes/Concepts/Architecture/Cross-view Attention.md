---
type: concept
aliases: [跨视角注意力, 多视角一致性注意力, Multi-view Attention]
---

# Cross-view Attention

## 定义

在多相机视角的视觉生成或预测任务中，通过跨视角注意力机制强制不同视角特征之间的空间一致性，使多视角输出在三维空间中相互兼容。

## 数学形式

设多视角特征 $\{F^v\}_{v=1}^{V}$，跨视角注意力计算为：

$$
\text{CrossViewAttn}(F^v) = \text{Softmax}\left(\frac{Q^v (K^{1:V})^\top}{\sqrt{d}}\right) V^{1:V}
$$

其中 $K^{1:V}$, $V^{1:V}$ 来自所有视角的特征拼接。

## 核心要点

1. **多视角打包**: 通常将多视角帧沿时间维度拼接，加入可学习视角嵌入（view embeddings）区分不同相机
2. **一致性约束**: 通过 attention 让每个视角的特征感知其他视角，隐式学习多视角几何约束
3. **与 Self-Attention 结合**: 通常与时序 Self-Attention 交替使用，分别处理时序和跨视角关系

## 代表工作

- [[A2World]]: 在 DiT 扩散模型中引入跨视角注意力实现多相机一致性机器人世界模型

## 相关概念

- [[Diffusion Transformer (DiT)|DiT]]
- [[Action Conditioning]]
- [[Action-Conditioned World Model]]
