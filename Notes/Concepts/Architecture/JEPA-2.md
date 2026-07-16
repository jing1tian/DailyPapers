---
type: concept
aliases: [Joint Embedding Predictive Architecture 2, 联合嵌入预测架构 2]
---

# JEPA-2

## 定义
JEPA-2 是 Meta/LeCun 组对 [[JEPA]] 的扩展，引入层级化预测结构，在 latent 空间同时预测短程动态（帧级）和长程目标（任务级），面向世界模型和 planning 应用。

## 数学形式
$$
\hat{s}_{t+k} = \text{Predictor}\left(s_t, a_{t:t+k}\right), \quad \mathcal{L} = \left\|\hat{s}_{t+k} - \text{sg}(s_{t+k})\right\|^2
$$

仅在 latent 空间计算预测损失，不重建原始像素，避免平均模糊。

## 核心要点
1. **Latent-space 预测**：区别于 pixel-level 预测，规避生成模糊均值问题
2. **层级预测**：短期和长期 predictor 分离，对应不同粒度的时序抽象
3. **自监督**：目标编码器 stop-gradient，类 BYOL 风格
4. **Action-conditioning**：通过动作序列条件化 predictor 支持规划

## 代表工作
- [[JEPA]]: I-JEPA (LeCun 2023)，图像版本基线
- [[Orca]]: BAAI 世界基础模型，采用 JEPA-2 启发的 next-state-prediction

## 相关概念
- [[JEPA]]
- [[Delta-JEPA]]
- [[AdaJEPA]]
