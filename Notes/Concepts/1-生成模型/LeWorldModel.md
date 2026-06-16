---
type: concept
aliases: [Le World Model, LeCun World Model, LeWM]
---

# LeWorldModel

## 定义
Meta/Yann LeCun 团队提出的 Joint Embedding Predictive Architecture（JEPA）世界模型，核心思路是在隐空间预测未来而非像素空间，避免重建不相关细节。

## 数学形式
JEPA 目标：在表示空间做预测
$$\hat{s}_{t+1} = f_\theta(s_t, a_t), \quad \mathcal{L} = \|s_{t+1} - \hat{s}_{t+1}\|^2$$

（不重建像素，直接预测 encoded 表示）

## 核心要点
1. 非生成式：不重建观测，只预测 latent representation
2. 避免在噪声/纹理等不相关信息上浪费建模容量
3. 被多篇 tactile/contact WM 工作用作 baseline（ContactWorld、ContactWM 等）
4. VLA-JEPA 是其在 VLA fine-tuning 领域的衍生

## 代表工作
- [[ContactWorld]]：以 LeWorldModel 作为视觉世界模型 baseline
- [[μ₀]]：action-free WM 与 Le 系方向有关联

## 相关概念
- [[Action-Conditioned World Model]]
- [[Dreamer]]
- [[VLA-JEPA]]
