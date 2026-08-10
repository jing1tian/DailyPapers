---
type: concept
aliases: [Long Forcing, 渐进蒸馏, 双向到因果蒸馏]
---

# LongForcing

## 定义
一种将双向（非因果）teacher 模型渐进蒸馏为单向因果 student 模型的训练技术，使视频世界模型能够在不损失长时序质量的前提下实现实时推理。

## 核心要点
1. **阶段一（Teacher）**：使用双向注意力（全序列可见）训练 teacher model，获得高质量的未来帧预测能力
2. **阶段二（Student）**：将 teacher 知识蒸馏到因果 student model（单向注意力），满足实时自回归生成需求
3. 渐进蒸馏（Progressive Distillation）策略逐步迁移知识，保留长时序连贯性
4. 与 [[Diffusion Forcing]] 思路相关：都涉及在生成过程中处理条件依赖的方向性问题

## 数学形式
$$\mathcal{L}_{LF} = \mathbb{E}[\| f_{\theta}^{causal}(x_{1:t}, a_{1:t}) - f_{\phi}^{bi}(x_{1:T}, a_{1:T}) \|^2]$$

其中 $f^{causal}$ 为因果 student，$f^{bi}$ 为双向 teacher，蒸馏目标对齐两者的中间表示或输出预测。

## 代表工作
- [[ABot-World-0]]: 首次提出 LongForcing，用于实现单 GPU 实时长视频世界模型推理

## 相关概念
- [[Diffusion Forcing]]
- [[DiT]]
- [[SageAttention2]]
