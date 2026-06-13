---
type: concept
aliases: [Representation Visual Tokenizer, 表示视觉Tokenizer]
---

# RepViTok

## 定义
由 [[RepWAM]] 提出的表示中心化视觉-动作 tokenizer，用对比学习和预测目标替代像素重建目标，学习与机器人控制任务相关的紧凑语义表示。

## 数学形式
训练目标由重建损失替换为表示对齐损失：
$$\mathcal{L} = \mathcal{L}_\text{contrastive}(z_v, z_a) + \mathcal{L}_\text{predictive}(z_t, z_{t+1})$$
其中 $z_v$ 和 $z_a$ 分别为视觉和动作的潜表示。

## 核心要点
1. **放弃像素重建**：不以 MSE/LPIPS 为优化目标，降低对纹理/背景的敏感性
2. **控制相关性**：学习的表示更紧凑，聚焦于与动作执行相关的场景信息
3. **与 ViTok 对比**：[[ViTok]] 重建取向，RepViTok 表示取向

## 代表工作
- [[RepWAM]]: 提出论文，在 AgiBot/RoboTwin 基准上验证

## 相关概念
- [[ViTok]]
- [[Latent Action]]
- [[Visual Tokenizer]]
