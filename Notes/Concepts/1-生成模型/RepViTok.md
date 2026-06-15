---
type: concept
aliases: [Representation Visual Tokenizer, 表示视觉Tokenizer]
---

# RepViTok

## 定义
由 [[RepWAM]] 提出的表示中心化视觉分词器，采用 Vision Transformer 自编码器，将视频潜变量与冻结视觉基础模型（教师模型）对齐，在重建质量的基础上额外引入语义对齐损失，使视觉 token 具备语义丰富性。

## 数学形式

重建损失：

$$\mathcal{L}_\text{rec} = \lambda_1 \|o - \hat{o}\|_1 + \lambda_\text{perc} \mathcal{L}_\text{perc}(o, \hat{o}) + \lambda_\text{gan} \mathcal{L}_\text{gan}(\hat{o})$$

语义对齐损失：

$$\mathcal{L}_\text{align} = \left\| \text{avg}(W_\text{align} z) - \text{avg}(G(o)) \right\|_2^2$$

其中 $G$ 为冻结的教师视觉基础模型，$W_\text{align}$ 为线性投影层。

## 核心要点
1. **语义对齐而非纯重建**：在标准重建损失之上加入教师模型对齐损失，使 token 语义信息更丰富
2. **时序建模**：初始帧用 $16\times16$ patch，后续帧用 $4\times16\times16$ 时空管元（tubelet）
3. **与 ViTok 对比**：[[ViTok]] 重建取向，RepViTok 表示+重建取向；消融显示 RepViTok gFVD 降低 9.5-13.2%
4. **降低 CFG 依赖**：语义空间已内在语言对齐，推理时无需额外 CFG 外推

## 代表工作
- [[RepWAM]]: 提出论文，在 AgiBot 预训练后于 RoboTwin 2.0 和真实 Franka 机器人上验证

## 相关概念
- [[ViTok]]
- [[Latent Action]]
- [[Visual Tokenizer]]
