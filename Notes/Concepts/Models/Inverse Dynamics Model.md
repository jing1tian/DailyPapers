---
type: concept
category: Models
tags: [robot-learning, action-labeling]
created: 2026-05-09
---

# Inverse Dynamics Model (IDM)

给定相邻两帧观测 $(o_t, o_{t+1})$ 反推动作 $a_t$ 的模型。常用于把无动作标签的视频（含合成视频）转成可训练的 (obs, action) 对。

## 代表工作

- VPT (OpenAI Minecraft)
- [[RLDX-1]]: 用 IDM 给 Cosmos-Predict2 生成的合成视频打动作标签，扩充 GR-1 humanoid 训练数据。
- [[WMRobotSurvey]]: 系统梳理 IDM 式解耦策略范式（UniPi、VidMan、Vidar、TC-IDM），归纳中间表示从像素→潜在→结构化的演化。
- [[SC3-Eval]]: 把 IDM 作为与正向动力学共享 backbone 的联合训练目标，训练时用作物理可行性正则化锚点抑制自回归漂移，推理时复用为逐 chunk 不确定性信号（$U_{\text{chunk}}(t) = \frac{1}{l}\sum \|a_i - \hat{a}_i\|_2$）触发早停。
- [[Lumo-2]]: 使用冻结 DINOv2 + 因果时空 Transformer 构建 IDM，从 5 帧视觉序列中提取[[Latent World Dynamics|潜在世界动态]] $\varphi$，通过 VQ Codebook 量化；IDM 与 action 自编码器协同训练（Stage 1），建立双向引导-正则约束。
