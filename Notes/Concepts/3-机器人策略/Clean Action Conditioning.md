---
type: concept
aliases: [干净动作条件化, Clean-Action Conditioning, Zero-Noise Action Conditioning]
---

# Clean Action Conditioning

## 定义

Clean Action Conditioning 是一种在 [[World Action Model]] 训练中向视频预测分支注入动作语义的技术：将动作 token 在扩散时间步 $t=0$（零噪声）的干净副本单独提供给视频 query 的 cross-attention，但对动作预测 query 屏蔽，从而在不产生目标泄漏的前提下使视频分支真正感知"将要执行的动作是什么"。

## 数学形式

$$
\text{Attention}(\mathcal{Q}_{\tilde{V}},\; K_{\{O,\,\tilde{V},\,A\}},\; V_{\{O,\,\tilde{V},\,A\}})
$$

$$
\text{Attention}(\mathcal{Q}_{\tilde{A}},\; K_{\{O,\,\tilde{A}\}},\; V_{\{O,\,\tilde{A}\}})
$$

其中 $A$ 为干净动作（$t=0$，零噪声），$\tilde{A}$ 为带噪动作，$\tilde{V}$ 为带噪视频 latent，$O$ 为当前观测。

## 核心要点

1. **防止目标泄漏**: 动作预测 query $\mathcal{Q}_{\tilde{A}}$ 无法访问干净动作 $A$，避免模型"走捷径"直接从条件中读取答案
2. **不对称信息流**: 视频 query 可见 $\{O, \tilde{V}, A\}$，动作 query 仅可见 $\{O, \tilde{A}\}$
3. **位置编码对齐**: 干净动作与对应带噪 token 共享时序位置编码，建立语义对应关系
4. **数值验证**: 干净/带噪两路径在各层激活的 NRMSE 最大误差约 $10^{-5}$，余弦距离约 $6\times10^{-11}$，确认信息完全隔离

## 代表工作

- [[SelfWAM]]: 首次提出并系统验证该技术，LPIPS 提升 32.5%、PSNR 提升 12.1%

## 相关概念

- [[World Action Model]]
- [[Mixture-of-Transformers]]
- [[Robot Self-Mask Prediction]]
- [[Fast-WAM]]
