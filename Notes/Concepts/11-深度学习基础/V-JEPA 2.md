---
type: concept
aliases: [V-JEPA2, Video JEPA 2]
---

# V-JEPA 2

## 定义
Meta 提出的第二代视频 Joint-Embedding Predictive Architecture，在 action-relevant latent 表示方面优于 VideoMAE V1 等方法，时序因果性是其 action-relevant 结构形成的关键。

## 核心要点
1. 基于 JEPA 框架（LeCun 提出），在 latent space 做预测而非像素空间
2. 维持时序因果性，latent 表示对动作预测高度相关
3. 被证明是目前最适合作为 WAM 视觉骨干的预训练编码器之一

## 代表工作
- Meta FAIR: V-JEPA 2 技术报告
- Yeom et al. (2026，arXiv 2606.07687): 诊断分析证明其 action-relevant 优势

## 相关概念
- [[V-JEPA]]
- [[World Model]]
- [[VideoMAE]]
