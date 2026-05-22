---
type: concept
aliases: [Self-Forcing, 自强制训练]
---

# Self-Forcing

## 定义
自回归视频生成模型的训练技巧，用模型自身的生成帧作为下一步的输入（而非 GT 帧），消除训练（teacher forcing）与推理（自回归）之间的分布偏移（exposure bias）。

## 数学形式
标准 teacher forcing（训练）：
$$\hat{f}_t = \text{model}(f_{t-1}^{GT}, \ldots)$$
Self-Forcing（训练时模拟推理）：
$$\hat{f}_t = \text{model}(\hat{f}_{t-1}, \ldots), \quad \hat{f}_{t-1} = \text{stopgrad}(\text{model}(\hat{f}_{t-2}))$$

## 核心要点
1. 解决自回归序列模型的 exposure bias 问题
2. 训练时用自身输出替代 GT，消除训练/推理分布差距
3. 与 KV-cache 结合，使长视频生成更稳定
4. 在 LongLive-2.0 中用于将双向 DiT 转为因果 AR 模型

## 代表工作
- [[LongLive-2.0]]：用 Self-Forcing 将预训练双向 DiT 微调为稳定的长视频 AR 生成模型

## 相关概念
- [[DiT]]
- [[CausVid]]
