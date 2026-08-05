---
type: concept
aliases: [VLA JEPA, JEPA VLA]
---

# VLA-JEPA

## 定义

将 JEPA（Joint-Embedding Predictive Architecture）引入视觉-语言-动作模型的工作，通过在独立编码器产生的潜变量空间内做动作条件预测来学习动力学表示。

## 核心要点

1. 基于约 2B 参数骨干，无大规模具身数据预训练
2. 使用独立的 JEPA 编码器生成预测目标（非策略骨干本身），可能存在目标-策略不匹配
3. LIBERO 仿真均值 96.1%；LIBERO-Plus 零样本迁移总均值 62.9%
4. 真实 UR5e 实验中 Pick & Place ID 成功率 35%，低于 SG-WAM 的 75%

## 与 SG-WAM 的区别

| 维度 | VLA-JEPA | SG-WAM |
|------|----------|--------|
| 预测目标来源 | 独立 JEPA 编码器 | 策略自身 EMA 拷贝（自引导） |
| 目标对齐 | 可能不匹配 | 天然对齐 |
| 几何监督 | 无 | VGGT 余弦相似度 |
| 参数量 | ~2B | ~0.9B |

## 代表工作

- [[SG-WAM]]: 主要对比基线之一，SG-WAM 在 LIBERO-Plus 和真实实验上均超越 VLA-JEPA

## 相关概念

- [[World Action Model]]
- [[Self-Guided World Predictor]]
- [[Vision-Language-Action Model]]
- [[Stop-Gradient]]
