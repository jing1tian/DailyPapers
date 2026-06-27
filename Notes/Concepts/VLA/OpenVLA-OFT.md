---
type: concept
aliases: [OpenVLA-OFT, OpenVLA Optimized Fine-Tuning]
---

# OpenVLA-OFT

## 定义

OpenVLA 的优化微调版本，引入并行动作解码、动作分块（action chunking）、连续动作表示和 L1 回归损失，实现 26× 推理加速并在 LIBERO 上达到 97.1% 平均成功率。

## 核心要点

1. **并行解码**: 替换自回归 token 解码，大幅加速推理
2. **Action Chunking**: 一次预测多步动作序列
3. **L1 回归损失**: 替换交叉熵，适合连续动作空间
4. **LoRA 微调**: 高效参数更新方式

## 代表工作

- [[CapVector]]: 以 OpenVLA-OFT 为基础模型，验证 capability vector 的有效性
- [[PAPO-VLA]]: 以 OpenVLA-OFT 为骨干，在 GRPO 优势估计中加入规划动作因果重要度加权
- [[GRA]]: 以 OpenVLA-OFT 为策略骨干，在其视觉骨干上加入2D几何辅助头，实现几何引导的两阶段合成数据训练框架
- [[PhysReflect-VLA]]: 以 OpenVLA-OFT 为基础策略骨干（Phys-OFT 变体，效果最佳），叠加双向物理一致性筛选和反思引导重采样后真实机器人平均成功率从 82.0% 提升至 85.0%

## 相关概念

- [[VLA]]
- [[LoRA]]
- [[Action Chunking]]
- [[监督微调]]
