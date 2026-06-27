---
type: concept
aliases: [Open VLA, OpenVLA-OFT]
---

# OpenVLA

## 定义
斯坦福开源的 7B 参数 Vision-Language-Action 模型，基于预训练 VLM 微调，支持通用机器人操作。

## 核心要点
1. 将视觉-语言预训练与机器人动作头结合，输出离散化的 7-DoF 动作
2. 在 Open X-Embodiment 数据集上训练，具备跨机器人泛化能力
3. OFT（Orthonormal Fine-Tuning）变体进一步提升微调效率

## 代表工作
- [[OpenVLA]]: Kim et al., 2024, "OpenVLA: An Open-Source Vision-Language-Action Model"
- [[PhysReflect-VLA]]: 以 OpenVLA 作为基础策略骨干（Phys-OVLA 变体），叠加物理可行性筛选和反思机制后真实机器人平均成功率从 74.2% 提升至 79.6%

## 相关概念
- [[VLA]]
- [[LoRA]]
- [[Open-X-Embodiment]]
