---
type: concept
aliases: [RDT, Robotics Diffusion Transformer]
---

# RDT

## 定义
一种基于 Diffusion Transformer 的机器人操控基础模型，通过大规模预训练支持多任务、多机器人的通用操控策略。

## 核心要点
1. 以 [[DiT]]（Diffusion Transformer）为骨干，动作生成用扩散过程建模
2. 在大规模异构机器人数据集上预训练，具备跨任务泛化能力
3. 支持语言条件化指令，属于 [[VLA]] 范畴
4. 作为 PAPO-VLA 等后续工作的基础模型对比基线

## 相关概念
- [[Diffusion Policy]]
- [[DiT]]
- [[VLA]]
- [[OpenVLA]]
