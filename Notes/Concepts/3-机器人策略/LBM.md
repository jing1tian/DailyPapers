---
type: concept
aliases: [Large Behavior Model, LBM]
---

# LBM

## 定义

LBM（Large Behavior Model）是 Toyota Research Institute 提出的大规模行为克隆基础模型，采用 Diffusion Transformer 架构在大规模多任务机器人数据上训练，是当前开源 VLA/行为克隆领域的重要基线之一。

## 核心要点

1. **架构选择**: 基于 DiT（Diffusion Transformer），训练数据规模和 batch size（2560）是社区参考的重要先例
2. **闭源细节**: 高层设计公开发表，但精细实现细节、训练代码基本未开源，社区难以复现其全部设计选择
3. **行业基线意义**: 常被后续工作（如 [[ABC]]）用作 batch size、模型规模、计算分配等设计选择的对比对象

## 代表工作

- [[ABC]]: 将 LBM 的 batch size（2560）选择与 OpenVLA（2048）、GR00T N1（16,384）对比，系统性研究 batch size 对行为克隆性能的影响

## 相关概念

- [[Diffusion Transformer (DiT)|DiT]]
- [[行为克隆]]
- [[VLA（视觉-语言-动作模型）]]
- [[OpenVLA]]
