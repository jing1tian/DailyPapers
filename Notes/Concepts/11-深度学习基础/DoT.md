---
type: concept
aliases: [Dock of Transformer, DoT架构]
---

# DoT (Dock of Transformer)

## 定义
Faster-WAM 提出的视频中心 WAM 设计原则：把预训练视频 Transformer 作为表征枢纽（representation hub），轻量独立的动作解码器通过显式**对接接口（docking interface）**从外部访问骨干**所有层**的表征，而非嵌入骨干内部各层或仅使用最终输出。

## 核心要点
1. 视频 backbone 作为表征 hub，轻量动作头通过对接接口访问骨干**全部 $L_v$ 层**的 Key/Value 缓存（而非仅最终层）
2. 动作头深度与 backbone 深度**完全解耦**：Faster-WAM 用单层动作头对接 30 层视频骨干
3. 与 [[MoT]]（Mixture-of-Transformers）的区别：MoT 要求视频流与动作流逐层一一对应；DoT 无此约束，动作层数可任意
4. 与 H-Bridge（MotuBrain）的区别：H-Bridge 仅允许固定中间层交互；DoT 通过 [[KV-Fusion]] 聚合全部层
5. 解耦带来的收益：推理延迟随动作解码器复杂度（而非骨干规模）增长；多层 KV 融合增强 OOD 泛化

## 代表工作
- [[Faster-WAM]]: 提出 DoT，通过 [[KV-Fusion]] 实例化，在 LIBERO / RoboTwin 2.0 / LIBERO-Plus 上验证

## 相关概念
- [[KV-Fusion]]: DoT 的具体对接机制（通道混合 + 跨层混合）
- [[MoT]]: 深度耦合的对比设计
- [[WAM-Survey]]
