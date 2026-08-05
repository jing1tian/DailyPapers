---
type: concept
aliases: [Dock of Transformer, DoT架构]
---

# DoT (Dock of Transformer)

## 定义
Faster-WAM 提出的视频中心 WAM 设计原则：把预训练视频 Transformer 作为固定表征枢纽（hub），轻量独立的动作解码器从外部对接（dock）视频 backbone，而非嵌入其内部各层。

## 核心要点
1. 视频 backbone 作为只读表征 hub，不参与动作解码的反向传播（可选冻结）
2. 动作解码器仅访问 backbone 的最终输出 token，深度可远小于 backbone
3. 与 [[MoT]]（Mixture-of-Transformers）和 shared-backbone WAM 的核心区别：后者将动作模块深度绑定到视频 backbone 深度，DoT 完全解耦
4. 解耦带来的收益：推理延迟随动作解码器复杂度而非 backbone 规模增长

## 代表工作
- [[Faster-WAM]]: 提出 DoT，在 RoboTwin + LingBot 上验证

## 相关概念
- [[MoT]]
- [[WAM-Survey]]
