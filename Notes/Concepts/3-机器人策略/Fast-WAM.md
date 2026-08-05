---
type: concept
aliases: [Fast World Action Model]
---

# Fast-WAM

## 定义

Fast-WAM 是一种高效的 World Action Model，以 Wan 2.2-5B Video Diffusion Transformer 为骨干，采用 Mixture-of-Transformers (MoT) 架构，通过单次前向传播实现高效动作生成，无需推理时生成未来视频帧。

## 核心要点

1. 训练阶段进行未来 RGB 视频预测，通过预测性监督学习丰富的场景表示
2. 推理阶段跳过视频生成，仅运行动作分支，效率高
3. Video DiT 与 Action DiT 共享注意力（MoT 架构），动作生成可感知世界模型表示

## 代表工作

- [[GeoSem-WAM]]: 在 Fast-WAM 基础上增加几何和语义辅助监督分支
- [[SelfWAM]]: 在 Fast-WAM 基础上引入干净动作条件化与机器人自我掩码预测，使视频分支真正感知动作语义

## 相关概念

- [[World Action Model]]
- [[Mixture-of-Transformers]]
- [[Video Diffusion Transformer]]
- [[Wan 2.2-5B]]
