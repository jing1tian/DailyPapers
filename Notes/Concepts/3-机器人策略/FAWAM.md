---
type: concept
aliases: [Fast World Action Model, FA-WAM]
---

# FAWAM

## 定义
Fast World Action Model：强调推理效率的 WAM 变体，通过减少 future frame 生成步骤或压缩预测范围加速 inference。

## 核心要点
1. 在保持 world model 监督信号的前提下降低推理延迟
2. 通常采用跳帧预测或 latent 压缩
3. 与 [[TurboWAM]]、[[Flash-WAM]] 等同属 WAM 加速赛道

## 代表工作
- [[Flash-WAM]]: 单步 flash attention 加速 WAM
- [[TurboWAM]]: 选择性预测 "what matters" 加速 WAM

## 相关概念
- [[WAM-Survey]]
- [[LaWAM]]
