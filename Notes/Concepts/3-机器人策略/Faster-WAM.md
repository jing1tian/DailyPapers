---
type: concept
aliases: [Faster World Action Model]
---

# Faster-WAM

## 定义
Faster-WAM：通过时序跳帧预测或特征复用加速 WAM 推理的变体，属于 WAM inference efficiency 赛道。

## 核心要点
1. 利用帧间时序冗余，对相似帧复用 KV cache 或跳过 full prediction
2. 在 LIBERO/RoboTwin 等 benchmark 上测试推理速度提升
3. 与 [[FAWAM]]、[[TurboWAM]]、[[Flash-WAM]] 同属 WAM 加速方向

## 相关概念
- [[WAM-Survey]]
- [[TurboWAM]]
- [[Flash-WAM]]
