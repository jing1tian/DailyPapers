---
type: concept
aliases: [PhyWorld, 物理世界 Benchmark]
---

# PhyWorld

## 定义

PhyWorld 是一个评估视频世界模型物理规律遵守能力的 benchmark，通过测量生成视频中的物理异常比率（Abnormal Ratio）和视频质量（FVD）来衡量模型对真实物理动态的理解程度。

## 核心要点

1. **OOT（Out-of-Template）设置**: 测试分布外场景，考察模型的物理泛化能力
2. **IT（In-Template）设置**: 测试分布内场景，考察训练场景的物理一致性
3. **评估指标**:
   - **FVD（Fréchet Video Distance）**: 生成视频与真实视频的分布距离，越低越好
   - **Abnormal Ratio**: 生成视频中出现物理违规（穿透、违反重力等）的帧比例，越低越好
4. **难点**: OOT 设置更难，模型需要真正理解物理动力学而非记忆训练分布

## 代表工作

- [[NextForcing]]: Next Forcing 在 PhyWorld OOT 异常率从 12% 降至 8%，FVD 从 5.3 降至 4.7

## 相关概念

- [[FVD]]: PhyWorld 的主要量化指标之一
- [[World Action Model]]: PhyWorld 主要评测的模型类型
