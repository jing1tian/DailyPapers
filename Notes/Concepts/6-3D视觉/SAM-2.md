---
type: concept
aliases: [SAM-2, Segment Anything Model 2, SAM2]
---

# SAM-2

## 定义

SAM-2（Segment Anything Model 2）是 Meta 推出的视频分割基础模型，在 SAM 的基础上扩展到视频域，支持通过点击、框、mask 等 prompt 在视频序列中追踪和分割任意目标。

## 核心要点

1. **视频追踪分割**: 在第一帧 prompt 后自动追踪目标到后续帧，形成一致的时序 mask
2. **内存注意力机制**: 用 memory bank 存储历史帧的特征，为当前帧的分割提供时序上下文
3. **零样本泛化**: 无需目标特定的重新训练，直接迁移到新场景
4. **3DGS 集成**: TrackRef3D 用 SAM-2 追踪多视角 2D 帧中的目标，再映射到 3DGS Gaussian

## 代表工作

- Ravi et al. 2024 (Meta FAIR): SAM-2 原始论文，ICCV 2024 highlight

## 相关概念

- [[SAM]]
- [[3D Gaussian Splatting]]
- [[ReferSplat]]
- [[DEVA]]
