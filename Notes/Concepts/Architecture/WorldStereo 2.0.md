---
type: concept
aliases: [WorldStereo2, 世界立体扩展, Memory-Driven World Expansion]
---

# WorldStereo 2.0

## 定义

HY-World 2.0 的记忆驱动世界扩展模块，在关键帧潜空间中操作，通过全局几何记忆（GGM）和空间立体记忆（SSM++）生成多视角一致的关键帧序列。

## 核心要点

1. **Keyframe-VAE**：仅做空间压缩（无时间压缩），保留高频细节，减少大视角变化下的伪影
2. **GGM（Global-Geometric Memory）**：利用全景点云维护全局 3D 几何结构一致性
3. **SSM++（Spatial-Stereo Memory++）**：检索相关参考关键帧，提供细粒度外观一致性
4. 相机控制：7D 向量（四元数 + 平移）编码位姿，Plücker 光线引导几何感知
5. 蒸馏：通过 [[Distribution Matching Distillation|DMD]] 压缩为 4 步快速推理
6. 三阶段训练：Domain-Adaptation → Middle-Training → Post-Training（DMD）

## 代表工作

- [[HYWorld2]]: HY-World 2.0 四阶段流水线的第三阶段

## 相关概念

- [[Video Diffusion Transformer]]
- [[Distribution Matching Distillation]]
- [[Rotary Position Encoding]]
- [[WorldNav]]
- [[WorldMirror 2.0]]
