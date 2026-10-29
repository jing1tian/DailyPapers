---
type: concept
aliases: [三分支混合专家Transformer, Tri-branch Mixture-of-Transformer]
---

# Tri-MoT（三分支 Mixture-of-Transformer）

## 定义

Tri-MoT 是 [[TraceWAM]] 提出的三分支 [[MoT|Mixture-of-Transformer]] 架构，包含视频预测专家、三维运动预测专家和动作生成专家，通过结构化注意力让运动监督塑造动作表征，同时确保推理时动作专家不依赖运动分支。

## 架构要点

1. **三个独立专家**: 视频专家（V）、运动专家（M）、动作专家（A），参数各自独立
2. **结构化注意力掩码**:
   - 当前视频 token：仅自注意力（因果）
   - 未来视频/运动/动作 token：各自与本组 + 当前视频交互
   - **关键约束**：动作 token 不与运动 token 交互
3. **推理解耦**：训练时三分支并行；推理时仅保留当前视频 + 动作分支，运动分支完全移除

## 核心设计

```
训练时:
  当前 RGB → [视频 VAE] → 当前视频 token
  未来 RGB → [视频 VAE] → 未来视频 token  (视频专家)
  运动标注 → [B-spline] → 运动 token       (运动专家)
  → 动作 token                              (动作专家)
  [动作 token 不看运动 token]

推理时:
  当前 RGB → [视频 VAE] → 当前视频 token → 动作 token → 动作
```

## 符号说明

- 隐藏维度：1,024
- 动作/运动 Horizon：$H_A = H_M = 32$
- 视频下采样：4× → 9 帧/chunk
- 运动专家 token：400 个（20×20 网格，每 token 打包 4 条轨迹）

## 代表工作

- [[TraceWAM]]: 提出 Tri-MoT，在 LIBERO-Plus（OOD）上比 Fast-WAM 提升 18.3%

## 相关概念

- [[MoT]]: Mixture-of-Transformer 基础架构
- [[MoE]]: Mixture-of-Experts 通用机制
- [[World Action Model]]: WAM 框架
- [[Flow Matching]]: 三分支均采用流匹配训练目标
