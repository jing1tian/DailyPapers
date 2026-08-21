---
type: concept
aliases: [WorldPack, Dynamic Frame Compression WM]
---

# WorldPack

## 定义
基于 Conditional Diffusion Transformer 的长上下文视频世界模型，用 Spatially-Aware Compressed Memory 动态决定历史帧的压缩比，在长时程导航任务中保持时空一致性。

## 核心要点
1. 动态帧压缩：根据场景内容动态分配历史帧压缩率，而非固定比例
2. Spatially-Aware Compressed Memory：保留关键空间位置信息而非均匀压缩
3. 在 LoopNav（室内导航）和 RECON（户外）上验证，有真实环境实验

## 代表工作
- Oshima et al., 2025/2026 — [[WorldPack]] (arXiv 2512.02473, University of Tokyo + Google DeepMind)

## 相关概念
- [[DIAMOND]]
- [[NWM]]
- [[WorldMem]]
- [[FramePack]]
