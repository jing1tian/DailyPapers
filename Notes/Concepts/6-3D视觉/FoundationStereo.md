---
type: concept
aliases: [Foundation Stereo]
---

# FoundationStereo

## 定义
基础模型级别的立体深度估计方法，在大规模合成数据上预训练，实现跨场景的零样本立体匹配和深度图生成。

## 核心要点
1. 大规模预训练实现强泛化能力
2. 支持零样本迁移到新场景
3. 输出稠密视差图，可转换为深度图
4. 在 DemoBridge 中用于单视角演示的深度估计

## 代表工作
- [[DemoBridge]]：用 FoundationStereo 从单视角视频恢复深度

## 相关概念
- [[MDE]]
- [[DepthSplat]]
