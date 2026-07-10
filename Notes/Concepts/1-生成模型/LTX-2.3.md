---
type: concept
aliases: [LTX Video 2.3, Lightricks Text-to-Video 2.3]
---

# LTX-2.3

## 定义
LTX-2.3 是 Lightricks 开发的开源视频扩散模型，基于 [[DiT]]（Diffusion Transformer）架构，支持高分辨率（720p）、高帧率（24fps）视频生成，是 [[AlayaWorld]] 等交互式世界模型的微调基础。

## 核心要点
1. **架构**: [[DiT]] 骨干 + [[Video Diffusion Model|视频扩散]] 框架，支持文本、图像条件生成
2. **规模**: 约 15B 参数级别的大型视频生成模型
3. **输出质量**: 支持 720p/24fps 高质量视频生成
4. **开源**: 可作为下游任务（世界模型、交互生成）的微调基础

## 相关特性
- 高效的 [[Causal VAE|因果 VAE]] 视频压缩
- 支持长时序视频生成
- 可与 [[DMD]] 等蒸馏方法结合实现少步推理

## 代表工作
- [[AlayaWorld]]: 基于 LTX-2.3 微调的交互式世界生成系统

## 相关概念
- [[DiT]]
- [[Video Diffusion Model]]
- [[视频生成 (Video Generation)]]
- [[DMD]]
