---
type: concept
aliases: [VACE, Video Adapter for Controlled Enhancement]
---

# VACE

## 定义
视频扩散模型的 adapter 框架，通过轻量级适配器对预训练视频生成模型做条件控制微调。

## 核心要点
1. 冻结预训练视频扩散模型权重，只训练 adapter
2. 支持多种条件控制（reference frame、mask、pose）
3. 计算高效，适合少量数据微调

## 代表工作
- [[ViDiHand]]: 用 VACE adapter 微调视频扩散模型做手势重建

## 相关概念
- [[DiT（扩散变换器）]]
