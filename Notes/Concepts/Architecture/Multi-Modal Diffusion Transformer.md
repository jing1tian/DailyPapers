---
type: concept
aliases: [MMDiT, 多模态扩散Transformer, Multi-Modal DiT]
---

# Multi-Modal Diffusion Transformer

## 定义

扩展自 Diffusion Transformer（DiT）的多模态架构，能够同时处理文本、图像、视频等多种模态的输入，通过联合注意力机制实现跨模态条件生成。

## 核心要点

1. 在标准 DiT 基础上引入多模态 token，支持文本/图像/视频的联合建模
2. 采用联合自注意力（Joint Self-Attention）使不同模态 token 相互感知
3. 常见于高质量视频生成和全景合成任务
4. 与 Cross-Attention 条件注入不同，MMDiT 让文本和图像 token 直接交互

## 代表工作

- [[HYWorld2]]: HY-Pano 2.0 使用 MMDiT 实现透视图到等矩形全景的隐式映射
- Stable Diffusion 3: 首次提出 MMDiT 架构

## 相关概念

- [[Video Diffusion Transformer]]
- [[3D Gaussian Splatting]]
