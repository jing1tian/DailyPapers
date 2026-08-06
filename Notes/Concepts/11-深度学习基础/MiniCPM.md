---
type: concept
aliases: [MiniCPM LLM]
---

# MiniCPM

## 定义
MiniCPM：面壁智能（ModelBest）开发的高效小型语言模型系列，在参数量受限条件下达到接近大模型的性能，适合边缘部署。

## 核心要点
1. 1.2B/2.4B 参数量，显著低于 LLaMA/Qwen 等
2. 采用 WSD（Warmup-Stable-Decay）学习率调度
3. 在 VLA 部署研究中作为轻量 backbone 候选

## 代表工作
- [[PhyAI]]: 用 MiniCPM 作为边缘端 VLA 部署的轻量 backbone

## 相关概念
- [[PhyAI]]
