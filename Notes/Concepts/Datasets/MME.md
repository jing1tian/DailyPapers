---
type: concept
aliases: [MME, MME Benchmark]
---

# MME

## 定义
多模态大模型（MLLM）的综合评测 benchmark，覆盖感知（existence、count、position、color）和认知（commonsense、numerical、text、code）共 14 个子任务。

## 核心要点
1. 使用是/否二分类问题，标准化评分（每题 0/1），总分 2800
2. 分感知分数（Perception）和认知分数（Cognition）两维度
3. 快速评估 MLLM 的基础视觉理解能力
4. 常用于对比 VLA 训练前后 VLM 能力保留情况

## 代表工作
- Fu et al. (2023): "MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models"

## 相关概念
- [[MMMU]]
- [[VLM]]
