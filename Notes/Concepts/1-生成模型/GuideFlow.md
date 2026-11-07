---
type: concept
---

# GuideFlow

## 定义
扩散/流模型的能量引导框架，通过在采样步骤注入可微分能量函数实现无训练的条件生成控制。

## 数学形式
$$\hat{x}_0 \leftarrow \hat{x}_0 - \alpha \nabla_{\hat{x}_0} E(f(\hat{x}_0))$$

## 核心要点
1. GuidedWAM 把此思路迁移到 World Action Model 领域
2. 无需重训练，直接在推理时注入目标

## 代表工作
- 见关联笔记

## 相关概念
- [[GuidedWAM]]
- [[Classifier-Free Guidance]]
