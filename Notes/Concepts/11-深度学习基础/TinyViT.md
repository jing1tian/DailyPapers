---
type: concept
---

# TinyViT

## 定义
轻量级视觉 Transformer，通过知识蒸馏从大型 ViT 迁移能力，用于边缘端实时视觉理解。

## 数学形式
$$y = \text{TinyViT}(x) \approx \text{KD}(\text{ViT-L}(x))$$

## 核心要点
1. AffectiveVLA 用它做实时意图识别（板载摄像头）
2. 蒸馏自大型 ViT，适合低算力部署

## 代表工作
- 见关联笔记

## 相关概念
- [[VideoLightFormer]]
