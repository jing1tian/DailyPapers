---
type: concept
aliases: []
---

# SparseFusion

## 定义
稀疏视角（sparse-view）3D 重建方法：将预训练的视角条件扩散模型（view-conditioned diffusion）蒸馏为一个 3D 一致的表示，用极少数输入视角（如 2-3 张）生成新视角，缓解传统多视角重建在视角稀疏时几何歧义严重的问题。

## 核心要点
1. 核心思路是"蒸馏"——把 2D 扩散模型对未见视角的生成能力压缩进显式/隐式 3D 表示，而不是直接做多视角几何三角化
2. 解决的是经典 NeRF/3DGS 类方法在输入视角极少时容易过拟合、几何坍缩的问题
3. 常被后续单目/稀疏视角 4D 重建工作当作 3D 先验来源或对比 baseline

## 代表工作
- [[Lift4D]]: 在方法对比中提及 SparseFusion 一类"蒸馏视角条件扩散模型"的稀疏视角重建思路

## 相关概念
- [[3D Gaussian Splatting]]
- [[Shape of Motion]]
