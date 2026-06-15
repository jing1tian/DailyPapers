---
type: concept
aliases: [Latent Plan, Neural State Abstraction Planning]
---

# LatPlan

## 定义
通过 VAE 将高维图像观测映射到离散潜在状态空间，再在该空间中执行符号规划（如 BFS/A*），最后将规划路径解码回图像的方法。

## 核心要点
1. 用 VQ-VAE 或 β-VAE 将图像离散化为有限状态集合
2. 在离散状态图上构建转移模型，支持经典规划算法
3. 把连续感知与符号规划解耦，泛化性强但依赖状态空间覆盖完整性

## 代表工作
- Asai & Fukunaga (2018): 原始 LatPlan，AAAI
- [[STRIPS-WM]]: 基于 FSQ 的改进版本，学习 STRIPS 风格命题

## 相关概念
- [[FSQ]]
- [[符号规划]]
- [[世界模型 (World Model)]]
