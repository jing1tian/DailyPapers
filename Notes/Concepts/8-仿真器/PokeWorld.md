---
type: concept
aliases: [Poke World, 物理参数可识别性测试环境]
---

# PokeWorld

## 定义
一个交互式受控实验环境，视觉外观相同但物理参数（质量、阻力、接触刚度）不同，用于测试 latent world model 对物理参数的可识别性（identifiability）。

## 核心要点
1. 核心设计：视觉完全相同的对象隐藏不同的物理参数，只有施加扰动（poke）时才能区分
2. 用于回答"WM latent 里到底编码了多少物理量？"这一理论问题
3. 测试对象包括 [[LeWM]]、[[JEPA]]、[[LeJEPA]] 等多种 WM 架构
4. 来自 NYU+CMU+Columbia 的 identifiability 分析论文

## 代表工作
- [[What Can Latent World Models Know]]: 使用 PokeWorld 分析物理参数可识别性

## 相关概念
- [[LeWM]]
- [[LeJEPA]]
- [[JEPA]]
