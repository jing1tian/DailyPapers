---
type: concept
aliases: [ViewConcat]
---

# ViewConcat

## 定义
WALL-WM 中的多视角特征融合模块，将多个相机视角的特征沿通道维度拼接后输入 Transformer，实现多视角感知。

## 核心要点
1. 简单高效的多视角融合：直接拼接而非 cross-attention
2. 配合 RoPE 位置编码处理不同视角的空间关系
3. 在 event-grounded WAM 框架中处理多相机输入

## 代表工作
- [[WALL-WM]]: ViewConcat 作为多视角融合组件

## 相关概念
- [[WALL-WM]]
- [[RoPE]]
