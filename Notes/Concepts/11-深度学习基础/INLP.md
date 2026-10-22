---
type: concept
aliases: [Iterative Nullspace Projection, 迭代零空间投影]
---

# INLP

## 定义
一种通过反复训练线性分类器并将表征投影到其零空间来移除线性可解码概念的方法，常用于 NLP 公平性/可解释性分析。

## 数学形式
重复执行：
1. 训练线性分类器 $W$ 预测概念 $c$ from 表征 $h$
2. 计算投影矩阵 $P = I - W^T(WW^T)^{-1}W$
3. 更新表征 $h \leftarrow Ph$

$$h_{clean} = P_n \cdots P_2 P_1 h$$

## 核心要点
1. 每次迭代移除一个线性方向的概念信息
2. 保证移除后的表征对该概念线性不可解码
3. 在 VLA 失效分析中用于解耦视觉腐蚀特征的可归因性
4. 前提假设：概念信息线性编码在表征中

## 代表工作
- INLP: Ravfogel et al., 2020 (EMNLP)

## 相关概念
- [[DINOv2]]
- [[SigLIP]]
