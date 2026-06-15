---
type: concept
aliases: [Closed-Loop Affine Activation Editing]
---

# CLAE

## 定义
闭环仿射激活编辑，一种不修改模型权重、直接在推理时编辑中间层激活来改变模型行为的方法。

## 核心要点
1. 用 SAE（Sparse Autoencoder）找到激活空间中的语义方向
2. 仿射变换（affine transformation）修改激活值
3. 闭环：根据当前行为反馈实时调整编辑方向

## 数学形式
$$h' = h + lpha \cdot d_{semantic}$$

其中 $d_{semantic}$ 是目标行为的语义方向向量。

## 代表工作
- USC CLAE 论文（2606.11489）

## 相关概念
- [[Activation Steering]]
- [[Conceptor]]
