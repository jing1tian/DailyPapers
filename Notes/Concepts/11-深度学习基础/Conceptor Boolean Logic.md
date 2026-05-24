---
type: concept
aliases: [Conceptor 布尔代数, Boolean Conceptor Operations]
---

# Conceptor Boolean Logic（Conceptor 布尔逻辑）

## 定义

Conceptor 矩阵支持的布尔代数运算，允许对子空间进行逻辑组合（AND、OR、NOT），实现子空间的交集、并集和补集。

## 数学形式

**NOT（补集/翻转）**:

$$
\neg C = I - C
$$

**AND（软交集）**:

$$
A \land B = (A^{-1} + B^{-1} - I)^{-1}
$$

**OR（软并集）**:

$$
A \lor B = \neg(\neg A \land \neg B)
$$

## 核心要点

1. NOT 运算翻转子空间：高响应方向变低，低响应方向变高
2. AND 运算计算同时被两个 Conceptor 保留的方向
3. COAST 使用 AND-NOT 构建对比 Conceptor：$C_{\text{steer}} = C_{\text{success}} \land \neg C_{\text{failure}}$
4. 由 Jaeger (2014) 的概念算子代数理论支撑

## 代表工作

- [[COAST]]: 核心利用 AND-NOT 布尔运算构建对比引导 Conceptor

## 相关概念

- [[Conceptor]]
- [[Activation Steering]]
