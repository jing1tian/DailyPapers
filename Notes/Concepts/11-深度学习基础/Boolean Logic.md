---
type: concept
aliases: [布尔运算, Boolean Operations]
---

# Boolean Logic（布尔逻辑）

## 定义

基于真/假（0/1）的逻辑运算体系，包括 AND（与）、OR（或）、NOT（非）等基本运算，在 Conceptor 代数中被扩展为软版本以处理连续子空间。

## 核心要点

1. 标准布尔逻辑作用于 {0,1} 上；[[Conceptor Boolean Logic]] 将其推广到 [0,1] 矩阵上
2. Conceptor 布尔运算保留了布尔代数的代数性质（如 De Morgan 定律）
3. COAST 中核心用途：$C_{\text{steer}} = C_{\text{success}} \land \neg C_{\text{failure}}$

## 相关概念

- [[Conceptor Boolean Logic]]
- [[Conceptor]]
