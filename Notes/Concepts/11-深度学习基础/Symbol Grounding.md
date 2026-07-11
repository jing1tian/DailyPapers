---
type: concept
aliases: [符号接地, 语言接地, Symbol Grounding Problem, 语义绑定]
---

# Symbol Grounding（符号接地）

## 定义

符号接地是将抽象离散符号与其所指代的现实世界对象/状态建立稳定对应关系的过程。在机器人学习中，特指让感知系统产生的离散符号（physical symbols）与人类语言标签（linguistic labels）建立一致映射。

## 数学形式

$$
\hat{y} = \arg\max_y P(y \mid s), \quad s = \text{symbol\_id}(o_t)
$$

其中 $o_t$ 为观测，$s$ 为物理引擎产生的离散符号，$\hat{y}$ 为接地后的语言标签。

在 [[WP-WM]] 的黑板实现中：

$$
\hat{y} = \arg\max_y B[s][y]
$$

## 核心要点

1. **经典困境**（Harnad 1990）：纯符号系统内部的符号相互定义，无法自举接地到外部世界
2. **物理优先原则**：物理交互比语言预训练更能建立稳定符号（V-JEPA 28.1% > CLIP 23.1%）
3. **写保护接地**：语言标签绑定到已有符号，而非修改符号本身——分离"建立符号"和"命名符号"两个过程
4. **共现计数**：无参数的黑板字典可通过共现频率实现接地，无需梯度更新

## 代表工作

- [[WP-WM]]: Write-Protected Discrete Bottlenecks for Language-Grounded World Models（2607.08312），提出梯度隔离黑板实现零参数接地
- [[ConceptGraphs]]: 在 3D 语义图中实现开放词汇符号接地

## 相关概念

- [[Discrete Bottleneck]]
- [[Blackboard Architecture]]
- [[Symbol Collapse]]
- [[GumbelBottleneck]]
- [[V-JEPA]]
