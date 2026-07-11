---
type: concept
aliases: [黑板架构, 黑板系统, Blackboard System, 梯度隔离黑板, Gradient-Isolated Blackboard]
---

# Blackboard Architecture（黑板架构）

## 定义

黑板架构是一种多智能体协作模式，多个独立的"知识源"通过读写共享的"黑板"数据结构进行协作，各知识源之间无直接通信且相互解耦。在[[WP-WM]]中特指**无参数、无梯度**的语义绑定字典，实现物理引擎与语言引擎的解耦通信。

## 数学形式

$$
B: \text{symbol\_id} \to \text{Counter}[\text{label}], \quad |B_\theta| = 0, \quad \nabla B = 0
$$

- **写操作**：$B[s][y] \leftarrow B[s][y] + 1$
- **查询操作**：$\hat{y} = \arg\max_y B[s][y]$

## 核心要点

1. **零参数**：黑板仅为计数字典，无可训练参数，无梯度流
2. **写保护**：物理符号层对语言侧不可见，语言侧只能向黑板写标签
3. **复杂度**：写 $O(1)$，查询 $O(1)$，空间 $O(|\text{符号数}| \times |\text{标签数}|)$
4. **共现学习**：通过多次观测的统计共现实现语义绑定，无需端到端训练

## 代表工作

- [[WP-WM]]: 在语言接地世界模型中实现梯度隔离黑板（97.2% 接地准确率，零参数语义绑定）

## 相关概念

- [[Symbol Grounding]]
- [[Stop-Gradient]]
- [[Discrete Bottleneck]]
- [[GumbelBottleneck]]
