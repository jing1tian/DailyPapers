---
type: concept
aliases: [Port-Hamiltonian System, PH System]
---

# Port-Hamiltonian 系统

## 定义
一种基于能量的动力系统建模框架，将物理系统的能量结构（Hamilton 量 H）显式编码到状态空间表示中，通过端口（port）描述能量输入/输出，天然保证能量守恒和耗散约束。

## 数学形式
$$
\dot{x} = (J(x) - R(x)) \nabla H(x) + g(x) u
$$
其中：
- $J(x)$ 反对称（能量保守结构矩阵）
- $R(x)$ 半正定（耗散矩阵）  
- $H(x)$ 哈密顿量（总能量）
- $u$ 外部输入（控制信号/端口变量）

## 核心要点
1. 显式编码能量守恒：结构矩阵 J 反对称保证无耗散时能量守恒
2. 耗散矩阵 R 确保物理可实现性（被动系统约束）
3. 模块化：子系统可通过端口互联，保持整体物理一致性
4. 用于 [[ELWM]]：将 Port-Hamiltonian 结构嵌入 latent world model，确保预测轨迹物理合理

## 相关概念
- [[MPC]]
- [[ELWM]]
- [[NTF]]
