---
type: concept
aliases: [潜在世界动态, 世界动态表征, Latent World Model, φ]
---

# Latent World Dynamics

## 定义

在机器人策略学习中，**潜在世界动态**（Latent World Dynamics）是指在**潜在空间**中对环境未来状态演化的紧凑表征 $\varphi$，捕捉动作相关的物理因果信息，而非原始像素层面的重建。

## 数学形式

$$
\varphi = f_\text{IDM}(o_t, o_{t+1}, \ldots, o_{t+k})
$$

在策略分解中：

$$
\pi_\theta(\varphi, a_{t:t+H} \mid o_t, \ell) = \pi_\theta(a_{t:t+H} \mid o_t, \varphi)\; \pi_\theta(\varphi \mid o_t, \ell)
$$

## 核心要点

1. **预测性**: 潜在世界动态编码未来状态的语义信息（而非像素重建），可从初始观测预测任务指令（Lumo-2 中达 ~90% 准确率）
2. **跨具身对齐**: 相同操作语义在不同机器人平台（甚至人手）中产生相似的潜在表征，涌现出跨具身语义聚类
3. **双向约束**: 世界动态引导动作生成（$\varphi \to A$），动作也反过来正则化世界动态学习（$A \to \varphi$）
4. **提取方式**: 通常通过 [[Inverse Dynamics Model|逆动力学模型（IDM）]] 从相邻帧特征差异中提取

## 代表工作

- [[Lumo-2]]: 通过 IDM + VQ Codebook 提取潜在世界动态，三阶段渐进对齐
- [[Fast-WAM]]: 世界动作模型，显式建模世界动态用于动作条件生成

## 相关概念

- [[Inverse Dynamics Model]]: 提取潜在世界动态的核心工具
- [[Vector Quantization]]: 将连续潜在动态量化为离散码本
- [[Fast-WAM]]: 世界动作模型框架
- [[Latent-Action]]: 动作的潜在表征，与世界动态协同对齐
