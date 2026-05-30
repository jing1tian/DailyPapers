---
type: concept
aliases: [Feature-wise Linear Modulation, 特征线性调制]
---

# FiLM (Feature-wise Linear Modulation)

## 定义
一种条件特征调制方法，通过学习 scale（γ）和 shift（β）参数对特征图做逐元素线性变换，实现对主网络的外部条件控制。

## 数学形式
$$\text{FiLM}(x | c) = \gamma(c) \odot x + \beta(c)$$

其中 $x$ 是待调制的特征，$c$ 是条件输入（如语言、传感器读数），$\gamma, \beta$ 由辅助网络从 $c$ 生成。

## 核心要点
1. 比直接 concatenate 条件更参数高效，比 cross-attention 计算更轻
2. 适合将外部信号（语言指令、任务嵌入、传感器状态）注入中间表示
3. [[VLAConf]] 用 FiLM 将置信度相关特征注入 VLA 推理过程
4. 最早用于视觉问答（VQA）的语言条件特征调制

## 代表工作
- [[VLAConf]]: 用 FiLM 融合置信度特征
- Perez et al. (2018): 原始 FiLM 论文（VQA 任务）

## 相关概念
- [[VLA（视觉-语言-动作模型）]]
- [[ControlNet]] — 另一种条件控制机制
