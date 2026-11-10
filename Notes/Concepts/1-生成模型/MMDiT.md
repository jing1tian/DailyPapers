---
type: concept
aliases: [MMDiT, Multimodal Diffusion Transformer, 多模态扩散变换器]
---

# MMDiT

## 定义

MMDiT（Multimodal Diffusion Transformer）是一种将多模态条件（视觉、语言、本体感知）与扩散/流匹配动作生成统一在 Transformer 框架中的动作头架构，通过对噪声动作块进行去噪生成连续控制信号。

## 数学形式

流匹配（JiT clean-sample 参数化）:
$$\mathcal{L}_{masked} = \frac{\sum_{h=1}^{H}\sum_{j=1}^{d_{max}} m_j \,\ell_{h,j}}{H \cdot \sum_j m_j}$$

其中 $m_j$ 为有效通道掩码，$d_{max}=20$（双臂各 10D）。

## 核心要点

1. 以 [[Flow-Matching]] 为训练目标，JiT clean-sample 参数化将预测的 $\hat{a}_0$ 转换为速度，使用截断分母 $t_\varepsilon=0.05$ 保证数值稳定。
2. 遮蔽损失设计支持不同机器人平台的异构动作维度（部分通道无效时不计入损失）。
3. 与骨干 LLM 共享权重，条件信息通过 cross-attention 注入。
4. 在 MR0 中用于中间训练（连续控制）阶段，替换预训练的 FAST 离散头。

## 代表工作

- [[MR0]]: 用于桌面操作 VLA 的连续动作头，支持 12 个机器人平台

## 相关概念

- [[Flow-Matching]]
- [[Action Chunking]]
- [[FAST]]
- [[DiT（扩散变换器）]]
