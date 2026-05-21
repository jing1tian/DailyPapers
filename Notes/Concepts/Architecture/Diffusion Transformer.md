---
type: concept
aliases: [DiT, 扩散变换器]
---

# Diffusion Transformer

## 定义
将 Transformer 架构与扩散模型结合的生成网络，用 Transformer Block 替换传统 U-Net 主干，通过自注意力建模全局依赖关系。

## 数学形式
$$
\mathbf{x}_0 = f_\theta(\mathbf{x}_t, t, c)
$$
其中 $f_\theta$ 为 Transformer 架构，$t$ 为扩散时间步，$c$ 为条件输入（语言、图像等）。

## 核心要点
1. 用 Transformer Block（Self-Attention + FFN）替代 U-Net 卷积层，更好捕获长程依赖
2. 时间步 $t$ 和条件 $c$ 通过 AdaLN 或交叉注意力注入
3. 可扩展性强：参数量与性能呈幂律关系

## 代表工作
- [[DiT]]: 最早将 DiT 应用于图像生成，证明 Transformer 替代 U-Net 的有效性
- [[RoVLA]]: 用 32 层 DiT 作为 VLA 动作生成器，结合 Flow Matching
- [[RDT]]: 机器人扩散变换器，用于双臂操作策略

## 相关概念
- [[Diffusion Model]]
- [[Flow Matching]]
- [[Transformer]]
- [[AdaLN]]
