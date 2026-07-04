---
type: concept
aliases: [通道拼接控制, Channel Concat Control, 通道级条件注入]
---

# Channel Concatenation Control（通道拼接控制）

## 定义

一种轻量级的扩散模型条件化方法：将控制信号（如草图、深度图、参考帧）的潜表示与去噪目标的潜表示在**通道维度**拼接后送入主干，通过零初始化扩展 patch embedding projection，以最少可训练参数实现条件控制。

## 数学形式

$$
Z_\text{input} = \text{Concat}\!\left[Z_\text{noisy},\ Z_\text{cond}\right] \in \mathbb{R}^{T \times H \times W \times 2C}
$$

其中 $Z_\text{noisy}$ 为去噪潜变量（$C$ 通道），$Z_\text{cond}$ 为条件信号潜变量（$C$ 通道）；新增通道对应的 patch projection 权重用**零值初始化**，保持初始输出幅度不变。

## 核心要点

1. **零初始化**: 扩展后的 patch embedding projection 对新通道使用零初始化权重，确保训练初期不破坏预训练模型的输出
2. **无额外控制网络**: 与 ControlNet 相比，无需复制整个 U-Net 或 DiT 主干，大幅减少参数量
3. **同一潜空间**: 条件信号和目标信号用同一 VAE 编码，消除了 ControlNet 中的"latent gap"伪影问题
4. **通常配合 LoRA**: 只微调 patch projection 扩展权重 + LoRA 权重，主干参数冻结

## 代表工作

- [[SketchColour]]: 草图引导的 2D 动画上色，用通道拼接替代 ControlNet，仅 1000 万参数实现 SOTA

## 相关概念

- [[ControlNet]]: 对比方法，参数量更大但更通用
- [[LoRA]]: 常与通道拼接控制一起使用的参数高效微调方法
- [[Causal VAE]]: 为通道拼接提供统一潜空间的编码器
- [[Diffusion Transformer]]: 常见的主干架构
