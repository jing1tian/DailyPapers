---
type: concept
aliases: [ControlNet, 条件控制网络]
---

# ControlNet

## 定义

Zhang et al. (2023) 提出的条件控制扩散模型范式：复制 U-Net（或 Transformer）编码器部分构成"侧分支"，以 zero-init 卷积与主干相连，在不破坏预训练权重的前提下注入外部控制信号（深度图、边缘、姿态等）。

## 数学形式

侧分支输出 $\mathbf{h}_c$ 通过零初始化卷积加入主干：

$$
\mathbf{h}_{\text{main}} \leftarrow \mathbf{h}_{\text{main}} + \mathcal{Z}(\mathbf{h}_c)
$$

其中 $\mathcal{Z}(\cdot)$ 为零初始化 1×1 卷积，训练初期不干扰主干输出。

## 核心要点

1. **冻结主干**: 主干权重不更新，保留预训练生成先验
2. **侧分支可训练**: 仅侧分支学习如何将控制信号映射为特征调制
3. **灵活接入点**: 可选择性地在特定层注入，控制影响粒度
4. **广泛应用**: 被大量后续工作采用，成为条件视频/图像生成的标准范式

## 代表工作

- ControlNet (Zhang et al., 2023): 原始提出
- [[Mirage]]: 用 ControlNet 风格侧分支将潜空间记忆读出注入 Wan2.2，接入层 {0,4,8,12,16,20,24,28}

## 相关概念

- [[VAE]]
- [[Latent Spatial Memory]]
- [[LoRA]]
