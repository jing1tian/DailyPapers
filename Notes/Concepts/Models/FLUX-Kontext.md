---
type: concept
aliases: [FLUX.1 Kontext, Kontext]
---

# FLUX-Kontext

## 定义
Black Forest Labs 发布的图像编辑模型，基于 Flow Matching 框架，支持以参考图像为上下文条件对输入图像进行语义编辑，能够精确修改图像内容同时保留无关区域的视觉一致性。

## 数学形式
基于 Flow Matching 目标（而非传统 DDPM），在 latent 空间训练：
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\left[\|v_\theta(\mathbf{x}_t, t, \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\|^2\right]$$
其中 $\mathbf{c}$ 为条件图像的 latent token 拼接作为上下文。

## 核心要点
1. **In-context 编辑**: 将输入图像的 latent token 直接拼接到去噪过程中，实现结构感知编辑
2. **Flow Matching 框架**: 相比 DDPM 采样路径更直，推理步骤更少
3. **DiT 架构**: 基于扩散变换器，支持高分辨率图像处理
4. **机器人适配**: 可通过 LoRA 微调适配机器人操作数据，用于关键帧预测

## 代表工作
- [[SWEET]]: 用 FLUX-Kontext LoRA 微调作为稀疏视觉规划器，预测操作任务关键帧

## 相关概念
- [[扩散变换器]]
- [[LoRA]]
- [[图像编辑世界模型]]
- [[FLUX.md|FLUX]]
