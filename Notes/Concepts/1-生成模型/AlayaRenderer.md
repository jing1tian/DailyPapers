---
type: concept
aliases: [AlayaRenderer-Flash, G-buffer Generative Renderer]
---

# AlayaRenderer

## 定义
一类 G-buffer 条件化的生成渲染器，接收游戏引擎导出的结构化几何信息（深度、法线、实例 ID）和文字 prompt，输出光照真实的 RGB 帧，用于 interactive world modeling。

## 数学形式
G-buffer 条件生成：
$$I_{\text{RGB}} = f_\theta(G_{\text{depth}}, G_{\text{normal}}, G_{\text{instance}}, \text{prompt})$$

AlayaRenderer-Flash（蒸馏版本）用 Distribution-Matching Distillation 从 30 步压缩到 4 步：
$$\mathcal{L}_{\text{distill}} = D_{\text{KL}}(p_\theta^{\text{4-step}} \| p_{\text{teacher}}^{\text{30-step}})$$

## 核心要点
1. G-buffer 条件化：保留场景结构（不改变 world dynamics），只改变外观
2. 自回归流式生成：chunk-by-chunk 保持时序一致性
3. Distilled tiny codec：比标准 VAE latent 更小，实时推理
4. 与 video WM 的区别：不生成世界状态，只渲染给定状态的外观

## 代表工作
- Lin et al. 2026: AlayaRenderer-Flash (UC Merced + Shanda)
- [[DiffusionRenderer]]: 类似思路的竞争方法

## 相关概念
- [[DiT]]（架构基础）
- [[DiffusionRenderer]]（基线对比）
- [[AlayaWorld]]（同 lab 的 video WM，不同方向）
