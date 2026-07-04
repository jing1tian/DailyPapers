---
type: concept
aliases: [草图上色, 草图着色, Sketch Colouring, Automatic Colorization]
---

# Sketch Colorization（草图上色）

## 定义

给定线稿草图（sketch）和可选的颜色参考，自动生成具有正确颜色和纹理的彩色图像或视频的任务。在 2D 动画制作中，草图上色是将黑白线稿转换为最终彩色帧的关键环节。

## 数学形式

**图像级上色**:
$$
\hat{I}_\text{color} = f_\theta(S, I_\text{ref})
$$

**视频级上色**:
$$
\hat{V} = f_\theta(S_{1:T},\ I_\text{start}) \in \mathbb{R}^{T \times H \times W \times 3}
$$

其中 $S$ 为草图输入，$I_\text{ref}$ 或 $I_\text{start}$ 为颜色参考帧。

## 核心要点

1. **图像级 vs 视频级**: 逐帧独立上色会导致时序闪烁（flickering），视频级方法需维持时序一致性
2. **参考帧引导**: 实际生产中通常给定一帧人工上好色的关键帧，模型自动为后续帧着色
3. **草图生成工具**: 通常用 Anime2Sketch 等工具从彩色帧反向生成草图，用于训练数据构建
4. **主要挑战**:
   - **颜色溢出（colour bleeding）**: 颜色扩散到相邻区域
   - **结构形变**: 输出形状偏离草图轮廓
   - **时序不一致**: 相邻帧颜色突变

## 代表工作

- [[SketchColour]]: 首个基于 DiT 的视频草图上色方法，通道拼接 + LoRA 微调实现 SOTA
- [[AniDoc]]: 基于 ControlNet 的动画草图上色方法
- [[LVCD]]: 基于 ControlNet 的参考帧引导视频上色
- [[ToonCrafter]]: 基于 U-Net 的动画插值/上色方法

## 相关概念

- [[Diffusion Transformer]]: SketchColour 所用骨干架构
- [[ControlNet]]: AniDoc/LVCD 所用控制方法
- [[Video Diffusion Model]]: 视频草图上色的基础技术
- [[Channel Concatenation Control]]: 轻量级草图条件注入方法
