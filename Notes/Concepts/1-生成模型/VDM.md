---
type: concept
aliases: [Video Diffusion Model]
---

# VDM

## 定义
Video Diffusion Model (Ho et al., 2022)：将图像扩散模型扩展到视频域的先驱工作，引入 3D U-Net 对空间和时间维度联合建模。

## 核心要点
1. 将 2D U-Net 扩展为 3D，同时处理空间和时间
2. 联合图像和视频训练，用 classifier-free guidance 控制条件生成
3. 奠定了后续视频扩散模型（LVDM、VDM++）的基础

## 代表工作
- [[DriftWorld]]: 与 VDM 类迭代扩散 WM 对比推理速度

## 相关概念
- [[LVDM]]
- [[CogVideoX]]
