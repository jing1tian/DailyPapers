---
type: concept
aliases: [Hunyuan3D, 混元3D]
---

# Hunyuan3D

## 定义

Hunyuan3D 是腾讯发布的大规模 3D 资产生成模型，能够从文字或图像描述自动生成高质量 3D 物体网格，在 GRAIL 中作为物体资产来源之一。

## 核心要点

1. 支持文字 / 图像条件的 3D 网格生成
2. 生成的网格可直接用于物理仿真（simulator-ready）
3. 在 GRAIL 中与 Robocasa、ComAsset、OMOMO 共同构成 1,000 个 3D 物体资产库
4. 为 GRAIL 提供了超出传统数据集的物体多样性

## 数学形式

Hunyuan3D 属于生成式 3D 模型，通常基于扩散过程或隐式场（如 NeRF/SDF）生成 3D 表示。

## 代表工作

- [[GRAIL]]: 使用 Hunyuan3D 生成的 3D 物体资产扩充场景多样性

## 相关概念

- [[Robocasa]]
- [[NeRF]]
- [[视频基础模型]]
