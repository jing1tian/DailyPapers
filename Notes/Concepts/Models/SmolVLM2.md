---
type: concept
aliases: [SmolVLM2, SmolVLM 2, 小型视觉语言模型]
---

# SmolVLM2

## 定义
HuggingFace 发布的轻量级开源视觉语言模型（VLM），SmolVLM 的第二版，参数量 2B 以下，适合边缘设备部署和 VLA 骨干用途。

## 核心要点
1. 轻量高效，推理内存友好，适合单 GPU / 机器人板卡
2. [[μ₀]] 用 SmolVLM2 作为 VLM 骨干做 3D 交互轨迹世界模型
3. 与 [[SmolVLA]] 使用的 SmolVLM 属同一系列（SmolVLM → SmolVLM2）
4. HuggingFace 开源，可直接通过 transformers 调用

## 代表工作
- [[μ₀]]：用 SmolVLM2 作为 trace 生成模型骨干

## 相关概念
- [[SmolVLA]]
- [[InternVL3]]
- [[PaliGemma]]
