---
type: concept
aliases: [Visuo-Tactile VLA, 视触觉视觉语言动作模型]
---

# VTLA

## 定义
Visuo-Tactile Vision-Language-Action Model，将触觉传感信号整合进 VLA 框架的多模态机器人策略，同时处理视觉、语言和触觉三种模态。

## 核心要点
1. 标准 VLA 只处理视觉+语言→动作；VTLA 额外引入触觉模态
2. 触觉信号（力、形变、剪切、滑动）对 contact-rich manipulation 至关重要
3. 挑战：触觉传感器噪声大、仿真-真实差距（sim-to-real gap）更严重
4. 与 [[TacWAM]] 配合，可将触觉预测整合进世界模型未来状态估计

## 代表工作
- [[TacWAM]]: 在 WAM 框架中集成触觉预测

## 相关概念
- [[VLA]]
- [[TacWAM]]
- [[DreamTacVLA]]
- [[TacForeSight]]
