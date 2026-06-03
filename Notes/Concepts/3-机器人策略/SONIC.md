---
type: concept
aliases: [SONIC]
---

# SONIC

## 定义
通用 Vision-Language-Action 模型，在 LEGS 框架中作为人形机器人 loco-manipulation 的骨干 VLA，通过 3DGS 渲染环境 fine-tune。

## 核心要点
1. 预训练 VLA，支持 fine-tune 适配新具身形态
2. LEGS 使用 SONIC 作为骨干，在 MuJoCo+3DGS 混合环境中训练
3. 无需遥操作数据，靠仿真环境 fine-tune

## 代表工作
- [[LEGS]]: 用 3DGS 世界 fine-tune SONIC 做人形机器人

## 相关概念
- [[LEGS]]
- [[OpenVLA]]
- [[VLA（视觉-语言-动作模型）]]
