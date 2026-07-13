---
type: concept
aliases: [VLA, Vision-Language-Action Model, 视觉-语言-动作模型]
---

# Vision-Language-Action (VLA)

## 定义
将视觉感知、语言理解与机器人动作输出统一在一个端到端模型中的框架，通常以预训练的视觉-语言模型（VLM）为骨干，输出连续/离散动作序列。

## 数学形式
$$a_t = \pi_\theta(o_t, l)$$

其中 $o_t$ 为视觉观测，$l$ 为语言指令，$a_t$ 为预测动作，$\pi_\theta$ 为 VLA 策略。

## 核心要点
1. 以 VLM（如 LLaVA、Qwen-VL）为骨干，通过在 robot demonstration 数据上微调使其具备动作输出能力
2. 动作头通常输出 7D 末端执行器姿态或关节角度
3. 泛化能力源自 VLM 的大规模预训练，但 sim-to-real gap 和真实数据稀缺仍是主要瓶颈

## 代表工作
- [[OpenVLA]]: 开源 VLA 基准，7B 参数
- [[LingBot-VLA-2.0]]: 多自由度全身操作 VLA
- [[π0]]: 扩散策略 + VLM
- [[InternVLA-A1.5]]: InternVL 骨干的 VLA

## 相关概念
- [[VLM]]
- [[Diffusion Policy]]
- [[WAM]]
- [[Imitation Learning]]
