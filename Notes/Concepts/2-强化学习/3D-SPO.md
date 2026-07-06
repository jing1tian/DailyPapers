---
type: concept
aliases: [3D Spatial Policy Optimization, 3D-SPO]
---

# 3D-SPO

## 定义
3D Spatial Policy Optimization：用 RL 训练 VLM 做 3D 场景布局生成的策略优化方法，通过多轮 refinement 循环和多维 reward 让模型内化几何约束。

## 数学形式
$$\mathcal{R} = \alpha \cdot R_{\text{format}} + \beta \cdot R_{\text{physical}} + \gamma \cdot R_{\text{perceptual}}$$

其中 $R_{\text{physical}}$ 对物体碰撞穿插和落地约束进行惩罚。

## 核心要点
1. 多轮 refinement：VLM 生成布局 → reward 评估 → 迭代更新，类似 RLHF 的循环
2. reward 三维度：格式准确性（输出可解析）、物理可行性（无穿插/悬空）、感知质量（视觉合理）
3. 相比 SFT：不需要 hard-coded 后处理优化器，几何约束由模型自身习得
4. 基于 Qwen2.5-VL 7B 验证，多轮 refinement 优于 single-step RL

## 代表工作
- [[MetaSpatial]]: 提出 3D-SPO，用于 metaverse 场景的 3D 室内布局生成

## 相关概念
- [[GRPO]]: 同类 RL 策略优化方法
- [[SFT]]: 3D-SPO 的对比 baseline
- [[Qwen2.5-VL]]: 基础模型
