---
type: concept
aliases: [World Action Model, WAM, 世界动作模型]
---

# WAM（World Action Model）

## 定义
World Action Model 是一类具身基础模型，统一了环境动态建模（世界建模）与运动控制（动作生成），对未来状态 $o'$ 和动作 $a$ 的联合分布建模：$p(o', a \mid o, l)$。

## 数学形式

$$
\mathcal{L}_{\text{WAM}} = \mathbb{E}_{(o,l,o',a) \sim \mathcal{D}} \left[ -\log p(o', a \mid o, l) \right]
$$

## 核心要点
1. **两个核心标准**: 前向预测建模（必须对未来状态 $o'$ 生成可量化表示）+ 耦合动作生成（动作必须与预测未来状态严格对齐）
2. **两大范式**: Cascaded WAM（$p(o',a|o,l) = p(a|o',o,l) \cdot p(o'|o,l)$）和 Joint WAM（直接建模联合分布）
3. **数据优势**: 能同时消化有动作标注的 $(o, a, o')$ 三元组和无动作标注的视频 $(o, o')$ 对

## 代表工作
- [[WAM-Survey]]: 首篇系统性 WAM 综述
- [[GigaWorld]]: Joint WAM 代表
- [[UniPi]]: Cascaded WAM 早期基础工作
- [[Flash-WAM]]: 通过模态感知一致性蒸馏实现 WAM 单步推理（23× 加速），首个系统解决 WAM 推理延迟问题的工作
- [[MotionWAM]]: 双 DiT 耦合架构，固定流时间步提取 Video DiT 中间特征驱动 Motion DiT，4.9 Hz 实时全身人形机器人控制，+32% vs GR00T-N1.7

## 相关概念
- [[VLA]]: WAM 的前驱，标准 VLA 不建模世界动态
- [[World Model]]: WAM 的另一基础
- [[Cascaded WAM]]: WAM 的一种架构范式
- [[JEPA]]: 预测潜表示方法，影响 VLA-JEPA 等 WAM 工作
