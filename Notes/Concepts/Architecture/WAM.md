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
- [[WorldPilot]]: 将 WAM 输出（场景演化潜在 + 预期轨迹）通过双路径注入 VLA，在 LIBERO-Plus 达到 84.7% SOTA
- [[Kairos]]: Joint WAM 代表，用 Mixture-of-Transformers（Video DiT + Action DiT）联合建模未来观测与动作，并以 Hybrid Linear Temporal Attention 替换标准全注意力实现线性复杂度长时程推理
- [[ABot-M0.5]]: 通过中间潜在动作（ALAM）+ 双层 MoT + Dream Forcing 解决 WAM 在移动操纵中的三大结构失配，RoboCasa365 SOTA 46.6%
- [[EgoWAM]]: 系统比较三种世界表征（Pixel/DINO/3D Flow）在野外第一视角人类数据 co-training 中的迁移效果，DINO OOD +4×，3D Flow 域内 +20-30%
- [[BadWAM]]: 首次揭示 WAM 特有的 [[World-Action Drift]] 漏洞，黑盒查询攻击可将 LIBERO 成功率从 96.5% 降至 43.1%
- [[GeoBoN]]: 利用冻结 VGGT 的跨视角深度重投影一致性对 WAM 候选 rollout 进行无训练评分与选择，在 5 个 benchmark-WAM 配置中实现测试时成功率提升

## 相关概念
- [[VLA]]: WAM 的前驱，标准 VLA 不建模世界动态
- [[World Model]]: WAM 的另一基础
- [[Cascaded WAM]]: WAM 的一种架构范式
- [[JEPA]]: 预测潜表示方法，影响 VLA-JEPA 等 WAM 工作
- [[World-Action Drift]]: WAM 特有的对抗攻击漏洞——动作输出可在想象未来保持合理时被单独劫持
