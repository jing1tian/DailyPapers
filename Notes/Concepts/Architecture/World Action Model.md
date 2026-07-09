---
type: concept
aliases: [WAM, World Action Models, 世界动作模型]
---

# World Action Model (WAM)

## 定义

World Action Models (WAMs) 是统一了环境动力学建模（世界建模）与运动控制（动作生成）的**具身基础模型**，目标是联合建模未来状态与动作的分布 $p(o', a | o, l)$，而非仅预测动作 $p(a | o, l)$。

## 数学形式

$$
\mathcal{L}_{\text{WAM}} = \mathbb{E}_{(o,l,o',a) \sim \mathcal{D}} \left[ -\log p(o', a | o, l) \right]
$$

WAM 两大范式的分解：
- **Cascaded WAM**（显式分解）：$p(o', a | o, l) = p(a | o', o, l) \cdot p(o' | o, l)$
- **Joint WAM**（联合建模）：$p(o', a | o, l)$ 在共享表示空间中联合优化

## 核心要点

1. **前向预测建模**：必须生成或利用未来状态 $o'$ 的可量化表示（显式像素预测或隐式潜在表示）
2. **耦合动作生成**：动作必须严格与预测的未来状态对齐
3. **数据灵活性**：可同时处理带动作标注的三元组 $(o_t, a_t, o_{t+1})$ 和无动作标注的视频 $(o_t, o_{t+1})$

## 与相关概念的区别

| 概念 | 目标分布 | 特点 |
|------|---------|------|
| [[VLA]] | $p(a \| o, l)$ | 反应式映射，无未来预测 |
| [[World Model]] | $p(o' \| o, a)$ | 动作条件化，无动作生成 |
| WAM | $p(o', a \| o, l)$ | 统一预测与生成 |
| Video Policy | $p(a \| o)$ | 结构遗产于视频生成骨干，可无预测承诺 |

## 代表工作

- [[WAMSurvey]]: 首篇系统性综述，提出正式定义与分类法
- [[WorldModelRoadmap]]: 将 WAM 定位为世界模型向具身决策演化的关键桥接，纳入世界模型统一架构趋势
- **Cascaded WAM**: UniPi, VLP, AVDC, VPP, LAPA, S-VAM
- **Joint WAM (Auto-regressive)**: GR-1, GR-2, CoT-VLA, WorldVLA, VLA-JEPA
- **Joint WAM (Diffusion)**: PAD, UWM, DreamZero, Cosmos Policy, FLARE, Motus, [[JOPAT]]
- **WAM 推理效率**: [[SANTS]] — 状态自适应调度器，动态决定最优去噪深度，延迟降低 ~80%
- **WAM 推理效率**: [[Flash-WAM]] — 模态感知一致性蒸馏，23× 加速（8.1s → 348ms），动作流线性参数化解决梯度消失
- **WAM 表征增强**: [[GeoSem-WAM]] — 引入几何（深度）和语义（分割）辅助监督分支，提升 latent 表征质量
- **WAM 力感知**: [[FAWAM]] — 三层次（感知/预测/执行）力信号融合，联合预测动作与力轨迹，在线残差修正，接触密集型操作成功率 85%

## 相关概念

- [[VLA|Vision-Language-Action Model]]
- [[World Model]]
- [[Action Chunking]]
- [[Inverse Dynamics Model]]
- [[JEPA|Joint-Embedding Predictive Architecture]]
- [[DiT|Diffusion Transformer]]
