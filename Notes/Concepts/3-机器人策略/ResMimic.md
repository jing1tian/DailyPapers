---
type: concept
aliases: [ResMimic, Residual Mimic]
---

# ResMimic

## 定义

ResMimic 是一种基于残差控制的人形机器人运动模仿方法，通过在预训练控制器上叠加残差策略来追踪参考运动，是 GRAIL 的对比基线之一。

## 核心要点

1. 使用残差策略在预训练控制器上做运动模仿
2. 在 GRAIL 的 loco-manipulation 追踪评测中：成功率 49.2%，物体位置误差 0.393m，MPJPE-L 80.9
3. GRAIL 的 Object-Aware Adaptor 设计思路与 ResMimic 有相似之处（均是残差注入）
4. **GRAIL vs ResMimic**: 81.4% vs 49.2%，GRAIL 领先 32.2 个百分点

## 代表工作

- [[GRAIL]]: 对比基线，GRAIL 通过特权 3D 配置和物体感知适配器大幅超越 ResMimic

## 相关概念

- [[HDMI]]
- [[人形机器人]]
- [[Object-Aware Adaptor]]
- [[近端策略优化]]
