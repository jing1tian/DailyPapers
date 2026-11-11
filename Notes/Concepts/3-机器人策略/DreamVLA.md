---
type: concept
aliases: [DreamVLA]
---

# DreamVLA

## 定义

通过生成人体运动轨迹引导仿真 RL 专家训练，再以参考条件 Recovery DAgger 蒸馏为 [[GR00T N1]] VLA 策略，无需新的遥操作数据即可完成人形机器人全身操作任务的端到端流水线。

## 核心要点

1. 生成式演示流水线：[[OmniControl]] 生成受时空关节约束的人体运动 → [[OmniRetarget]] 重定向 → [[HumanoidMimicGen]] 增强接触点
2. 残差 RL 专家：在 [[SONIC]] 名义 token 上叠加可学习残差修正 $\Delta z_t$，保留运动自然性同时大幅提升任务成功率（Picking: 0% → 88.6%）
3. Recovery DAgger：从失败学员状态以相同参考后缀重启专家，只聚合成功恢复片段，解决接触敏感任务中的分布偏移问题

## 代表工作

- [[DreamVLA]]: 原始论文 (Park, Rasyad, Kalaria, Sastry, CoRL 2026 Workshop)，在仿真四任务上验证完整流水线

## 相关概念

- [[VLA（视觉-语言-动作模型）]]
- [[SONIC]]
- [[DAgger]]
- [[Residual Policy]]
- [[OmniControl]]
- [[Whole-Body Control]]
- [[Loco-Manipulation]]
