---
type: concept
aliases: [GR-3, Fourier GR3, 傅里叶 GR-3]
---

# Fourier GR-3

## 定义

Fourier GR-3 是由傅里叶智能（Fourier Intelligence）研发的全尺寸双臂人形机器人，具备高自由度双臂操作能力，常用于长流程双臂灵巧操作任务的真实场景验证。

## 核心要点

1. **形态**: 全尺寸人形，双臂，支持双手协同操作
2. **应用**: 在机器人策略学习研究中常作为真实部署平台，验证仿真训练策略的迁移效果
3. **数据采集**: 通常通过遥操作（teleoperation）采集演示数据用于模仿学习
4. **动作空间**: 双臂，动作维度约 14（双臂各 7 个关节自由度）

## 代表工作

- [[HiLo-WAM]]: 在 Fourier GR-3 上验证双流 WAM 的真实机器人性能，完成 Build House / Disassemble House / Disassemble Heart 三个长流程双臂任务，成功率 33%，进度评分 6.9/10

## 相关概念

- [[Dual-Arm Robot]]
- [[世界动作模型（WAM）]]
- [[Action Chunking]]
