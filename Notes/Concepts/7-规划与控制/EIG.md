---
type: concept
aliases: [Expected Information Gain, 期望信息增益]
---

# EIG (Expected Information Gain)

## 定义
主动感知中的目标函数，量化从某个观测位置/动作中可获得的地图/模型不确定性减少量，用于指导机器人选择最有信息价值的下一步动作。

## 数学形式
$$\text{EIG}(a) = \mathbb{E}_{o \sim p(o|a, m)}\left[H(m) - H(m | o, a)\right]$$
其中 $H(m)$ 是当前地图熵，$H(m|o,a)$ 是观测后的条件熵。

## 核心要点
1. 本质是信息论中的互信息（Mutual Information）
2. 高 EIG 的位置能最大程度减少地图不确定性
3. 与安全约束（CBF）结合时存在冲突：信息最丰富的位置往往也是风险最高的位置
4. 常用于 Next-Best-View（NBV）规划和主动建图

## 代表工作
- [[SAFER]]：与 CBF 在 QP 框架下统一求解，平衡感知收益与安全代价

## 相关概念
- [[CBF]]
- [[NBV]]
- [[SLAM]]
