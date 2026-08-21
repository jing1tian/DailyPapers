---
type: concept
aliases: [Predictive Driver Model Score, PDMS, EPDMS, Extended PDMS]
---

# PDMS（Predictive Driver Model Score）

## 定义

NAVSIM benchmark 中用于评估自动驾驶规划器质量的综合评分指标，通过模拟器滚出预测驾驶行为，综合衡量安全性、合规性和行驶效率。

## 数学形式

$$
\text{PDMS} = f(\text{NC}, \text{DAC}, \text{EP}, \text{TTC}, \text{Comfort})
$$

各分项均为 0-100 分，PDMS 为加权聚合（具体权重见 NAVSIM 官方文档）。

## 核心要点

1. **NC（No-at-fault Collision）**: 无过失碰撞率，衡量是否与其他交通参与者发生碰撞
2. **DAC（Drivable Area Compliance）**: 可行驶区域合规率，衡量车辆是否保持在合法行驶区域
3. **EP（Ego Progress）**: 自车进度，衡量沿参考路径的行驶进展
4. **TTC（Time To Collision）**: 碰撞时间，衡量与前方障碍物的安全裕量
5. **Comfort**: 舒适度，衡量加速度/急动变化的平滑性
6. **EPDMS**: NAVSIM-v2 的扩展版本，新增 DDC（双重动态合规）、TL（交通灯）、LK（车道保持）、HC（头部约束）、EC（紧急约束）指标

## 代表工作

- [[NAVSIM]]: 提出该 benchmark 和评估指标的工作
- [[DA-WAM]]: NAVSIM-v1 93.7 PDMS（SOTA as of Aug 2026）；NAVSIM-v2 87.7 EPDMS
- [[DiffusionDrive]]: NAVSIM-v1 88.1 PDMS

## 相关概念

- [[NAVSIM]]
- [[BEV]]
- [[TransFuser]]
