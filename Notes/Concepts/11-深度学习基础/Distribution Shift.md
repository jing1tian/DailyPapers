---
type: concept
aliases: [分布偏移, 分布漂移, Domain Shift, Covariate Shift, OOD Generalization]
---

# Distribution Shift（分布偏移）

## 定义

测试（部署）时的数据分布与训练时不同，导致模型性能下降的现象。机器人学习中常见于：不同实验室环境、新颖物体、新桌面背景等。

## 核心要点

1. **协变量偏移（Covariate Shift）**: 输入分布 $p(x)$ 改变，标签条件分布 $p(y|x)$ 不变
2. **标签偏移（Label Shift）**: $p(y)$ 改变，但 $p(x|y)$ 不变
3. **概念偏移（Concept Drift）**: $p(y|x)$ 本身发生变化
4. **机器人部署挑战**: VLA 策略对背景、光照、物体纹理高度敏感，轻微环境改变即可导致大幅性能下降

## 应对策略

- **数据增强**: 训练时引入多样化视觉扰动
- **Domain Randomization**: 仿真中随机化外观
- **策略引导 ([[Policy Steering]])**: 部署时通过价值函数筛选鲁棒动作
- **微调**: 收集少量目标域数据进行适配

## 代表工作

- [[DreamSteer]]: 通过 latent world model + value model 引导，在不同实验室环境（OOD）中将 VLA 成功率从 23.75% 提升至 66.25%

## 相关概念

- [[Policy Steering]]
- [[VLA（视觉-语言-动作模型）]]
- [[行为克隆]]
