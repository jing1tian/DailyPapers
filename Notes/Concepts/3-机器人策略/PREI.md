---
type: concept
aliases: [Physical Residual Event Integration, 物理残差事件集成]
---

# PREI（Physical Residual Event Integration）

## 定义

PREI 是一种将异步事件流压缩为三通道物理残差图的事件表征方法，通过指数衰减加权分别捕捉瞬时、显著和持续三种时间尺度的运动信息，在视觉退化条件下提供稳定的结构性表征。

## 数学形式

**活动图计算**（指数衰减加权）：

$$
A_\tau(u) = \sum_{i:(x_i,y_i)=u} \exp\left(-\frac{t - \tau_i}{\tau}\right)
$$

**三通道分解**：

$$
E_t^{\mathrm{prei}} = [E_t^{\mathrm{ins}},\, E_t^{\mathrm{sal}},\, E_t^{\mathrm{per}}] \in [0,1]^{H \times W \times 3}
$$

## 核心要点

1. **瞬时通道** $E_t^{\mathrm{ins}}$：短衰减常数 $\tau_\mathrm{ins}$，反映机器人-物体最近运动
2. **显著通道** $E_t^{\mathrm{sal}}$：用平滑背景归一化，突出局部显著活动
3. **持续通道** $E_t^{\mathrm{per}}$：基于事件计数，RGB 退化时保留短期轮廓
4. 相比[[时间面|时间面（Time Surface）]]，在弱光条件下提供更稳定、更具结构性的表征
5. 通过[[特征蒸馏]]与 VLA 视觉编码器特征空间对齐，降低 domain gap

## 代表工作

- [[EventVLA]]: 提出 PREI 作为事件流表征，在 LL-Severe 条件下相比时间面提升 4.4%（95.6% vs 91.2%）

## 相关概念

- [[事件相机]]: PREI 所处理的传感器输出
- [[时间面]]: 对比基线表征方法
- [[特征蒸馏]]: 用于训练 PREI 编码器的技术
