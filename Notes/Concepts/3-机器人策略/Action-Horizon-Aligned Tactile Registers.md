---
type: concept
aliases: [动作时间范围对齐的触觉寄存器, tactile registers]
---

# Action-Horizon-Aligned Tactile Registers

## 定义
[[Tactile-WAM]] 中用于将触觉特征压缩为固定数量紧凑 token、并与动作预测时间范围对齐的表示方式，使动作 token 能在生成修正动作的具体时间步读取预测的接触状态。

## 数学形式

触觉 token 总数：

$$
L_{\tau}=ASQ
$$

单个触觉寄存器：

$$
r^{\tau}_{i,s,q}=W_{\tau}\tilde{z}^{\tau}_{i,s,q}+e^{time}_{i}+e^{anchor}_{i}+e^{sensor}_{s}+e^{slot}_{q}
$$

## 核心要点
1. 每个动作块使用 $A$ 个触觉锚点，每个锚点含 $S=2$ 个触觉传感器（左右指），每个 anchor-sensor 对压缩为 $Q$ 个 slot token
2. 触觉特征先经冻结的触觉表征模型（如 [[UniT]]）编码，再用交叉注意力 token 适配器 + 残差 MLP 压缩为固定数量 token
3. 时间、锚点、传感器、槛位四类嵌入分别编码，使触觉寄存器具备完整的位置/身份信息
4. "动作时间范围对齐"是关键设计：寄存器按动作预测的时间步组织，而非按原始触觉采样率组织

## 代表工作
- [[Tactile-WAM]]: 提出该寄存器设计，是 TAAM 之外的关键工程组件

## 相关概念
- [[Touch-Aware Bias]]
- [[UniT]]
- [[Action Chunking]]
