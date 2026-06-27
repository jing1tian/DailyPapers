---
type: concept
aliases: [触觉状态代理, 触觉变化代理, touch-state proxy, touch-change proxy]
---

# Touch-State Proxy

## 定义
[[Tactile-WAM]] 中用于约束触觉隐状态的运动衍生辅助监督目标，包含触觉状态代理 $c^\tau$（捕捉局部接触负载）和触觉变化代理 $\Delta c^\tau$（突出阶跃式接触转变）。两者均从触觉图像运动中衍生，**不是**标定力标签，也**不是**额外的观测 token。

## 数学形式

$$
\Delta c_{i}^{\tau}=c_{i}^{\tau}-c_{i-1}^{\tau}
$$

$$
\left(\hat{u}^{\tau}_{i},\hat{c}^{\tau}_{i},\widehat{\Delta c^{\tau}}_{i}\right)=D_{\phi}(h^{\tau}_{i})
$$

监督损失（[[SmoothL1 Loss]]）：

$$
\mathcal{L}_{\mathrm{state}} = \mathrm{SmoothL1}\left(\hat{c}^{\tau},c^{\tau,\star}\right), \qquad
\mathcal{L}_{\mathrm{change}} = \mathrm{SmoothL1}\left(\widehat{\Delta c^{\tau}},\Delta c^{\tau,\star}\right)
$$

## 核心要点
1. **触觉状态代理 $c^\tau$**：捕捉局部接触负载（contact loading）的连续量
2. **触觉变化代理 $\Delta c^\tau$**：相邻步状态代理的差分，强调首次接触、滑动开始、压缩变化、卡阻等阶跃事件
3. 用轻量代理头 $D_\phi$ 从触觉隐状态预测，使模型在推理时无需显式解码未来触觉图像即可获得接触变化信号
4. 这种"运动衍生代理"设计避开了对标定力/力矩传感器的依赖，降低了部署门槛，但也意味着它不是真实物理力的度量

## 代表工作
- [[Tactile-WAM]]: 提出该代理目标，驱动 [[Touch-Aware Bias|触觉感知偏置]]的构造

## 相关概念
- [[Touch-Aware Bias]]
- [[Tactile Asymmetric Attention Mechanism (TAAM)]]
- [[SmoothL1 Loss]]
