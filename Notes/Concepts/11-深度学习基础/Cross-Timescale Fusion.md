---
type: concept
aliases: [跨时间尺度融合, Cross-Timescale Attention, 时间尺度融合]
---

# Cross-Timescale Fusion（跨时间尺度融合）

## 定义

跨时间尺度融合是一种在异步双速架构中将慢速分支的陈旧语义上下文与快速分支的即时视觉感知融合的 [[Cross-Attention]] 机制，使低延迟动作头在缺乏实时高层语义时仍能利用当前视觉信息做出精确动作。

## 数学形式

**层级路由（DiT 层 $d$ 对应 VLM 层 $\phi(d)$）:**

$$
\phi(d) = \left\lceil \frac{\left\lfloor 4d / L_{\text{dit}} \right\rfloor + 1}{4}\, L_{\text{vlm}} \right\rceil
$$

**Key/Value 构造（陈旧语义 + 新鲜视觉拼接）:**

$$
K_d = V_d = \Big[\, W_{\phi(d)} H^{(\phi(d))}_{t-\Delta t} \;;\; W_v V^{\text{fresh}}_t \,\Big]
$$

**DiT 块前向:**

$$
Z^{\text{sa}}_{d+1} = \mathrm{FFN}\big(\mathrm{CrossAttn}(Q = Z^{\text{sa}}_d,\, K_d,\, V_d)\big)
$$

## 核心要点

1. **双源 Key/Value**: 将陈旧 VLM 隐藏状态（提供高层语义）和即时 ResNet 编码视觉（提供当前场景）拼接为统一的 Key/Value 序列
2. **层级路由**: 将 DiT 层均匀映射到对应的 VLM 层，实现多层次语义注入
3. **冻结骨干 + 新鲜视觉**: VLM 冻结（保留预训练特征），仅训练 ResNet 编码器和 DiT
4. **时序解耦**: 新鲜视觉弥补 VLM 陈旧带来的感知滞后，二者在注意力中互补

## 代表工作

- [[DASH]]: 提出 Cross-Timescale Fusion，用于异步 VLM + DiT 动作头架构的视觉-语义融合

## 相关概念

- [[Cross-Attention]]
- [[DiT]]
- [[ResNet]]
- [[Desynchronized Training]]
- [[Decoupled Policy]]
