---
type: concept
aliases: [Future-KV Interface, Future KV Cache, Future Slot KV Prefill]
---

# Future-KV（未来槽 KV 预填充接口）

## 定义

在直接策略 World Action Model 中，通过对当前帧潜变量与噪声初始化的未来槽执行单次视频 DiT 前向传播（KV 预填充），将预测性动力学以隐式 Key-Value 缓存形式注入 Action DiT 动作去噪过程，而无需在推理时显式生成任何未来帧。

## 数学形式

$$
\tilde{z}^{F_{\text{sub}}}_{1:T} = \operatorname{concat}(z_{\text{cur}}(o),\ \varepsilon_F), \quad \varepsilon_F \sim \mathcal{N}(0, I)
$$

$$
(D_\theta, H_{KV}) = \operatorname{KVPrefill}_\phi(\tilde{z}^{F_{\text{sub}}}_{1:T},\, l,\, p)
$$

其中 $H_{KV}$ 为逐层 Key-Value 状态缓存，供 Action DiT 动作去噪全程读取。

## 核心要点

1. **单次前向传播**: 训练时有监督真实未来帧，推理时以噪声初始化未来槽，一次 KV 预填充产生预测性上下文
2. **训练-推理非对称**: Ground-truth 未来帧仅训练时使用，推理时无此依赖，实现计算效率与预测能力的解耦
3. **KV 复用**: 预填充产生的 $H_{KV}$ 在整个动作去噪过程中被复用，避免重复计算
4. **随机性注入**: $\varepsilon_F \sim \mathcal{N}(0,I)$ 使未来槽保持随机性，视频主干仍能从中提炼隐式预测动力学

## 代表工作

- [[ForeWAM]]: 提出 Future-KV Interface 的原始工作，结合 Dynamics Registers 在 LIBERO-Plus 上超越 Fast-WAM +10.1 分

## 相关概念

- [[KV Cache]]: 底层机制
- [[Dynamics Registers]]: 与 Future-KV 互补的动力学寄存器组件
- [[ActionDiT]]: 读取 Future-KV 缓存的动作去噪网络
- [[Flow-Matching|Flow Matching]]: 视频 DiT 预填充所用的训练框架
- [[Latent-Foresight]]: 相关但不同（显式预测潜在未来状态，而非隐式 KV 缓存）
