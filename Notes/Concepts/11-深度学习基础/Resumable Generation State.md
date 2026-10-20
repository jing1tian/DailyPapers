---
type: concept
aliases: [可恢复生成状态, 生成状态持久化, Generation State Checkpointing]
---

# Resumable Generation State

## 定义

可恢复生成状态（Resumable Generation State）是 [[OxyGen]] 提出的核心数据结构：将一个语言生成请求的完整推理状态封装为三元组 $\sigma_t = (\mathcal{K}_t, y_t, \delta_t)$，使其可以在任意解码步暂停、存储，并在后续帧中恢复继续解码。

## 数学形式

$$\sigma_t = (\mathcal{K}_t,\; y_t,\; \delta_t)$$

- $\mathcal{K}_t = \{(K_{t,l}, V_{t,l})\}_{l=1}^{L}$：当前 KV Cache（prefill KV + 解码累积 KV）
- $y_t = y_{t,1:j}$：已生成的 token 序列（长度 $j$，$j < N$）
- $\delta_t \in \{0,1\}$：终止标志（是否已生成 EOS）

**批次状态**（跨帧批处理时）：

$$\hat{\sigma} = (\hat{\mathcal{K}},\; \hat{y},\; \hat{\delta})$$

将 $B$ 个活跃请求的状态拼接为批次，统一推进 $k$ 步解码。

## 核心要点

1. **跨帧持久化**：语言请求通常需要 $N$ 步才能完成，但每帧只解码 $k$ 步（$k \ll N$），未完成的状态需要被保存到下一帧
2. **四操作接口**：Store / Retrieve / Update / Remove，构成统一 KV Cache 管理器的完整 API
3. **内存效率**：避免每帧重新启动请求（重新 prefill），仅存储增量 KV
4. **终止检测**：通过 $\delta_t$ 标志自动识别完成的请求并释放内存

## 代表工作

- [[OxyGen]]: 提出 Resumable Generation State 作为统一 KV Cache 管理器的核心数据结构

## 相关概念

- [[KV Cache]]
- [[Continuous Batching]]
- [[Prefill]]
- [[Autoregressive Generation]]
