---
type: concept
aliases: [OmniControl, Omni-Control]
---

# OmniControl

## 定义

一种可控人体运动生成模型，支持对任意关节在指定时刻施加空间约束，同时保持文本描述的运动语义，实现细粒度的时空可控人体运动生成。

## 数学形式

$$
H \sim p_{\mathrm{motion}}(H \mid \ell, \mathcal{G}), \qquad \mathcal{G} = \{(j_k, \mathbf{v}_k, t_k)\}_{k=1}^{K}
$$

其中 $\ell$ 为语言指令，$\mathcal{G}$ 为时空关节约束集合，$j_k$ 为关节，$\mathbf{v}_k$ 为目标值，$t_k$ 为时刻。

## 核心要点

1. 可对任意关节（腕部、足部、根部等）施加位置/速度约束，实现精确的时空控制
2. 约束与文本语义联合建模，保证约束满足的同时维持运动自然性
3. 在 DreamVLA / DreamControl 等方法中用于生成任务参考运动（如抓取时的腕部约束）

## 代表工作

- OmniControl (Xie et al., ICLR 2024, arXiv:2310.08580): 原始论文

## 相关概念

- [[OmniRetarget]]
- [[Human-to-Robot Retargeting]]
- [[运动捕捉]]
- [[DreamVLA]]
