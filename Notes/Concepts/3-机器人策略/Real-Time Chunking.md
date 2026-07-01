---
type: concept
aliases: [Real-Time Chunking, RTC, 实时动作分块]
---

# Real-Time Chunking

## 定义

Real-Time Chunking（RTC）是一种用于 [[Action Chunking|动作分块]] 策略推理时的实时执行技术：在当前动作块（chunk）尚未执行完毕时，以已执行/已承诺的动作前缀（action prefix）作为条件提前推理生成下一个动作块，从而消除分块边界处的执行停顿，实现连续无缝的闭环控制。

## 核心要点

1. **动作前缀条件化**: 将上一个 chunk 中已经执行或即将执行的部分动作作为条件输入给下一次推理，保证新旧 chunk 在过渡处的动作连续性
2. **消除推理延迟空档**: 传统分块策略需等待整块动作执行完才能开始下一块推理，RTC 通过提前触发推理与前缀条件化重叠这一过程，避免控制中断
3. **提升高频控制下的流畅度**: 对于需要高频闭环控制的精细操作任务（如折叠、插拔）尤为重要

## 代表工作

- [[ABC]]: 研究不同动作前缀长度对策略性能的影响（消融实验），验证 RTC 风格的前缀条件化在实际部署中的效果
- [[Qwen-RobotManip]] (2026): 推理时采用 RTC + $K_{\text{repeat}}=8$ 步 Flow Matching 采样，实现无缝闭环控制

## 相关概念

- [[Action Chunking]]
- [[Diffusion Policy]]
- [[VLA（视觉-语言-动作模型）]]
