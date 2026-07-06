---
type: concept
aliases: [Action Diffusion Transformer, ActionDiT]
---

# ActionDiT

## 定义
Kairos 架构中专门用于动作序列生成与预测的 Diffusion Transformer 组件，与 VideoDiT 共享骨干但针对离散/连续动作空间专门设计。

## 数学形式
$$a_{t+1:t+H} = \text{ActionDiT}(v_{1:t}, a_{1:t}; \theta_a)$$

输入历史视觉帧和动作序列，输出未来 H 步的动作预测分布。

## 核心要点
1. Kairos 统一架构的组成部分：理解（VLM）+ 生成（VideoDiT）+ 预测（ActionDiT）
2. 与 VideoDiT 在同一 DiT backbone 上通过任务头区分
3. 支持 cross-embodiment：不同机器人平台共享同一 ActionDiT 参数
4. 原生预训练阶段（Native Pre-training）包含 Action 数据联合训练

## 代表工作
- [[Kairos]]: 提出 ActionDiT 作为 Physical AI 原生世界模型的动作组件

## 相关概念
- [[VideoDiT]]: Kairos 中的视频生成组件
- [[DiT]]: 基础架构
- [[WAM]]: World Action Model，ActionDiT 所服务的更大框架
- [[VLA]]: ActionDiT 的前序方向
