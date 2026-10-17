---
type: concept
aliases: [FlowDPPO, Flow Diffusion PPO, Flow-based DPPO]
---

# Flow-DPPO

## 定义
Flow-DPPO 是将扩散策略优化（Diffusion PPO）框架应用于流匹配生成模型的在线 RL 微调方法，使用 PPO 风格的裁剪目标对去噪过程进行策略优化，支持信任域约束。

## 核心要点
1. **PPO 裁剪目标**：采用 PPO 的概率比裁剪机制控制策略更新幅度
2. **ODE→SDE 转换**：将确定性 ODE 采样过程转换为带随机性的 SDE 以定义状态转移分布
3. **信任域约束**：通过 KL 散度或概率比约束防止策略更新过大

## 方法关联
Flow-DPPO 的 ODE→SDE 转换方式被 [[DiffusionART]] 借鉴用于定义信任域掩码中的参考 Gaussian 转移分布。

## 实验性能（FLUX.1-dev GenEval eval reward）
- Flow-DPPO: 0.9183
- DiffusionART: 0.9684（高 5.5%）

## 代表工作
- [[DiffusionART]]: 在文生图任务上将其作为对比基线，并借鉴其 ODE→SDE 转换思路

## 相关概念
- [[PPO]]: 基础算法框架
- [[FlowGRPO]]: 同属流匹配模型的反向过程 RL 方法
- [[流匹配]]: 基础生成框架
