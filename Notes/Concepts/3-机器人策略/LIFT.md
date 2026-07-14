---
type: concept
aliases: [LIFT adaptation]
---

# LIFT

## 定义
一种针对预训练机器人 policy 的快速微调/适应方法，通过直接在 policy 参数上施加微调来适配新场景，与 FlowDAgger 的 latent space 干预形成对比。

## 核心要点
1. 直接微调 policy 网络参数（非 latent adapter）
2. 适用于 distribution shift 较小的场景
3. 相比 latent space 方法需要更多 gradient 更新
4. FlowDAgger 在 sample efficiency 上优于 LIFT

## 代表工作
- [[FlowDAgger]]：在相同设置下与 LIFT 对比 sample efficiency

## 相关概念
- [[DAgger]]
- [[LoRA]]
