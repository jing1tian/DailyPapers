---
type: concept
aliases: [pi0, π₀, π0, Physical Intelligence, pi-zero]
---

# π₀ (pi0)

## 定义

Physical Intelligence 团队发布的通用机器人 [[VLA]] 基础模型，采用 PaliGemma VLM + flow-matching 动作头架构，在大规模机器人数据上预训练，具备强泛化能力。

## 核心要点

1. 基于预训练 VLM（PaliGemma）融合语言与视觉信息
2. 使用 flow-matching 生成连续动作序列（[[Action Chunking]]）
3. 在 LIBERO 基准上取得 94.4% 平均成功率，是当前主要 SFT 基线
4. DyGRO-VLA 以其 SFT 版本为出发点，通过离线 + 在线两阶段提升至 97.1%

## 代表工作

- [[Pi05]]: π0 的改进版本（π0.5）
- [[DyGRO-VLA]]: 以 π₀ 为基础策略，通过混合残差 RL 微调解决多任务灾难性遗忘

## 相关概念

- [[VLA]]
- [[Action Chunking]]
- [[信息瓶颈]]
