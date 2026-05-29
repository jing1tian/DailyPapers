---
type: concept
aliases: [UniVLA]
---

# UniVLA

## 定义
统一的视觉-语言-动作模型，通过大规模跨具身数据预训练，实现多机器人平台、多任务的通用策略。

## 核心要点
1. 多具身、多任务统一训练，一个模型适配多种机器人
2. 使用跨平台数据混合训练，提升零样本泛化
3. 在 LIBERO 等 benchmark 上表现较强
4. 作为 adversarial attack 的目标模型被 VLA-Hijack 测试

## 代表工作
- [[VLA-Hijack]]：对 UniVLA 进行可迁移 patch 攻击测试

## 相关概念
- [[OpenVLA]]
- [[VLA]]
- [[LIBERO]]
