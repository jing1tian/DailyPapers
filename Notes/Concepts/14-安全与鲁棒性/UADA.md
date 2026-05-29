---
type: concept
aliases: [Universal Adversarial Attack, UADA attack]
---

# UADA

## 定义
针对 VLA 模型的通用对抗补丁攻击方法，通过最大化代理模型的动作 token 预测损失来优化补丁，支持单架构（UADA1）和多架构集成（UADA1-3）两种变体。

## 核心要点
1. 直接优化代理 VLA 模型的动作预测损失（离散 token 或 logit）
2. 单架构版本（UADA1）白盒效果好但迁移性差
3. 多架构集成版本（UADA1-3）白盒性能提升，但跨架构迁移仍受动作头异质性限制
4. 因攻击目标依赖于架构特定的动作空间，跨架构迁移 Transfer Avg 约 37.53%

## 代表工作
- [[VLA-Hijack]]：将 UADA 作为主要基线，迁移均值超越 UADA1-3 约 +24%

## 相关概念
- [[对抗补丁攻击]]
- [[VLA（视觉-语言-动作模型）]]
- [[视觉本体感知]]
