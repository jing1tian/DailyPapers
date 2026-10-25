---
type: concept
aliases: [长时域操作, 长时域任务, Long-horizon Task, Long-horizon Planning]
---

# Long-horizon Manipulation（长时域操作）

## 定义

需要连续执行大量步骤（数十至数百步）才能完成的机器人操作任务，要求模型在整个过程中保持状态追踪、进度维护和错误恢复能力。

## 数学形式

$$
\text{AP} = \frac{1}{N}\sum_{i=1}^{N}\frac{\text{正确前缀技能数}}{\text{总技能数}_i}
$$

Average Progress (AP) 是量化长时域完成度的核心指标。

## 核心要点

1. 主要挑战：进度丢失（后续动作破坏前期成果）、不可逆失败、上下文窗口限制
2. 扩展步数预算（10×）对长时域任务提升显著（LIBERO-Ultra-Long: 82.2% → 98.9%）
3. LIBERO-Long++、LIBERO-Ultra-Long、LIBERO-PRO 是代表性基准

## 代表工作

- [[GPT6Astra-EmbodiedEval]]: 分析三类长时域失败模式，LIBERO-Ultra-Long 10× 达 98.9% SR

## 相关概念

- [[LIBERO]]
- [[Action Chunking|动作分块]]
- [[In-Context Learning]]
