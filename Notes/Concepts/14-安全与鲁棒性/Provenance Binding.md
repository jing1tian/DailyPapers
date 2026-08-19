---
type: concept
aliases: [溯源绑定, 审计链, provenance-binding, cryptographic audit trail]
---

# Provenance Binding（溯源绑定）

## 定义

**溯源绑定**指通过密码学哈希将系统状态转移的完整上下文（父状态、提议变更、开发数据、提交决定）绑定为不可篡改的审计记录，使得每次状态演化均可独立验证。

## 数学形式

$$
\mathcal{T}_r = (h(\Omega_r),\; h(\delta_r),\; h(D_r),\; b_r,\; \mathcal{R}_r,\; h(\Omega_{r+1}))
$$

验证链完整性：

$$
h(\Omega_r^{\text{parent}}) = h(\Omega_r)
$$

其中 $h(\cdot)$ 为密码学哈希函数，$b_r \in \{0,1\}$ 为提交决定，$\mathcal{R}_r$ 为守护条件结果。

## 核心要点

1. **不可篡改性**: 密码学哈希确保任何中间状态的修改均会被检测
2. **完整可追溯**: 从初始状态 $\Omega_0$ 到最终冻结状态 $\Omega_R$ 的每一步均有记录
3. **分数隔离验证**: 结合 [[Score Isolation]] 约束，可验证持出评估分数是否影响了状态演化
4. **可重现性**: 外部审计者可独立重现整个开发历史

## 代表工作

- [[SCOPE]]: 在视频世界模型推理时自适应框架中引入溯源绑定，作为可审计性保证

## 相关概念

- [[Score Isolation]]
- [[Inference-Time Adaptation]]
