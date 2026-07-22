---
type: concept
aliases: [Predicate Supervisor, 谓词监督器, 执行验证器]
---

# Predicate Supervisor

## 定义

Predicate Supervisor（谓词监督器）是一种基于**几何谓词**的执行验证模块，在机器人执行每个动作块后评估物理关系是否满足任务目标，并根据状态诊断触发定向恢复策略。

## 数学形式

**谓词结构**:

$$p = \langle \kappa, \alpha, \mathrm{op}, \nu, n \rangle$$

**证据评估**:

$$\phi(\kappa, \alpha, \mathcal{M}^{\tau_i}_\xi)$$

## 核心要点

1. **谓词类型** $\kappa$: containment（包含）, support（支撑）, proximity（近接）, displacement（位移）, bimanual（双臂）, handover（交递）
2. **时序稳定性**: 需在连续 $n$ 帧内满足条件才判定为真，避免抖动误判
3. **状态机输出**: `in_progress` / `done` / `blocked` / `failed` / `uncertain`
4. **定向恢复**: 根据诊断触发重观测、重试、重定位或重规划

## 与通用验证方法的区别

- 不同于 VerifyLLM/ReplanVLM 使用独立感知，谓词监督器与动作策略共享同一 3D 物体记忆（见 [[对象中心表示]]），实现真正闭环
- 谓词基于几何量（距离、包含、对齐）而非视觉特征，鲁棒性更强

## 代表工作

- [[POT-VLA]]: 首次将谓词监督器与 VLA 的 3D 物体 Token 系统集成，在人形 Loco-Manipulation 上验证

## 相关概念

- [[对象中心表示]]
- [[Loco-Manipulation]]
- [[VLA]]
- [[符号抽象]]
