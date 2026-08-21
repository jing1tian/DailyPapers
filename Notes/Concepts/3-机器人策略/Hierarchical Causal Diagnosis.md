---
type: concept
aliases: [层次化因果诊断, Hierarchical Causal Diagnosis, HCD, 分层因果诊断]
---

# Hierarchical Causal Diagnosis

## 定义

针对具身智能系统失败案例的自顶向下六层根因定位框架，通过最小干预原则在最高抽象层级实施修复，防止过度修改污染动作分布。

## 诊断层级结构

```
Evaluation  →  Critic  →  State  →  Planning  →  Recovery  →  Parameter
   (最高)                                                         (最低)
```

遵循**最小干预**原则：在能解决问题的最高层修复，不下探到更低层。

## 数学形式

第一个未达成里程碑定位：

$$
m^* = \min\!\left\{m_k \in \mathcal{M} \;\middle|\; m_k \notin \{\mu_t\}_{t=0}^T\right\}
$$

最早可观测偏差点 (EOD)：

$$
t_{EOD} = \min\!\left\{t \;\middle|\; \mathrm{dist}(s_t,\, s_t^{ref}) > \varepsilon,\;\; s_t^{ref} \in \mathcal{I}_{succ}(\mu_t)\right\}
$$

## 核心要点

1. **自顶向下遍历**: 优先检查高层（评估标准是否合理、批评器是否误触）再下探到低层（状态感知、规划、恢复方案、参数设置）
2. **最小干预**: 若批评器触发阈值错误即可修复，不修改底层动作规划；防止在特定 seed 上过拟合
3. **里程碑分解**: 将复杂任务分解为有序里程碑序列 $\mathcal{M}$，先定位失败阶段再定位失败时刻
4. **成功参考对比**: 用 $\mathcal{I}_{succ}(\mu)$ 构建正常状态分布，通过统计距离检测偏离

## 代表工作

- [[Zetta]]: 在演化 Phase II 应用层次化因果诊断，结合 EOD 和里程碑定位实现精准根因修复

## 相关概念

- [[Runtime Critics]]
- [[Evolvable Harness]]
- [[Closed-Loop Control]]
