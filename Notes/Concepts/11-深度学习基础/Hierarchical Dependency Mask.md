---
type: concept
aliases: [层次依赖掩码, HDM, Hierarchical Dependency Mask]
---

# Hierarchical Dependency Mask

## 定义
一种用于多视界预测的注意力掩码机制，通过定义视界间的偏序关系（短期 ≼ 里程碑 ≼ 目标），使较高层次的视界预测可以关注较低层次视界的查询，从而建立跨时间视界的信息依赖。

## 数学形式

在预测视界 $h$ 时，HDM 允许查询 $\bm{Q}^h$ 在注意力计算中访问所有满足 $h' \preceq h$ 的视界查询 $\bm{Q}^{h'}$：

$$
\bm{U}_t^h = \mathcal{H}_\theta\!\left(\bm{Q}^h;\, [\bm{V}_t;\, \bm{E}_L;\, \underbrace{\bm{Q}^{\preceq h}}_{\text{HDM 控制}}]\right)
$$

偏序关系：$s \preceq m \preceq g$（短期 ≼ 里程碑 ≼ 目标）

## 核心要点

1. **因果性**: 短期视界只看自身，里程碑视界可借助短期信息，目标视界可综合所有视界——符合信息流方向
2. **层次信息聚合**: 低视界的预测充当高视界的"子目标锚点"，使目标预测更连贯
3. **与标准因果掩码的区别**: 标准因果掩码作用于时间序列中的 token，HDM 作用于不同**预测视界**之间

## 代表工作
- [[HiFOR]]: 提出 HDM 用于层次化预见预测（HFP）

## 相关概念
- [[Hierarchical Foresight Prediction]]: HDM 应用的上层框架
- [[Attention Mechanism]]: 基础技术
