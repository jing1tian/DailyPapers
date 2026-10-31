---
type: concept
aliases: [层次化预见预测, HFP, 多视界预测, Hierarchical Foresight]
---

# Hierarchical Foresight Prediction

## 定义
在机器人操作策略中，同时预测短期状态变化、中期语义里程碑和最终目标状态的多视界联合预测框架，通过跨视界依赖掩码使各时间视界的预测相互关联而非独立建模。

## 数学形式

$$
\bm{U}_t^h = \mathcal{H}_\theta\!\left(\bm{Q}^h;\, [\bm{V}_t;\, \bm{E}_L;\, \bm{Q}^{\preceq h}]\right), \quad h \in \{s, m, g\}
$$

其中 $s \preceq m \preceq g$ 表示短期（short）、里程碑（milestone）、目标（goal）三个视界的偏序关系，[[Hierarchical Dependency Mask|HDM]] 控制各视界查询的可见范围。

## 核心要点

1. **三视界联合预测**: 短期（下一执行间隔）、里程碑（下一任务阶段完成时刻）、目标（最终状态）
2. **跨视界依赖**: 短期预测可"看到"自身，里程碑可看到短期，目标可看到全部——信息从粗到细逐级传递
3. **语义特征空间**: 在 [[DINOv2]] 特征空间而非 RGB 像素空间预测未来，更抗噪声
4. **辅助状态预测**: 短期视界额外预测机器人状态，提供精确的运动学监督信号

## 代表工作
- [[HiFOR]]: 提出 Hierarchical Foresight Prediction 并结合 AWR 用于 VLA 操作

## 相关概念
- [[Hierarchical Dependency Mask]]: 控制跨视界信息依赖的注意力掩码
- [[VLA（视觉-语言-动作模型）]]: 应用场景
- [[World Model]]: 与隐式世界模型的关系
- [[DINOv2]]: 预测目标特征空间
