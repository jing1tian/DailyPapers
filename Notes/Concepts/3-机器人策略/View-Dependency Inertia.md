---
type: concept
aliases: [视角依赖惯性, View Dependency Inertia, Static View Bias]
---

# View-Dependency Inertia

## 定义

多视角 VLA 模型在整个任务执行过程中对某一特定相机视角（通常是腕部/局部相机）保持固定偏好，无法根据任务阶段的变化动态调整视角依赖比例的现象。

## 数学形式

$$
D_t = |a_{\text{global},t} - a_{\text{local},t}|
$$

Dominance Margin $D_t$ 衡量每步的视角依赖不均衡程度；局部依赖变化度 $V = \text{Var}(\Delta L_t)$ 衡量模型的视角切换能力。

失败轨迹通常表现为：更大的 $D_t$（视角偏差更强）和更小的 $V$（视角切换更少）。

## 核心要点

1. 固定融合（如将所有相机帧均等拼入上下文）导致模型在任务不同阶段无法差异化利用全局/局部视觉信息
2. 量化指标：Dominance Margin（$D_t$）和局部依赖时序变化度（$V$）
3. 验证方式：遮蔽实验（规划阶段遮蔽局部视图使成功率从 85% 跌至 20%）

## 代表工作

- [[FocusVLA]]: 首次定义并量化该现象，提出 Future Intent Bridge + Action-Side Dynamic Routing 解决方案

## 相关概念

- [[Dynamic View Routing]]
- [[Future Intent Prediction]]
- [[VLA]]
