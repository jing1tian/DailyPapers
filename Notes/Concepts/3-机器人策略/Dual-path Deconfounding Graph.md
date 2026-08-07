---
type: concept
aliases: [DDG, 双路去混淆图]
---

# Dual-path Deconfounding Graph

## 定义
CofactVLA 提出的因果图框架，通过同时运行事实分支（语言+视觉）和反事实分支（仅视觉），显式建模并分离语言驱动路径与视觉混淆路径，为 VLA 的因果干预提供理论基础。

## 核心要点
1. **事实分支**: 输入 = 语言指令 $l$ + 观测 $o_t$，输出速度场 $v_\text{cond}$
2. **反事实分支**: 输入 = 仅 $o_t$（屏蔽语言），输出速度场 $v_\text{uncond}$
3. **关键假设**: 两分支的差异精确捕获了视觉混淆变量对动作的虚假贡献
4. **双层干预**: 在动作层（OPG）和特征层（CCR）分别截断虚假路径

## 代表工作
- [[CofactVLA]]: 提出 DDG，并基于其设计 OPG 和 CCR 两种干预机制

## 相关概念
- [[Vision-Override Phenomenon]]
- [[Orthogonal Projection Guidance]]
- [[Counterfactual Covariance Reduction]]
- [[因果推断]]
