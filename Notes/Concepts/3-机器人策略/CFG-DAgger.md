---
type: concept
aliases: [CFG-DAgger, 分类器自由引导数据增强, Classifier-Free Guided DAgger]
---

# CFG-DAgger

## 定义

CFG-DAgger 是一种基于 [[Rollback]] 收集正负轨迹样本、结合 [[Classifier-Free Guidance (CFG)]] 在动作空间进行引导推理的在线模仿学习方法，使策略在推理时远离负样本预测方向、朝向正样本预测方向。

## 数学形式

**数据集**:
$$D^+ = D_{demo} \cup D_{HIL},\quad D^- = \bigcup_j \tau_j^-$$

**推理时动作空间 CFG**:
$$a_{CFG} = a_- + \gamma(a_+ - a_-),\quad \gamma = 1.25$$

## 核心要点

1. 训练一个条件策略：正样本用 $D^+$（condition dropout 概率 0.3），负样本 $D^-$ 始终保留条件标记。
2. 推理时运行正、负两个分支，以负预测为排斥基准向正方向外推（$\gamma=1.25$）。
3. 无需价值学习或优势估计，直接利用 Rollback 的结构化负样本。

## 代表工作

- [[MR0]]: 在以太网插线任务中达到 70% SR / 0.90 PS，优于 RECAP

## 相关概念

- [[Rollback]]
- [[Classifier-Free Guidance (CFG)]]
- [[HG-DAgger]]
- [[RECAP]]
- [[DAgger]]
