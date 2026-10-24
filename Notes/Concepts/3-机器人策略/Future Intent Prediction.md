---
type: concept
aliases: [未来交互意图预测, Intent Prediction, Future Intent]
---

# Future Intent Prediction

## 定义

在机器人操作任务中，从历史轨迹（动作序列、机器人状态）和语言指令出发，预测未来若干步内机器人是否将进入"交互阶段"（如精确抓取、对齐、按压等需要近距离操作的步骤）的前瞻性预测机制。

## 数学形式

$$
I_t = [i^{(1)}, i^{(2)}, \ldots, i^{(H)}], \quad i^{(k)} \in [0, 1]
$$

其中 $i^{(k)}$ 为从当前时刻起第 $k$ 步处于交互阶段的预测概率，$H$ 为预测步长（FocusVLA 中 $H=30$）。

## 核心要点

1. **前瞻性**：与响应式方法（基于当前帧推断）不同，通过历史动作序列预测未来意图，使路由权重具备预判能力
2. **自动化标签**：无需人工/VLM 标注，从轨迹的末端执行器运动学（速度、夹爪状态）自动提取意图窗口
3. **轻量监督**：采用类别平衡 BCE 损失（$\lambda_+ = 15$），配合 5K 步意图预热训练
4. **Teacher Forcing 退火**：训练时 5K-15K 步逐步从 teacher forcing 过渡到自回归，推理时完全自回归

## 代表工作

- [[FocusVLA]]: 提出 Future Intent Bridge 模块实现该能力，意图标签与人工标注 F1 达 82.9%

## 相关概念

- [[Dynamic View Routing]]
- [[View-Dependency Inertia]]
- [[Action Chunking]]
- [[Cross-Attention]]
