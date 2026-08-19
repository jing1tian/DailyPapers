---
type: concept
aliases: [Test-Time Computation, 测试时计算, Test-Time Scaling]
---

# TTC (Test-Time Computation)

## 定义
在模型推理阶段（非训练阶段）动态分配额外计算资源以提升输出质量的范式；通过搜索、采样或验证来代替直接的单次前向传播。

## 核心要点
1. **核心假设**：对于难题，花更多推理时间搜索/验证可以提高准确率，即便模型权重不变
2. **主要实现方式**：Best-of-N 采样 → Verifier 排序；MCTS；迭代精炼；World Model 验证
3. **在机器人领域的应用**：[[τ₀-VLA]] 用世界模型作为 TTC 验证器，在推理时对高层子目标采样并验证

## 相关概念
- [[MCTS]]
- [[World Model]]
- [[DreamerV3]]
