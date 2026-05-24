---
type: concept
aliases: [Atari 100K Benchmark, Atari 100K 基准]
---

# Atari 100K

## 定义

一个严格限制数据量的视觉强化学习基准：智能体在 26 款 Atari 游戏中仅允许与环境交互 100,000 步（约 2 小时游戏时间），重点考察样本效率（sample efficiency）。

## 核心要点

1. **评测指标**:
   - **IQM**（Interquartile Mean）：去掉最高/最低各 25% 的得分后取均值，对异常值鲁棒
   - **Optimality Gap**：$1 - \text{human-normalized score}$，越小越好
2. **标准化得分**: 相对于 human expert 和随机 agent 进行归一化
3. **26 款游戏**: 覆盖不同难度、视觉复杂度和动作空间
4. **样本效率导向**: 100K 步远少于标准 Atari 评测（200M 步），旨在考察 model-based RL 和迁移学习的优势

## 代表工作

- [[ITC]]: IQM 1.092，Optimality Gap 0.376，当前 SOTA
- [[DreamerV3]]: IQM 0.487
- [[STORM]]: IQM 0.561
- Diamond: IQM 0.641
- Simulus: IQM 0.990

## 相关概念

- [[Craftax]]: 同样用于 World Model 评测的另一基准
- [[World Model]]: Atari 100K 上 SOTA 方法通常基于 World Model
- [[Visual RL]]: 从像素图像直接学习策略的强化学习范式
