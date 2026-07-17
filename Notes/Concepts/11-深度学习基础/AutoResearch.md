---
type: concept
aliases: [AutoResearch, 自动超参优化, 自动化研究流水线]
---

# AutoResearch（自动化超参优化 Agent 流水线）

## 定义
AutoResearch 是一种基于 LLM/Agent 的自动化超参搜索与训练管理流水线：先用短期（1K-step）试探性扫描快速评估候选超参组合，再将最优配置延伸到完整训练，替代人工反复调参的工程流程。

## 数学形式

$$
\theta^* = \arg\min_{\theta \in \Theta} \; \mathcal{L}_{\text{eval}}\!\left(f_\theta; \mathcal{D}_{\text{val}}\right)
$$

其中 $\Theta$ 为超参搜索空间（学习率、批大小等），$\mathcal{L}_{\text{eval}}$ 为短期试验验证集损失（代理指标）。

## 核心要点

1. **两阶段流程**: ① 1K-step 试探性扫描（pilot sweep）快速排除劣质超参；② 选出最优配置后延伸到完整训练（如 30K steps）
2. **Agent 驱动**: 由 LLM Agent 解析训练日志、自动触发下一组试验、汇总结果并做决策，无需人工干预
3. **低成本代理**: 以短期训练损失（如 train action loss 和 eval MSE）作为代理指标，相关性高于最终成功率但计算成本极低
4. **应用场景**: 机器人策略学习、扩散模型训练等实验周期长、单次评估昂贵的任务

## 代表工作
- [[GigaWorldPolicy0.5]]: 首次在 WAM 训练中引入 AutoResearch，1K-step 扫描 4 个学习率候选，自动选出 lr=6×10⁻⁵

## 相关概念
- [[Flow Matching]]: AutoResearch 在 GigaWorldPolicy0.5 中优化的目标训练框架
- [[World Action Model]]: AutoResearch 的应用场景
