---
type: concept
aliases: [CALVIN benchmark, CALVIN ABC→D]
---

# CALVIN

## 定义

CALVIN（Composing Actions from Language and Vision）是一个长时序多技能机器人操作基准，要求策略在未见过的场景配置（域 D）中根据语言指令完成连续 5 个子任务序列，训练域为场景 A、B、C。

## 数学形式

评估指标为 Success@k（完成序列前 k 步的概率）和平均序列长度：

$$
\text{Avg. Len} = \sum_{k=1}^{5} \text{Success@}k
$$

## 核心要点

1. **零样本跨域泛化**: 训练在 ABC 三个场景，测试在语义相同但视觉/配置不同的 D 场景。
2. **长时序评测**: 标准设置为 1,000 条 5 步序列，Avg. Len 满分为 5.0。
3. **技能多样性**: 涵盖滑动、旋转、开关灯、开关抽屉等多类原语技能（约 34 类子任务）。
4. **误差积累敏感**: Success@4/5 的提升难度远高于 @1/2，是区分方法长时序能力的主要指标。

## 代表工作

- [[MAST-VLA]]: SOTA，Avg. Len 4.65（截至 2026 年 10 月）
- OpenVLA-OFT: 4.10
- VLA-Adapter-Pro: 4.50

## 相关概念

- [[VLA（视觉-语言-动作模型）]]
- [[Action Chunking]]
