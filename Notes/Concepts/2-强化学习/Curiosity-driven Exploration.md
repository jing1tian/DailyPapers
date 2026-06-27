---
type: concept
aliases: [好奇心驱动探索, Curiosity Reward, Plan2Explore, 内在好奇心奖励]
---

# Curiosity-driven Exploration

## 定义

一类用模型自身的预测误差/不确定性作为内在奖励（intrinsic reward）来驱动智能体探索的方法。智能体不依赖外部任务奖励，而是主动寻访那些会让世界模型/预测网络感到"意外"或"不确定"的状态-动作区域，从而高效扩大数据覆盖范围。代表性方法包括 Pathak et al. (2017) 的自监督预测误差奖励、Burda et al. (2019) 的 Random Network Distillation，以及 Sekar et al. (2020) 的 Plan2Explore（将好奇心信号缩放到想象轨迹中，用于基于模型的强化学习）。

## 数学形式

以预测误差为内在奖励的一般形式：

$$
r^{\text{intrinsic}}_t = u(s_t, a_t)
$$

其中 $u(\cdot)$ 为某种不确定性/误差度量（如前向模型预测误差、模型集成方差、或 [[Shortcut Flow-Matching|flow-matching]] 去噪过程的不稳定性）。智能体的探索策略 $\pi^{\text{explore}}$ 最大化累积内在奖励：

$$
\pi^{\text{explore}} = \arg\max_\pi \, \mathbb{E}_\pi\left[\sum_t \gamma^t r^{\text{intrinsic}}_t\right]
$$

## 核心要点

1. **无需外部奖励**: 探索目标完全由模型自身的不确定性定义，适合在没有任务奖励信号的阶段进行数据采集
2. **与不确定性量化的天然耦合**: 任何能输出"预测置信度"的信号（集成方差、重建残差、去噪稳定性等）都可直接复用为好奇心奖励
3. **从单任务探索扩展到数据采集**: 传统好奇心驱动探索用于驱动单任务策略学习；新的应用方式是将其用于**生成式世界模型本身的数据采集**——以模型对某状态的不确定性高低决定是否采集更多该区域的数据

## 代表工作

- [[MMBench2]]: 将三个幻觉预测信号（$u_r^{\text{norm}}, u_f^{\text{norm}}, u_s^{\text{norm}}$）中的 tokenizer 残差信号 $u_r^{\text{norm}}$ 用作好奇心奖励，驱动对未见任务的目标导向数据采集，仅用 50 条轨迹即可让 350M 世界模型适配新环境，效果接近专家/人类数据采集水平的 90%

## 相关概念

- [[世界模型]]
- [[强化学习]]
