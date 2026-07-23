---
type: concept
aliases: [Guide Think Act, GTA VLA, 交互式体现推理 VLA]
---

# GTA-VLA

## 定义
一种交互式 VLA 框架，通过用户提供的空间 visual cue 引导机器人策略的空间感知推理，采用结构化 CoT + 异步 Flow-Matching 动作生成。

## 数学形式
三阶段流程：
1. **Guide**: 用户提供空间 prior $s = \{(\text{type}_i, \text{coords}_i)\}$（点击/框选/区域）
2. **Think**: 基于 $s$ 生成结构化 CoT $c = (\text{SUBTASKS}, \text{CURRENT}, \text{LOC})$
3. **Act**: 条件化 Flow-Matching 生成动作 $a_t = f_\theta(o_t, c)$

异步解耦：CoT 推理速率与动作生成速率分离。

## 核心要点
1. 结构化 CoT（字段固定）vs. 自由文本 CoT：前者在 OOD 上显著更鲁棒
2. 空间 cue 以视觉先验形式注入，使分布外物体位置可指定
3. 异步 flow-matching：推理（慢）和执行（快）解耦，不相互阻塞
4. 在 SimplerEnv 的四维 OOD 扰动上做了系统评测

## 代表工作
- Ling et al. 2026: GTA-VLA（Futian Lab + HIT）

## 相关概念
- [[OpenVLA]]（基线对比）
- [[Flow-Matching]]（动作生成方式）
- [[SimplerEnv]]（评测平台）
- [[VLA]]（所属大类）
