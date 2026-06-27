---
type: concept
aliases: [Reflector, LLM-based Reflection Module, 反思模块]
---

# LLM反思模块

## 定义
一个轻量级模块，接收当前状态、刚执行的最优动作、执行偏差和原始语言指令，输出结构化的纠正性指导 token，用于在线诊断 VLA 执行失败并指导后续重采样。

## 数学形式

$$
\mathcal{R}_\omega: (s_t, a_{best}, \delta_t, \ell) \to g
$$

训练目标（反思学习阶段）：

$$
\mathcal{L}_{ref} = \lambda_{text} \mathcal{L}_{text} + \lambda_{act} \mathcal{L}_{act}
$$

其中 $\mathcal{L}_{text}$ 训练模块输出与教师 VLM 标注一致的指导 token，$\mathcal{L}_{act}$ 训练基础策略在增强指令下生成正确的纠正动作。

## 核心要点
1. 监督信号来自教师 VLM 对失败样本的标注（具体教师模型型号通常未公开）
2. 输入包含失败上下文的四元组：状态、已执行动作、偏差、原始指令
3. 输出的指导 token 以文本形式拼接进指令，而非直接修改动作空间，保持与任意 VLA 策略的兼容性
4. 是 [[反思引导的重采样]] 机制的核心组件

## 代表工作
- [[PhysReflect-VLA]]: 提出该模块，与可行性筛选结合构成闭环自我修正控制管线

## 相关概念
- [[反思引导的重采样]]
- [[执行偏差]]
- [[Vision-Language-Action Model]]
