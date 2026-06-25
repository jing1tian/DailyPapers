---
type: concept
aliases: [Action-Conditioned WM, 动作条件世界模型, ACWM]
---

# Action-Conditioned World Model（动作条件世界模型）

## 定义
动作条件世界模型描述环境如何响应智能体动作而演化：给定当前状态和执行动作，预测由此产生的未来状态，从而捕捉动作对环境动态的因果影响。

## 数学形式

$$
P(o' \mid o, a)
$$

其中 $o$ 为当前观测，$a$ 为智能体执行的控制信号，$o'$ 为预测的下一状态。

## 核心要点
1. **显式世界模型**: 直接在像素空间预测未来帧（ACVP、CDNA、SV2P 等早期工作）
2. **隐式潜空间动态**: 在紧凑潜空间中学习转移函数（PlaNet RSSM、Dreamer 系列）
3. **与 VLA 的区别**: VLA 学习 $p(a \mid o, l)$，Action-Conditioned WM 学习 $P(o' \mid o, a)$，两者互补

## 代表工作
- [[Dreamer]]: 经典 RSSM 动作条件世界模型，支持潜空间规划
- [[WAM-Survey]]: Sec 3.2.1 对该范式的系统梳理
- [[GE-Sim2]]: 扩展为闭环仿真器，增加本体感知状态预测和奖励判别，实现 omni 仿真
- [[SC3-Eval]]: 在标准正向动力学目标 $\mathcal{W}^{fd}(v_{t+1} \mid v_{1:t}, a_{1:t})$ 之上联合训练逆向动力学与跨视角一致性，把动作条件世界模型从"动作生成工具"改造为"策略评测器"

## 相关概念
- [[World Model]]: 上层概念
- [[Language-Conditioned World Model]]: 以语言而非低级动作为条件的互补范式
- [[WAM]]: 将动作条件世界模型与动作生成统一的更高级范式
- [[Dreamer]]: 代表性实现
