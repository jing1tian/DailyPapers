---
type: concept
aliases: [OPG, 正交投影引导]
---

# Orthogonal Projection Guidance

## 定义
CofactVLA 提出的动作级因果干预方法：将事实分支速度场中与视觉偏差（反事实速度场）共线的分量剔除，保留与视觉偏差正交的纯语义驱动分量，并以引导尺度 $\gamma$ 放大，得到最终因果干预速度场。

## 数学形式

$$
v_\text{proj} = \frac{\langle v_\text{cond}, v_\text{uncond} \rangle}{\|v_\text{uncond}\|_2^2 + \varepsilon} \cdot v_\text{uncond}
$$

$$
v_\perp = v_\text{cond} - v_\text{proj}
$$

$$
v_\text{causal} = v_\text{cond} + \gamma \cdot v_\perp
$$

## 核心要点
1. **几何直觉**: $v_\perp$ 是事实速度在视觉偏差方向上的正交补，代表纯语义意图
2. **与 CFG 的区别**: CFG 直接相减可能超出动作流形边界；OPG 保持在事实速度附近，仅去除共线分量
3. **超参数**: 引导尺度 $\gamma = 2.0$ 为最优值（过大损害任务多样性）
4. **在 Flow Matching 中适用**: 利用速度场与得分函数的线性等价关系，可在速度场域直接操作

## 代表工作
- [[CofactVLA]]: 提出 OPG，在 LIBERO 仿真和真实机器人实验中验证

## 相关概念
- [[Dual-path Deconfounding Graph]]
- [[Flow Matching]]
- [[Classifier-Free Guidance]]
- [[Score Function]]
