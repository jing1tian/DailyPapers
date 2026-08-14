---
type: concept
aliases: [Dynamics Register, Latent-Action-Supervised Dynamics Registers, 动力学寄存器]
---

# Dynamics Registers（动力学寄存器）

## 定义

在 World Action Model 中，通过冻结的 Latent Action World Model（LaWM）教师监督，训练一组紧凑的 Register Token 学习捕获交互引发的视觉转换（物体运动、接触变化、任务进度），为直接策略的动作预测提供显式的动力学上下文。

## 数学形式

$$
\mathcal{L}_{LA} = \Bigl\|g_\psi\!\Bigl(\frac{1}{N_D}\sum_{i=1}^{N_D} D_i\Bigr) - \operatorname{sg}(z_{LA})\Bigr\|_2^2
$$

其中 $g_\psi$ 为可训练投影头，$N_D$ 为寄存器数量，$D_i$ 为第 $i$ 个寄存器激活，$z_{LA}$ 为教师的潜在动作目标，$\operatorname{sg}$ 为 stop-gradient 算子。

## 核心要点

1. **知识蒸馏监督**: 使用冻结 [[LaWAM|LaWM]] 教师将视觉转换 $(o_t, o_{t+1})$ 编码为 32 维潜在动作目标，作为寄存器的蒸馏目标
2. **紧凑表示**: 默认 $N_D = 16$ 个寄存器，通过均值池化聚合后映射到潜在动作空间
3. **训练-推理非对称**: 教师和真实未来帧仅训练时使用，推理时寄存器自主捕获动力学信息
4. **与 Future-KV 互补**: 两者均提升鲁棒性（各自约 +5 分），组合后达到 +8 分，呈现协同效应

## 代表工作

- [[ForeWAM]]: 提出 Latent-Action-Supervised Dynamics Registers，在 LIBERO-Plus 摄像头视角扰动类别上提升 +46.1 分

## 相关概念

- [[Register Token]]: 设计灵感来源，Dynamics Registers 是 Register Token 在 WAM 动力学建模中的具体应用
- [[Future-KV]]: 与 Dynamics Registers 互补的 KV 预填充机制
- [[Knowledge Distillation]]: 使用 LaWM 教师进行潜在动作蒸馏的训练范式
- [[LaWAM]]: 提供潜在动作目标的冻结教师模型
- [[Latent-Action]]: Dynamics Registers 所学习的潜在动作表示
