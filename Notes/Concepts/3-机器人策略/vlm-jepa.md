---
type: concept
aliases: [VLM-JEPA, VLM JEPA, 视觉语言模型联合嵌入预测架构]
---

# vlm-jepa

## 定义

将 [[JEPA]]（联合嵌入预测架构）提升到 **VLM（视觉-语言模型）语义级别** 的时序潜在动作学习模块，利用 [[EMA]] 目标分支（教师）在无标注视频上提供时序自监督信号，无需独立预训练阶段。

## 数学形式

在线分支（学生）预测潜在动作：

$$
\hat{z}_t = P_\phi(X_t, Q_Y), \quad \hat{z}_t \in \mathbb{R}^{Q_a \times d_z}
$$

EMA 教师更新：

$$
\xi \leftarrow m\xi + (1-m)\theta_Y
$$

时序对齐损失（L1）：

$$
\mathcal{L}_{\text{temp}} = \left\|\hat{z}_t - \operatorname{LN}_z(\operatorname{sg}(z_t^Y))\right\|_1
$$

## 核心要点

1. **非对称设计**: 在线 VLM 只看当前帧，EMA VLM（教师）同时看当前+未来帧，使推理时无需未来帧输入
2. **VLM 级别**: 相比 [[I-JEPA]]（patch 级）和 [[V-JEPA]]（帧级），将预测目标提升到 VLM 语义嵌入空间，结合语言指令提供任务感知归纳偏置
3. **可学习查询令牌**: 通过 $Q_a$ 个可学习动作查询令牌 $Q_Y$ 提取任务相关的时序变化信息
4. **[[Stop-Gradient]]**: 教师分支采用停止梯度，防止模式崩溃，稳定训练
5. **推理零开销**: 教师分支训练后完全移除，不引入额外参数或延迟

## 代表工作

- [[TS-WAM]]: vlm-jepa 的提出工作，与 [[delta-lwm]] 联合用于端到端 WAM 训练

## 相关概念

- [[JEPA]]: 联合嵌入预测架构基础范式
- [[V-JEPA]]: 视频 JEPA，vlm-jepa 的直接前驱
- [[I-JEPA]]: 图像级 JEPA
- [[EMA]]: 教师分支的参数更新机制
- [[Stop-Gradient]]: 防止梯度回传到教师的关键技术
- [[delta-lwm]]: 与 vlm-jepa 联合的潜在世界模型模块
- [[Latent Action]]: vlm-jepa 输出的潜在动作表示
- [[Teacher-Student]]: vlm-jepa 采用的整体训练范式
