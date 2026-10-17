---
type: concept
aliases: [Teacher-Student Learning, 教师-学生学习, 知识蒸馏训练范式]
---

# Teacher-Student（教师-学生学习）

## 定义

一种训练范式：**教师模型**（通常参数更多或获取更多信息）提供监督目标，**学生模型**（通常更轻量或获取更少信息）在训练时拟合教师的输出，推理时仅用学生模型。

## 核心要点

1. **信息不对称**: 教师模型在训练时可以访问额外信息（更多数据、更大模型能力、未来帧等），学生模型则只有部分信息，从而学习"浓缩"的表示
2. **推理独立性**: 推理时只需部署学生模型，教师仅用于训练
3. **多种教师来源**:
   - **预训练外部教师**（如知识蒸馏）：使用独立大模型作为教师
   - **[[EMA]] 自教师**（如 [[BYOL]]、[[vlm-jepa]]）：教师由学生参数的指数移动平均构建，无需外部模型
   - **集成教师**：多个模型集成作为教师
4. **[[Stop-Gradient]]**: 教师分支通常不接受反向传播，保持稳定目标

## 数学形式

EMA 自教师更新：

$$
\xi_{\text{teacher}} \leftarrow m \cdot \xi_{\text{teacher}} + (1-m) \cdot \theta_{\text{student}}
$$

学生损失（一般形式）：

$$
\mathcal{L} = d\!\left(f_\theta(x_{\text{student}}),\; \operatorname{sg}(f_\xi(x_{\text{teacher}}))\right)
$$

其中 $d$ 为距离度量（L1/L2/余弦距离），$\operatorname{sg}$ 为 [[Stop-Gradient]]。

## 代表工作

- [[BYOL]]: 使用 EMA 自教师的视觉自监督方法
- [[MoCo]]: 动量对比学习，教师由队列+EMA构建
- [[I-JEPA]] / [[V-JEPA]]: JEPA 系列也采用教师-学生思想
- [[vlm-jepa]] / [[TS-WAM]]: 将 EMA 教师-学生范式用于 VLM 级时序潜在动作学习

## 相关概念

- [[EMA]]: EMA 自教师的参数更新机制
- [[Stop-Gradient]]: 防止梯度流向教师的关键操作
- [[Knowledge Distillation]]: 教师-学生学习的经典应用
- [[JEPA]]: 采用教师-学生思想的联合嵌入预测架构
