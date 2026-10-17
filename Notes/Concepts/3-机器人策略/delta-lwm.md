---
type: concept
aliases: [Delta-LWM, 增量潜在世界模型, Latent World Model with Delta]
---

# delta-lwm

## 定义

一种**增量潜在世界模型**，通过预测冻结视觉编码器（如 [[DINOv3]]）的特征增量 $\Delta F_t = F_{t+1} - F_t$ 来将潜在动作与可验证的视觉变化绑定，并结合 [[Ranking Loss|批次不匹配排名损失]] 确保解码器对潜在动作的真实依赖。

## 数学形式

特征增量预测：

$$
\Delta F_t = F_{t+1} - F_t, \quad \widehat{\Delta F}_t = D_\psi(F_t, \hat{z}_t)
$$

世界模型损失（双路：教师 + 学生）：

$$
\mathcal{L}_{\text{world}} = \left\|D_\psi(F_t, z_t^Y) - \Delta F_t\right\|_2^2 + \left\|D_\psi(F_t, \hat{z}_t) - \Delta F_t\right\|_2^2
$$

批次不匹配排名损失（强制动作依赖）：

$$
\mathcal{L}_{\text{dep}} = \left[\mu + \left\|D_\psi(F_t, \hat{z}_t) - \Delta F_t\right\|_2^2 - \left\|D_\psi(F_t, \tilde{z}_t) - \Delta F_t\right\|_2^2\right]_+
$$

预测未来特征：

$$
\hat{F}_{t+1} = F_t + \widehat{\Delta F}_t
$$

## 核心要点

1. **特征增量而非像素重建**: 在冻结 [[DINOv3]] 的语义特征空间内预测变化量，避免像素级生成的高开销，且语义特征更利于动作生成
2. **双路监督**: 同时用教师 VLM 的潜在动作 $z_t^Y$ 和学生预测的 $\hat{z}_t$ 计算世界损失，确保两个分支输出都与视觉变化对齐
3. **[[Ranking Loss]] 依赖约束**: 利用 batch 内置换的潜在动作作为负样本，以 hinge 损失强制解码器在正确配对的潜在动作下产生更低误差，防止"走捷径"
4. **条件化动作生成**: 预测的未来特征 $\hat{F}_{t+1}$ 作为 [[Flow Matching|流匹配]] 动作头的条件输入，形成"潜在动作→视觉预测→动作生成"闭环
5. **训练-推理一致性**: 训练时混合预测/真实未来特征，推理时全用预测特征，混合比率缓解分布偏移

## 代表工作

- [[TS-WAM]]: delta-lwm 的提出工作，与 [[vlm-jepa]] 联合形成端到端世界-动作模型

## 相关概念

- [[vlm-jepa]]: 提供潜在动作 $\hat{z}_t$ 的上游模块
- [[DINOv3]]: 提供冻结空间特征网格的视觉编码器
- [[Ranking Loss]]: 确保潜在动作条件依赖性的对比训练技术
- [[Latent Action]]: delta-lwm 以潜在动作为条件预测特征变化
- [[World Action Model]]: delta-lwm 所属的整体范式
- [[Flow Matching]]: 以 $\hat{F}_{t+1}$ 为条件的动作生成方法
- [[DINO-WM]]: 同样使用 DINO 特征的世界模型方法
