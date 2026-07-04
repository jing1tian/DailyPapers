---
type: concept
aliases: [未来潜变量对齐, Future Latent Prediction, 未来帧潜变量预测]
---

# Future Latent Alignment（未来潜变量对齐）

## 定义

一种 VLA 预训练辅助监督策略，要求模型在生成动作的同时预测当前帧后若干步的视觉潜变量特征，从而约束模型理解动作的物理后果与状态转移规律。

## 数学形式

$$
\mathcal{L}_\text{lat} = \|\hat{z}_\text{fut} - z_\text{fut}\|^2
$$

其中：
- $\hat{z}_\text{fut}$：模型预测的未来帧潜变量
- $z_\text{fut}$：冻结视觉编码器（如 [[V-JEPA2|V-JEPA 2]]）提取的真实未来帧特征

## 核心要点

1. **视觉编码器冻结**: 使用预训练好的视觉编码器（如 V-JEPA 2）提取未来帧特征，避免与动作学习相互干扰
2. **注意力 Mask 隔离**: 潜变量 query 只能 attend VLM 上下文和当前帧特征，不能 attend 动作 token，防止走捷径直接从动作预测未来状态
3. **损失权重调度**: 预训练阶段使用 1:1（动作:潜变量）确保足够的监督强度；微调阶段弱化为 0.1:1，作为正则项
4. **未来帧偏移**: 预测偏移约 8 帧的未来帧效果最优；过长偏移（如 32 帧）反而有害
5. **与动作损失互补**: 动作损失监督"做什么动作"，未来潜变量损失监督"动作的结果是什么"

## 代表工作

- [[VLAFlow]]: 提出 MindWPI 和 MindLWPI 范式，系统验证 Future Latent Alignment 对异构数据迁移的提升效果（WidowX Avg: 65.9% → 74.5%）
- [[WorldVLA]]: 类似思路，使用世界模型预测未来状态

## 相关概念

- [[V-JEPA2|V-JEPA 2]]：常用的未来帧特征提取器
- [[Flow Matching]]：VLAFlow 中与之配合的动作生成框架
- [[Meta-Action Space|元动作空间]]：从理论角度解释 Future Latent Alignment 的作用
- [[KV-Cache Conditioning]]：VLAFlow 中实现潜变量条件注入的机制
