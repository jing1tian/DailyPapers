---
type: concept
aliases: [GeoPlan, GeoPlan tokens, 几何规划 token]
---

# GeoPlan Token

## 定义

一组可学习的特殊输入标记（learnable tokens），追加在 [[VLA]] 的 [[VLM]] 输入序列末尾，通过冻结的几何基础模型（[[VGGT]]）提供的点图和末端执行器轨迹作为 [[Flow Matching]] 监督目标，将 3D 场景理解内化到 VLM 的 latent 空间，推理时无需任何 3D 预测器。

## 数学形式

$$
\mathbf{c}_t = \mathrm{Proj}(\mathbf{z}_t) \in \mathbb{R}^{K \times D_g}
$$

其中 $\mathbf{z}_t$ 是 VLM 最后一层在 $K$ 个 GeoPlan 标记处的输出状态，$\mathrm{Proj}$ 为线性投影到几何工作空间。

监督目标：

$$
\mathbf{Y}_t = [\mathbf{P}_t;\, \mathbf{W}_t]
$$

$\mathbf{P}_t$：[[VGGT]] 提供的当前帧点图；$\mathbf{W}_t$：未来末端执行器（EEF）轨迹。

## 核心要点

1. **训练时**：冻结 [[VGGT]] 教师网络，辅助 [[Flow Matching]] 损失 $\mathcal{L}_\text{geo}$ 对 GeoPlan token 的投影 $\mathbf{c}_t$ 施加点图 + EEF 轨迹监督。
2. **推理时**：移除几何教师和辅助解码头，GeoPlan token 仅通过 VLM 前向传播，无额外计算开销。
3. **作用**：GeoPlan token 的状态 $\mathbf{c}_t$ 作为几何查询进入三路[[Causal Retrieval Memory|时序记忆]]的几何流，积累跨时间步的 3D 场景信息。
4. **EEF 权重大于场景**：$\lambda_\text{eef}=3.0$ vs $\lambda_\text{scene}=0.5$，强调未来运动预测对操作的重要性。

## 代表工作

- [[GeoMem-VLA]]: 提出 GeoPlan Token 的原始论文

## 相关概念

- [[VGGT]]: 几何教师网络
- [[Flow Matching]]: 用于几何监督的生成建模框架
- [[Causal Retrieval Memory]]: 使用 GeoPlan token 作为几何记忆流的检索机制
- [[VLA]]: 应用场景
