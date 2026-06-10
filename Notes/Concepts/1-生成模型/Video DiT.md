---
type: concept
aliases: [Video Diffusion Transformer, 视频扩散Transformer]
---

# Video DiT

## 定义
Video DiT（视频扩散 Transformer）是将 [[DiT]]（扩散 Transformer）架构应用于视频生成的模型类别，通过对时序帧序列的潜空间进行去噪，生成时空一致的视频内容，是世界模型和视频预测的重要基础。

## 数学形式

$$
\mathcal{L}_{\text{video}} = \mathbb{E}\left[\left\|v_\theta^{\text{video}}(z_{t+1}^{\tau_v}, \tau_v \mid z_t^0, l) - (\varepsilon_v - z_{t+1}^0)\right\|_2^2\right]
$$

## 核心要点
1. **架构基础**: 基于 [[DiT]] 的 Transformer 骨干，处理时空潜变量序列
2. **训练目标**: [[Flow Matching]] 或 DDPM 等扩散目标
3. **条件信号**: 支持文本指令、前帧图像等多模态条件
4. **中间特征复用**: 在 [[WAM]] 中，Video DiT 中间隐层特征（固定去噪时间步）可作为动作策略的世界预测信号，无需完整迭代去噪

## 代表工作
- [[MotionWAM]]: 以 Cosmos-Predict2.5-2B 作为 Video DiT 骨干，固定 $\tau_f \approx 1$ 提取中间特征驱动 Motion DiT
- [[Cosmos-Predict2]]: 具体的 Video DiT 实现（2B 参数）

## 相关概念
- [[DiT]]: 扩散 Transformer 基础架构
- [[Flow Matching]]: 常用训练目标
- [[WAM]]: Video DiT 作为世界模型组件的应用框架
- [[Motion DiT]]: 与 Video DiT 耦合的动作生成组件
