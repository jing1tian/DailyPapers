---
type: concept
aliases: [Action Caching, VLA推理加速]
---

# ActionCache

## 定义
ActionCache：一种无需训练的 VLA 推理加速方法，通过缓存 flow-matching action head 的中间去噪结果并在相邻时间步复用，减少 Number of Function Evaluations（NFE），降低推理延迟。

## 数学形式
$$\hat{a}_t \approx \text{Denoise}(a_{t-1}^{\text{cached}}, z_t), \quad \text{if } \|z_t - z_{t-1}\| < \epsilon$$

当视觉特征变化小于阈值 $\epsilon$ 时，复用前一步的去噪中间状态，跳过部分 NFE 步骤。

## 核心要点
1. 针对 flow-matching based VLA（如 π0 系列）的推理瓶颈：迭代去噪 action head
2. 机器人执行连续动作时相邻帧高度相似，cache 的 reuse 率高
3. 无需任何重训练，即插即用（training-free）
4. 减少约 60% NFE，在 LIBERO 基准上任务成功率损失在可接受范围内
5. 借鉴视频扩散 cache 加速思路（[[DeepCache]]、[[PAB]]）

## 代表工作
- Training-Free Acceleration for VLA Models with Action Caching（今日论文）

## 相关概念
- [[DeepCache]]（视频扩散的 cache 加速先驱）
- [[PAB]]（Pyramid Attention Broadcast，另一类 diffusion cache 加速）
- [[VLA]]（应用场景：flow-matching VLA 推理）
