---
type: concept
aliases: [Cascaded Super-Resolution, 级联精炼器, 级联超分辨率]
---

# Cascaded Refiner（级联精炼器）

## 定义
视频/图像生成的两阶段级联设计，基础生成器在低分辨率（如 480p）生成全局语义和运动，独立的精炼器网络将其上采样至高分辨率（如 1080p），专注于高频细节恢复，避免直接高分辨率生成的巨大计算开销。

## 核心要点
1. **分阶段职责分离**：基础生成器负责全局布局和时序一致性，精炼器专用于高频纹理和细节恢复，各自优化不同目标
2. **条件 Rectified Flow**：精炼器不从纯噪声去噪，而学习从退化条件 $x_{lr}$（上采样并添加合成降质）到清晰目标 $x_0$ 的条件流
3. **阈值轨迹**：训练时步限定在 $t \in [0, \tau]$（$\tau \sim [0.85, 0.95]$），推理时从 $t=\tau_{inf}$ 开始去噪，减少全程去噪步骤
4. **合成降质训练**：高斯模糊 + 压缩等降质对下采样视频模拟真实上采样 artifacts，提升精炼器对非完美低分辨率输入的鲁棒性

## 数学形式

训练轨迹：
$$
x_t = \left(1-\frac{t}{\tau}\right)x_0 + \frac{t}{\tau}x_\tau, \quad v^\star = \frac{x_\tau - x_0}{\tau}
$$

推理：从 $x_{\tau_{inf}} = (1-\tau_{inf})x_{lr} + \tau_{inf}\epsilon$ 出发，反向 ODE 积分至 $t=0$。

## 代表工作
- Imagen Video (Ho et al., 2022)
- [[LingBot-Video]]: 480p 基础生成 → 1080p 精炼，显著提升面部细节和 OCR 文字清晰度

## 相关概念
- [[Rectified Flow]]
- [[Flow Matching]]
- [[DiT]]
- [[Video Diffusion Model]]
