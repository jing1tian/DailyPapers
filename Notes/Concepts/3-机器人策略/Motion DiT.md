---
type: concept
aliases: [Motion Diffusion Transformer, 运动扩散Transformer]
---

# Motion DiT

## 定义
Motion DiT（运动扩散 Transformer）是以 [[DiT]] 为骨干、专门预测机器人运动潜变量的策略网络，通常与 [[Video DiT]] 耦合，接受视频世界模型的中间预测特征作为条件，通过 [[Flow Matching]] 生成连续动作或离散运动 token。

## 数学形式

$$
\mathcal{L}_{\text{motion}} = \mathbb{E}\!\left[\left\|v_\phi^{\text{motion}}(m_t^{\tau_a}, \tau_a \mid h_t^{\tau_f}, p_t, e) - (\varepsilon_m - m_t^0)\right\|_2^2\right]
$$

## 核心要点
1. **条件输入**: Video DiT 中间特征 $h_t^{\tau_f}$、本体感受状态 $p_t$、具身标签 $e$
2. **输出**: 统一运动潜变量（离散运动 token + 连续末端执行器命令）
3. **训练目标**: [[Flow Matching]] 速度场回归
4. **实时性**: 相比 Video DiT 轻量，可在 Video DiT 单次前向后快速完成推理

## 代表工作
- [[MotionWAM]]: 首次提出 Motion DiT 与 Video DiT 耦合的双 DiT 架构，实现 4.9 Hz 全身人形机器人实时控制

## 相关概念
- [[Video DiT]]: 提供世界预测条件的配对组件
- [[DiT]]: 基础架构
- [[Flow Matching]]: 训练目标
- [[WAM]]: 双 DiT 架构所属的 World Action Model 框架
- [[SONIC]]: Motion DiT 输出解码所使用的全身运动控制器
