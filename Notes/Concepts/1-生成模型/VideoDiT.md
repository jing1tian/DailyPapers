---
type: concept
aliases: [Video Diffusion Transformer, VideoDiT]
---

# VideoDiT

## 定义
Kairos 架构中负责视频帧生成与预测的 Diffusion Transformer 组件，是 World Model 中"想象未来"能力的核心。

## 数学形式
$$\hat{v}_{t+1:t+T} = \text{VideoDiT}(v_{1:t}, a_{1:t}; \theta_v)$$

以历史视觉帧和动作条件为输入，去噪生成未来 T 帧的视频预测。

## 核心要点
1. Kairos 统一架构三大组件之一（与 ActionDiT、VLM Head 并列）
2. 采用 Hybrid Linear Attention（SWA + GLA）提升长视频建模效率
3. 原生支持多分辨率、多帧率，适应异构机器人数据
4. 训练时与动作预测头联合优化，确保视频生成与动作语义对齐

## 代表工作
- [[Kairos]]: 在 Physical AI World Model 中引入 VideoDiT

## 相关概念
- [[ActionDiT]]: Kairos 中的动作预测组件
- [[DiT]]: 基础架构
- [[DSWA]]: VideoDiT 中使用的动态注意力机制
- [[DreamGen]]: 同类视频生成用于机器人策略
