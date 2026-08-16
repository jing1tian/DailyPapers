---
type: concept
aliases: [SLIM, Action-Grounded Predictive Latents]
---

# SLIM-0.5B

## 定义
一个 0.5B 参数的紧凑机器人操作策略，通过 IDM（逆动力学模型）+ FDM（前向动力学模型）学习 action-grounded 潜变量表征，在计算受限场景下达到与大型 VLA 相当的性能。

## 核心要点
1. **Action-Grounded Latents**：IDM 推断动作、FDM 预测下一观察，两者共同约束 latent 空间只保留与 manipulation 相关的信息
2. **MoT 骨干**：使用轻量 Mixture-of-Tokens 架构而非大型 VLM backbone
3. 在 LIBERO、CALVIN ABC 和真实机械臂 5 任务（含 OOD：distractor、背景、光照）上验证
4. has_real_world=True，OOD 实验设计合理

## 适用场景
计算资源受限的机器人部署（边缘设备、消费级 GPU、嵌入式平台）。

## 代表工作
- SLIM-0.5B: Learning Action-Grounded Predictive Latents for Robot Manipulation (2026)

## 相关概念
- [[SmolVLA]]
- [[MoT]]
- [[IDM]]
- [[WorldVLA]]
