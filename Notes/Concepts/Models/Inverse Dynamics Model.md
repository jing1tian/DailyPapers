---
type: concept
category: Models
tags: [robot-learning, action-labeling]
created: 2026-05-09
---

# Inverse Dynamics Model (IDM)

给定相邻两帧观测 $(o_t, o_{t+1})$ 反推动作 $a_t$ 的模型。常用于把无动作标签的视频（含合成视频）转成可训练的 (obs, action) 对。

## 代表工作

- VPT (OpenAI Minecraft)
- [[RLDX-1]]: 用 IDM 给 Cosmos-Predict2 生成的合成视频打动作标签，扩充 GR-1 humanoid 训练数据。
- [[WMRobotSurvey]]: 系统梳理 IDM 式解耦策略范式（UniPi、VidMan、Vidar、TC-IDM），归纳中间表示从像素→潜在→结构化的演化。
