---
type: concept
aliases: [Embodied Question Answering, EQA]
---

# EQA

## 定义
Embodied Question Answering：智能体在三维环境中主动探索，回答关于环境的自然语言问题。

## 数学形式
$$\text{Answer} = f(\text{Environment}, \text{Question}, \text{Navigation Policy})$$

## 核心要点
1. 结合导航（主动感知）和视觉问答（VQA）
2. 智能体需要决定去哪里收集信息来回答问题
3. 与 VLN 的区别：EQA 目标是回答问题，VLN 目标是到达地点
4. 常用环境：MP3D、HM3D、AI2-THOR

## 代表工作
- [[Uni-LaViRA]]：将 EQA 与其他具身任务统一

## 相关概念
- [[VLN]]
- [[VLM]]
- [[具身智能]]
