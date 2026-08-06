---
type: concept
aliases: [Flow Hijack Attack]
---

# FlowHijack

## 定义
针对 flow-matching VLA 的对抗攻击方法，通过干扰 ODE 积分轨迹来劫持动作生成过程。

## 核心要点
1. flow-matching VLA（如 pi0）用 ODE 积分生成动作
2. FlowHijack 通过扰动初始条件或中间状态来 derail 轨迹
3. 是 [[DRIFT]] 的前驱/同类工作

## 代表工作
- [[DRIFT]]: 进一步证明 flow-matching VLA 可被 trajectory-level 攻击

## 相关概念
- [[EDPA]]
- [[对抗补丁攻击]]
