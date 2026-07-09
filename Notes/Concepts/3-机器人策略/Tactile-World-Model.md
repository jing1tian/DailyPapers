---
type: concept
aliases: [触觉世界模型, Visuo-Tactile World Model]
---

# Tactile World Model（触觉世界模型）

## 定义
以触觉信号为主要感知模态，预测未来触觉状态（或接触动态）的世界模型，用于机器人接触密集操作中的失败检测和自修正。

## 核心要点
1. 输入：RGB 图像 + 当前触觉信号；输出：未来触觉状态预测
2. 触觉 WM 作为"哨兵"：监测 VLA 策略的接触质量，检测偏差后触发修正
3. 比纯视觉 WM 对接触扰动更敏感，能捕捉视觉不可见的失败模式
4. 训练数据来源：通过自适应触觉采集或合成，无需大量人工标注

## 代表工作
- [[TACO]]：触觉 WM 作为 self-corrector，通过 Advantage-Conditioned 策略做 VLA 后训练
- [[Dream-Tac]]：触觉 WM 早期探索
- [[VT-WAM]]：视触觉 World Action Model

## 相关概念
- [[Predictive-Coding]]
- [[Contact-Rich Manipulation]]
- [[World Model]]
