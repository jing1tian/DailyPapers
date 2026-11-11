---
type: concept
aliases: [Foundation Tactile Policy, FTP1]
---

# FTP-1（Foundation Tactile Policy）

## 定义

一种通用触觉策略基础模型，将异构传感器观测通过形态感知（morphology-aware）token 空间统一表示，支持通过传感器专属编码器适配至新硬件，用于接触丰富操作的基础策略预训练。

## 核心要点

1. **形态感知 token**：将不同形状/尺寸的传感器阵列映射为统一 token 空间，保留形态信息
2. **传感器专属编码器适配**：每引入一个新传感器（包括已有传感器的测试），需要为该传感器单独训练一个输入适配器（encoder adapter）并与探针头联合训练
3. **局限性**：在未见传感器（XHand1）上，即使配备专属适配器，各项接触物理探针误差仍显著高于 LingBot-Touch（无需任何适配器）：力恢复 3.72 N vs 1.32 N，接触中心 5.89 mm vs 0.16 mm
4. **跨传感器迁移差距**：在触觉到视频检索任务上，FTP-1（XHand1）Top-5 Recall 仅 6.0%，而 LingBot-Touch 达 47.1%

## 代表工作

- Yuan et al., "FTP-1: A generalist foundation tactile policy across tactile sensors for contact-rich manipulation," arXiv 2606.13102, 2026

## 相关概念

- [[LingBot-Touch]]: 在所有探针任务和操作任务上超越 FTP-1，且无需传感器专属适配
- [[触觉传感器]]: FTP-1 覆盖磁性等多类传感器
- [[Dexterous Manipulation]]: FTP-1 的目标应用场景
