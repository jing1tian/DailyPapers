---
type: concept
aliases: [Stable World Model, stable-worldmodel]
---

# SWM (Stable World Model)

## 定义
一个模块化、可复现的世界模型研究生态系统，提供标准化环境、数据收集工具、规划算法和 baseline 实现。

## 核心要点
1. 提供统一的 `World` 接口，兼容用户自有训练代码
2. 环境支持可控变化因子（visual/physical FoV），用于鲁棒性和持续学习研究
3. 内置规划算法（MPC、CEM）和多个 baseline（含 DINO-WM）
4. 发现 DINO-WM 在 Push-T 任务上对视觉/物理 FoV 变化敏感性极高

## 代表工作
- [[stable-worldmodel-v1]]: 原始框架论文，LeCun 团队参与

## 相关概念
- [[DINO-WM]]
- [[MPC]]
- [[PLDM]]
- [[MuJoCo]]
