---
type: concept
aliases: [A Low-cost Open-source Hardware System for Bimanual Teleoperation]
---

# ALOHA

## 定义
低成本开源双臂遥操作机器人平台，由 Stanford 开发，广泛用于灵巧操作数据采集和 VLA 评估。

## 核心要点
1. 低成本双臂设计（约 $20K）
2. 用于 ACT 等灵巧操作算法的数据采集
3. Mobile ALOHA 扩展支持移动操作
4. 开源机械和软件设计

## 代表工作
- ALOHA (2023): Learning Fine-Grained Bimanual Manipulation
- Mobile ALOHA (2024)
- ACT (2023)
- [[vla.cpp]]: 在 ALOHA 双臂平台上测试 GR00T-N1.6，vla.cpp 推理延迟（470ms）vs PyTorch（620ms），Task 1 成功率从 15% 提升至 90%
- [[SVP-IL]]: 在 Aloha-AgileX 平台上做真实世界双臂评测（摆桌、桌面清理、双臂分拣），平均成功率 60.0%，超过 π₀ 和 DP

## 相关概念
- [[ACT]]
- [[GeoAlign]]
- [[VLA]]
