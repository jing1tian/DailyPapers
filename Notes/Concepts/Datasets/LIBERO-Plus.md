---
type: concept
aliases: [LIBERO-Plus, LIBERO Plus]
---

# LIBERO-Plus

## 定义
LIBERO 系列机器人操作 benchmark 的扩展版，在原始 LIBERO 基础上增加多类视觉和语言扰动（共 7 种），用于系统性评测 VLA 模型的鲁棒性。

## 核心要点
1. 7 种扰动维度：Camera（视角）、Robot（机器人型号）、Language（指令改写）、Light（光照）、Background（背景）、Noise（图像噪声）、Layout（场景布局）
2. 两种评测设置：Zero-shot 泛化（仅在 LIBERO 上训练）和 Disturbance-Exposed（在 LIBERO-Plus 上微调）
3. 训练数据量：15,874 条成功演示
4. 每种扰动独立评测，便于分析模型各维度鲁棒性

## 代表工作
- [[RoVLA]]: 在 LIBERO-Plus 上系统验证三类一致性约束，达到 82.0% 总体成功率
- [[OpenVLA-OFT]]: 在 LIBERO-Plus 上 79.5%（干扰暴露设置）

## 相关概念
- [[VLA]]
- [[LIBERO]]
- [[RoboTwin 2.0]]
