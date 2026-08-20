---
type: concept
aliases: [MolmoAct 2, MolmoAct-2]
---

# MolmoAct2

## 定义
基于 Molmo 视觉语言模型的第二代动作生成 VLA，整合了 open-vocabulary 感知与精细化动作预测，在多个操作基准上达到竞争力水平。

## 核心要点
1. 以 Molmo VLM 为骨干，在大规模机器人数据上微调为 VLA
2. 支持多种动作表示（末端执行器、关节角度）
3. 在 LIBERO-VIFO、EATR-Stereo 等基准中作为被评估对象出现

## 代表工作
- [[LIBERO-VIFO]] (2608.17600): 评估 MolmoAct2 的视觉线索遵从安全性
- [[Hydra-0]] (2608.18077): 对比基线之一

## 相关概念
- [[VLA]]
- [[OpenVLA]]
