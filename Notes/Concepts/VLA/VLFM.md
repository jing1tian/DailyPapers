---
type: concept
aliases: [VLFM, Vision-Language Frontier Maps]
---

# VLFM (Vision-Language Frontier Maps)

## 定义
将视觉语言模型（VLM）与传统前沿探索（frontier-based navigation）结合的导航方法，用 VLM 对每个 frontier 评分（与目标物体的语义相关度），引导机器人向最相关区域探索。

## 核心要点
1. Frontier：可探索边界的候选点（已知空间与未知空间的交界）
2. VLM 评分：用 BLIP-2 等 VLM 估计"沿某条路径能否找到目标"
3. 不需要语义地图或预建地图，在线建图 + 在线评分
4. 在 HM3D、MP3D ObjectNav benchmark 上作为强 baseline

## 代表工作
- Shah et al. (2023): "VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation"
- [[OpenFrontier]]: 以 VLFM 为 baseline，用 InternVL3 做更强的语义评估

## 相关概念
- [[VLM]]
- [[OpenFrontier]]
- [[InternVL3]]
