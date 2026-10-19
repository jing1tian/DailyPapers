---
type: concept
aliases: [CityAnchor method]
---

# CityAnchor

## 定义
CityAnchor 是用于 city-scale 3D 场景语言 grounding 的 candidate ranking 方法，先生成 instance proposals，再对候选集排序选出目标物体的 3D bounding box。

## 核心要点
1. 采用传统 candidate ranking 范式：预计算 instance proposals → 语言-视觉匹配排序
2. 在大规模场景中 proposal 生成计算成本高
3. 作为 city-scale grounding 的强 baseline，被 [[Urbex]] 超越

## 代表工作
- [[Urbex]]：用 agentic active search 取代 candidate ranking，在 [[CityRefer]] 上超越 CityAnchor

## 相关概念
- [[CityRefer]]
- [[SoftGroup]]
- [[InstanceRefer]]
