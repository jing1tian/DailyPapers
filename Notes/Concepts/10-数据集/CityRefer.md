---
type: concept
aliases: [CityRefer dataset, CityRefer benchmark]
---

# CityRefer

## 定义
CityRefer 是用于 city-scale 3D 场景语言 grounding 的数据集和 benchmark，包含户外大规模 3D 点云场景 + 自然语言描述标注，评测模型在城市级环境中定位目标物体的能力。

## 数学形式
$$\text{Hit@1} = \frac{\text{top-1 predictions correct}}{N} \quad \text{IoU} = \frac{|\hat{B} \cap B^*|}{|\hat{B} \cup B^*|}$$

## 核心要点
1. 覆盖城市级尺度 3D 场景（比室内 benchmark 如 ScanRefer 复杂得多）
2. 主要评测指标：Hit@1（准确率）和 3D IoU
3. 与 STPLS3D-Refer 并列为城市级 3D grounding 主流 benchmark

## 代表工作
- [[CityAnchor]]：传统 candidate ranking 方法
- [[Urbex]]：agentic active search 方法，在 CityRefer 上达到最优

## 相关概念
- [[CityAnchor]]
- [[Urbex]]
- [[SoftGroup]]
