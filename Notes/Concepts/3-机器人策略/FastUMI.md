---
type: concept
aliases: [Fast Universal Manipulation Interface]
---

# FastUMI

## 定义
Universal Manipulation Interface (UMI) 的高效版本，一种无需特定机器人遥操作硬件的手持式数据采集工具，用于大规模采集机器人操作演示数据。

## 核心要点
1. 采用腕装摄像头采集第一视角操作数据，成本低、部署灵活
2. 主要挑战：腕装摄像头导致视角偏移，缺乏物理约束验证
3. [[VISTA]] 专门解决 UMI 数据的视角偏移和物理可行性问题
4. 采集的数据可直接用于 VLA 模型微调

## 代表工作
- [[VISTA]]: 专为 UMI 数据设计的清洗和适配 pipeline

## 相关概念
- [[VISTA]]
- [[VLA]]
- [[DROID]]
