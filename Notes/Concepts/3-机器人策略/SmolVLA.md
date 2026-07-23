---
type: concept
aliases: [SmolVLA Model]
---

# SmolVLA

## 定义
一个轻量级 VLA 模型，设计目标是低参数量下保持竞争性的操作性能，支持在资源受限设备上部署。

## 核心要点
1. 参数量显著小于 OpenVLA、π0 等大型 VLA
2. 用 action tokenization 方式表示动作（与 [[MVA]] 形成对比）
3. 由 Hugging Face 的 LeRobot 生态支持

## 代表工作
- HuggingFace LeRobot Team: SmolVLA

## 相关概念
- [[VLA]]（所属大类）
- [[OpenVLA]]（大型 VLA 基线）
- [[LeRobot]]（所在框架）
- [[MVA]]（不同动作表示方法的对比基线）
