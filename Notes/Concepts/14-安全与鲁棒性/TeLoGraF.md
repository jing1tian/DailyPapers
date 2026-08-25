---
type: concept
aliases: [Temporal Logic Graph Framework, STL Graph Encoder]
---

# TeLoGraF

## 定义
基于语法图（syntax graph）的 Signal Temporal Logic (STL) 编码框架，将 STL 公式结构化表示为图嵌入，供 VLA 模型在推理时条件化使用。

## 核心要点
1. 将 STL 公式解析为语法树，转换为图结构
2. 图节点表示 STL 的操作符和原子命题，边表示逻辑/时序关系
3. 输出可与 VLA 的视觉-语言特征融合的结构化嵌入

## 代表工作
- [[Logic-VLA]]: 在 VLA 中使用 TeLoGraF 编码 STL 约束，实现形式化要求驱动的机器人控制

## 相关概念
- [[STL]]（Signal Temporal Logic）
- [[OpenVLA]]
- [[IPO]]
