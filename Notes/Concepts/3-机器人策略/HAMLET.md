---
type: concept
aliases: [Hierarchical Action Memory for Long-horizon Tasks, 长程任务分层记忆]
---

# HAMLET

## 定义
一种为 VLA 模型设计的外部分层记忆管理系统，通过维护结构化的任务记忆来支持长程操作任务。

## 核心要点
1. 将任务历史按层次组织：短期 buffer 记录最近观察，长期 memory 存储关键事件摘要
2. 作为 pretrained VLA 的外挂模块，不改变 VLA 本体权重
3. 与 NativeMEM 等"原生内存"方法对比，属于"外挂 memory manager"路线

## 代表工作
- [[NativeMEM]]: 指出 HAMLET 等外挂方案限制 memory horizon 或反应速度，提出原生压缩替代

## 相关概念
- [[LaMem]]
- [[NativeMEM]]
- [[MemoryVLA]]
