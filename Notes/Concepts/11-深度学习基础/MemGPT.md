---
type: concept
aliases: [MemGPT, Memory-GPT]
---

# MemGPT

## 定义
为 LLM 引入分层记忆管理系统（短期工作记忆 + 长期外部存储），让 LLM 能够处理超出 context 长度的任务，类比操作系统的虚拟内存机制。

## 核心要点
1. 短期记忆：LLM context window
2. 长期记忆：外部数据库，按需召回
3. 控制逻辑：模型自行决定何时 read/write 长期记忆
4. 常用于 long-horizon navigation 和 agent 任务

## 代表工作
- [[HAM-VLN]]: 借鉴 MemGPT 分层记忆结构用于 VLN

## 相关概念
- [[LLM]]
- [[RAG]]
