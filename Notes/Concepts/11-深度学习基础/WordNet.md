---
type: concept
aliases: [WordNet, 词网, English WordNet]
---

# WordNet

## 定义

WordNet 是英语的大型词汇数据库（Miller, 1992），将词汇组织成**同义词集（synset）**，并通过有语义的关系（上位词/下位词/同义词等）连接。广泛用于 NLP 任务的语义相似度计算和文本增强。

## 核心结构

- **Synset（同义词集）**: 表达同一概念的词组（如 {apple, eating apple} 同属一 synset）
- **Hypernym（上位词）**: 语义更宽泛的词（apple → fruit → food）
- **Hyponym（下位词）**: 语义更具体的词（fruit → apple, mango, ...）
- **Shortest path length**: 两 synset 之间图距离，路径长度 1 表示直接语义邻居

## 在机器人/NLP 中的应用

**原则性词替换（Principled Word Substitution）**: 通过选择路径长度为 1 的语义邻居词进行替换，生成语义相近但措辞不同的指令，用于测试语言理解的鲁棒性。

例：
- 原始: *Pick up the **apple** and **put** it on the **bowl**.*
- 替换: *Pick up the **eating apple** and **set** it on the **vessel**.*

## 代表工作

- [[VLA-Arena]] (2512.22539): 使用 WordNet 构建 W0-W4 语言扰动梯度，测试 VLA 的语义定位能力

## 相关概念

- [[CBDDL]]: VLA-Arena 中基于 WordNet 的语言扰动任务规范
- [[VLA]]: 被 WordNet 语言扰动评测语义理解能力的模型
