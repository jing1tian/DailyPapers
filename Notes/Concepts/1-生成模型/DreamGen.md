---
type: concept
aliases: [DreamGen]
---

# DreamGen

## 定义

一种基于大规模视频扩散模型的机器人演示生成方法，输入文本描述和任务指令即可生成机器人操作视频演示，再通过逆动力学模型提取动作序列用于策略训练。

## 核心要点

1. **联合生成**: 同时生成机器人和环境，无需渲染机器人作为条件
2. **逆动力学提取**: 从生成视频中用逆动力学模型（IDM）抽取动作标签，间接获得策略训练数据
3. **局限性**: 联合生成方式易出现 embodiment hallucination（生成的机器人形态或动力学偏离真实机器人），限制了生成数据的可用性

## 代表工作

- [[RoboDream]]: 对比基线，RoboDream 通过渲染机器人条件化避免了 embodiment hallucination

## 相关概念

- [[视频扩散模型]]
- [[AnchorDream]]
- [[RoboDream]]
