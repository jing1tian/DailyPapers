---
type: concept
aliases: [Terminal-Bench, Terminal Benchmark]
---

# Terminal-Bench

## 定义
Qwen-AgentWorld 评估体系中用于终端/命令行（Terminal domain）智能体任务的 benchmark，测试 LWM 模拟终端环境动态的能力。

## 核心要点
1. 属于 Qwen-AgentWorld 7 大 domain 之一（Terminal domain）
2. 评估 LWM 对 shell 命令执行结果（输出、错误信息、状态变化）的预测准确性
3. 与 SWE-bench 等代码执行 benchmark 类似，但侧重环境动态模拟而非代码生成

## 代表工作
- [[QwenAgentWorld]]: 在 Terminal domain 使用 Terminal-Bench 评估语言世界模型

## 相关概念
- [[AgentWorldBench]]: 更全面的 Agent 评估框架
- [[Claw-Eval]]: 机器人领域的同类 benchmark
