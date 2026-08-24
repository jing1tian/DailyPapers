---
type: concept
aliases: [SWE-bench, SWE-Bench Verified]
---

# SWE-Bench

## 定义
一个软件工程 agent 评测 benchmark，每条样本是一个真实 GitHub issue + 对应代码库，要求 agent 自动完成 bug 修复或功能实现，以 "SWE-Bench Verified" 子集为标准版本（经人工筛选排除不可解 issue）。

## 数学形式
评测指标：$\text{Resolved Rate} = \frac{\text{通过所有测试用例的 issue 数}}{\text{总 issue 数}} \times 100\%$

## 核心要点
1. 所有任务来自真实开源项目，避免数据泄漏和 cherry-picking
2. Verified 版本过滤了 ambiguous/不可解 issue，使评测更可靠
3. LLM-as-a-Verifier 在 SWE-Bench Verified 上达到 78.2%，超过多数 agent baseline

## 代表工作
- [[LLM-Verifier]]: 用 SWE-Bench Verified 验证验证框架的泛化性

## 相关概念
- [[Terminal-Bench]]: 另一个 agent 能力 benchmark
- [[MedAgentBench]]: 医疗领域 agent benchmark
