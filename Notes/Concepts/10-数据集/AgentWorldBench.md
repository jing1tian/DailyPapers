---
type: concept
aliases: [AgentWorldBench, AWB]
---

# AgentWorldBench

## 定义

AgentWorldBench 是专为评估语言世界模型（LWM）仿真质量而设计的综合性基准，包含 2,170 个样本，基于 5 个前沿模型在 9 个已有 benchmark 上的真实交互轨迹构建。

## 数学形式

五维评分（五维平均作为最终分数）：

$$
Score = \frac{1}{5}(Format + Factuality + Consistency + Realism + Quality)
$$

## 核心要点

1. **来源 benchmark**: Terminal-Bench 1.0/2.0、OSWorld-Verified、Tool Decathlon、MCPMark、WideSearch
2. **五维评估体系**:
   - Format（格式）: 符合域的结构约定
   - Factuality（事实性）: 陈述事实是否正确
   - Consistency（一致性）: 内部一致性与跨轮次一致性
   - Realism（真实性）: 仿真与真实环境行为的吻合度
   - Quality（质量）: 相对参考答案的完整性与简洁性
3. **跨裁判一致性高**: Gemini 3 Flash、Claude Sonnet 4.5、GPT-5.2 三个裁判 Spearman 相关系数 ρ = 0.92–0.99
4. 采用参考锚定的 LLM 裁判（Reference-grounded LLM judging），由 GPT-5.2 评分

## 代表工作

- [[QwenAgentWorld]]: 提出并使用该 benchmark，397B 变体取得 58.71 总均值

## 相关概念

- [[Language World Model]]: 评测对象
