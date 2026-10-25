---
type: concept
aliases: [In-Context Learning, 上下文学习, few-shot prompting]
---

# ICL

## 定义
In-Context Learning（上下文学习）：大模型在推理时无需更新参数，仅凭 prompt 中的少量示例（demonstrations）即可泛化到新任务的能力。

## 数学形式
$$p(y | x, \mathcal{D}_{\text{ctx}}) \approx p_\theta(y | x, (x_1, y_1), \ldots, (x_k, y_k))$$

模型以示例对 $(x_i, y_i)$ 为条件，在不更新 $\theta$ 的情况下推断新输入 $x$ 的输出。

## 核心要点
1. 零样本（zero-shot ICL）：只提供任务描述；少样本（few-shot ICL）：提供若干完整示例
2. 与 fine-tuning 的本质区别：权重不变，仅靠 prompt 中的信息驱动行为
3. 在 embodied 场景中，GPT-6-Astra 通过 ICL from robot demonstrations 驱动操控策略
4. ICL 的质量强依赖示例的选取策略（diversity、relevance）和模型的参数规模

## 代表工作
- [[GPT-6-Astra]]: 在 12 个机器人基准上通过 ICL 实现 zero-shot embodied control，与 fine-tuned policy 用同等数量示例对比

## 相关概念
- [[VLA]]: 通常通过 fine-tuning 而非 ICL 适配机器人任务
- [[Chain-of-Thought]]: ICL 的增强变体
