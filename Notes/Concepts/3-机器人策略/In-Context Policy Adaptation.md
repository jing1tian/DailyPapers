---
type: concept
aliases: [In-Context Policy Adaptation, 上下文策略适配, In-Context Adaptation]
---

# In-Context Policy Adaptation

## 定义

In-Context Policy Adaptation 是一种在推理时利用近期观测-动作历史（context chunk）作为条件来动态适配机器人操控策略的方法，类似于 LLM 的 in-context learning，通过提供少量历史轨迹示例让策略在无需梯度更新的情况下适应新具身或新场景。

## 核心要点

1. **上下文条件化推理**: 策略以结构化具身提示（机器人平台、控制频率、执行速度）和最近 $N$ 步观测-动作对作为上下文，生成当前动作
2. **随机上下文采样（Stochastic Context Sampling）**: 训练时随机采样上下文长度和来源，防止策略退化为简单复制历史动作的平凡解
3. **跨具身零样本迁移**: 通过提供目标具身的少量示范轨迹作为上下文，无需微调即可迁移到新平台
4. **行为对齐功能**: 结合具身提示（embodiment prompt）解决不同数据源的执行风格差异（速度、频率）

## 应用场景

- 新机器人平台的快速适配（few-shot cross-embodiment transfer）
- 反应式错误恢复（reactive recovery）：失败后将失败+重试历史作为上下文，引导策略自主恢复
- 长时域任务的在线状态跟踪

## 代表工作

- [[Qwen-RobotManip]] (2026): 将 In-Context Policy Adaptation 与结构化具身提示结合，在 ARX ALOHA 跨具身迁移上达到 55.0%（4× 消融基线）

## 相关概念

- [[Action Chunking]]: 历史动作块作为上下文的基本单位
- [[Cross-Embodiment]]: 上下文适配是实现跨具身迁移的关键机制
- [[视觉-语言-动作模型]]: In-Context Policy Adaptation 在 VLA 框架下的应用
