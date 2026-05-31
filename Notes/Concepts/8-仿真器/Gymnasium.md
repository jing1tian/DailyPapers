---
type: concept
aliases: [OpenAI Gym, gym, gymnasium]
---

# Gymnasium

## 定义

Gymnasium（原 OpenAI Gym）是强化学习中最广泛使用的标准化环境接口库，定义了 `reset() / step(action)` 的统一 API，使得 RL 算法可以不修改代码地在不同环境间切换。

## 标准接口

```python
env = gymnasium.make("CartPole-v1")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

每次 `step` 返回五元组：观测、奖励、是否终止、是否截断、辅助信息。

## 核心要点

1. **标准化契约**: 定义了 RL 研究的通用接口，是事实上的行业标准
2. **环境生态**: 支持 Atari、MuJoCo、Box2D 等数百个环境
3. **向量化支持**: `VectorEnv` 支持批量并行环境
4. **与 dm_control 的关系**: [[DMControl]] 提供了 Gymnasium 兼容的包装器

## stable-worldmodel 的设计差异

[[stable-worldmodel]] 的 `World` 类**有意偏离** Gymnasium 标准：

- Gymnasium: `step()` 直接返回 `(obs, reward, done, info)` 元组
- SWM: `step()` 不返回值，所有数据存入 `world.infos` 字典，由策略主动查询

这种设计将控制逻辑（策略）与环境执行完全解耦，更适合 WM 研究中复杂的规划与评估需求。

## 代表工作

- [[stable-worldmodel]]: 在 Gymnasium 基础上构建了更适合 WM 研究的 World 接口
- [[DINO-WM]]: 使用 Gymnasium 接口的世界模型框架

## 相关概念

- [[DMControl]]: 常通过 Gymnasium 包装器使用
- [[MuJoCo]]: Gymnasium 的重要环境后端
- [[世界模型]]: WM 研究的标准环境接口
