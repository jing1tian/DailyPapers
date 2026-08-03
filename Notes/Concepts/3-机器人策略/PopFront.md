---
type: concept
aliases: [Pop-Front, Queue Pop Front]
---

# PopFront

## 定义
SmolVLA 异步推理框架中的队列管理策略：从动作队列头部弹出下一步执行的动作，同时在后台异步生成下一批动作块，实现"边执行边预测"的连续控制。

## 数学形式
$$a_t = \text{queue.pop\_front}(), \quad \text{while } |\text{queue}| < k: \text{async\_predict}()$$

## 核心要点
1. 与 [[AsyncInfer]] 配合实现异步 VLA 推理
2. 机器人按固定频率从队列头部消费动作，推理线程异步填充队列
3. 避免 VLA 模型推理延迟阻塞机器人执行
4. 是 SmolVLA 实现低延迟、低成本部署的关键机制之一

## 代表工作
- [[SmolVLA]]: "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics" (2506.01844)

## 相关概念
- [[AsyncInfer]]
- [[SmolVLA]]
- [[TurboVLA]]
