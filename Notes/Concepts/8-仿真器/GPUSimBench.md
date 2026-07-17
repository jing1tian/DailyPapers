---
type: concept
aliases: [GPUSimBench]
---

# GPUSimBench

## 定义
针对 GPU 加速仿真器在具身 AI 中的可扩展性和可靠性的系统性基准测试框架。

## 核心要点
1. 评测对象：MuJoCo、MJX（JAX/XLA）、IsaacLab、ManiSkill3/PhysX、MuJoCoWarp、PyBullet
2. 评测维度：仿真 throughput（env/s）、物理精度（EMD）、多 GPU 可扩展性
3. 架构分类：XPBD、ECS（Entity Component System）、JAX 编译三类
4. 核心发现：throughput vs accuracy 存在明显 trade-off

## 代表工作
- GPUSimBench（2607.13059, Shanghai AI Lab + NTU）: 首个系统对比 GPU 加速仿真器的基准

## 相关概念
- [[MuJoCo]]
- [[IsaacLab]]
- [[ManiSkill3]]
