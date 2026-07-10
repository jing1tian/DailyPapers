---
type: concept
aliases: [Robotics Benchmark for Video Generation, 机器人视频生成基准]
---

# RBench

## 定义
专为机器人中心的视频生成交互正确性评估设计的公开自动化基准（Deng et al., ICML 2026），包含 650 个文本-图像提示，从任务正确性和机器人具身两个维度综合评估视频生成模型在具身场景的表现。

## 核心要点
1. **两大评估轨道**：
   - **任务导向（250 条）**：操作（Manipulation）、空间关系（Spatial）、多实体协作（Multi-entity）、长时程规划（Long-horizon）、视觉推理（Visual Reasoning）
   - **具身专属（400 条）**：单臂（Single-arm）、双臂（Dual-arm）、人形（Humanoid）、四足（Quadruped）
2. **评测模型**（截至 2026-07）：LingBot-Video、Cosmos 3 Super、Wan 2.2 A14B/5B、HunyuanVideo 1.5、LongCat-Video、Seedance 1.5、Wan 2.6、Veo 3 等
3. **代表性分数**：LingBot-Video 0.620（开源最高），Wan 2.6 0.607（闭源最高），Cosmos 3 Super 0.581

## 代表工作
- RBench (Deng et al., ICML 2026): 基准论文
- [[LingBot-Video]]: RBench 开源模型第一

## 相关概念
- [[Physics-IQ Verified]]
- [[World Model]]
- [[Video Foundation Model]]
