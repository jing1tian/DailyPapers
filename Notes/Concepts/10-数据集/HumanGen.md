---
type: concept
aliases: [HumanGen, HumanGen Dataset]
---

# HumanGen

## 定义

Zero-WAM 提出的自动化人机配对 ICL 数据集，包含 74.2K 个（人类视频, 机器人轨迹）配对样本，覆盖 8.6K 个操作任务，通过 VLM + 图像编辑 + 视频生成流水线全自动构建，无需人工采集。

## 核心要点

1. **规模**: 74.2K 配对样本，8.6K 任务类别，远超已有人机配对数据集（MIME ~3K、RH20T ~44K 但任务类别 <500）。
2. **四个子集**:
   - Pre-train ICL (External): 41,188 对，来自 5 个公开数据集（AgiBot、InternData-A1、Open-X-Embodiment、RoboCOIN、RoboMIND），覆盖 45+ 机器人形态
   - Pre-train ICL (In-house): 30,247 对，来自内部双臂 Franka、Galaxea R1 Pro 平台
   - Simulation ICL: 针对 RoboTwin 2.0 仿真任务
   - Real-world ICL: 针对真实世界适配
3. **生成流水线**（四阶段）:
   - VLM 分析机器人视频语义（Gemini 3.1 Pro / Qwen3.6-Plus）
   - 图像编辑模型生成多样化人类场景初始帧（Nano Banana 2 / Qwen-Image-2.0）
   - 视频生成模型合成人类操作序列（Wan 2.7 / Kling AI 3.0）
   - VLM 验证语义一致性和物理合理性

## 代表工作

- [[Zero-WAM]]: HumanGen 的提出论文，用于训练 ICL 机器人策略

## 相关概念

- [[In-Context Learning]]
- [[World Action Model]]
- [[In-Context Future Chunk Prediction]]
- [[Wan2.2-TI2V-5B]]
