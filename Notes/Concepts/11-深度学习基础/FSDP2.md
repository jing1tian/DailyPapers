---
type: concept
aliases: [FSDP2, Fully Sharded Data Parallel v2, 全分片数据并行 v2]
---

# FSDP2（Fully Sharded Data Parallel v2）

## 定义

PyTorch 的第二代全分片数据并行（ZeRO-3 风格）训练后端：将模型参数、梯度和优化器状态切分到各数据并行 rank 上，每个模块仅在需要本地计算时临时materialize 出完整参数，大幅降低单 GPU 的模型状态显存占用。

## 核心要点

1. **ZeRO-3 风格切分**：参数（P）、梯度（G）、优化器状态（OS）三者均切分到各 rank，而非像 ZeRO-1/2 那样只切分优化器状态或梯度，显存节省最彻底
2. **按需 materialize**：每个模块在前向/反向计算时才临时聚合出完整参数，计算结束后立即释放，用通信换显存
3. **支持多网络协同训练**：使同时训练学生、教师、fake-score、EMA 等多个大型网络的蒸馏流水线在显存受限的 GPU 上变得可行
4. **分布式检查点（DCP）**：直接跨 rank 保存/恢复分片后的模型与优化器状态，无需先聚合到单进程，适合大规模视频 DiT 训练
5. **与 JVP 等高阶算子的协同**：在涉及前向模式自动微分（[[JVP]]）等需要逐层处理的场景，推荐按层实现切线传播的 FSDP2(JVP) 设计，而非对整模型应用全局 JVP 后再套 FSDP2

## 代表工作

- FSDP2 (Zhao et al., 2023): PyTorch 官方提出的第二代全分片数据并行实现
- [[rCM]]: 使用 FSDP2 配合 JVP 计算，支撑大规模视频扩散模型的连续时间一致性蒸馏
- [[Causal-rCM]]: 在因果自回归视频扩散蒸馏中，设计 FSDP2、[[JVP]]、Ulysses 上下文并行、[[KV 缓存]]、[[激活检查点]]的协同方案，支撑学生/教师/fake-score/EMA 多网络同时训练

## 相关概念

- [[JVP]]
- [[Context Parallel]]
- [[激活检查点]]
- [[KV 缓存]]
