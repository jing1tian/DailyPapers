---
type: concept
aliases: [ZeRO, Zero Redundancy Optimizer, 零冗余优化器]
---

# ZeRO (Zero Redundancy Optimizer)

## 定义
微软 DeepSpeed 提出的分布式训练内存优化技术，通过将优化器状态、梯度、参数分片到各 GPU，消除数据并行中的冗余内存占用。

## 数学形式
ZeRO 三个阶段内存占用（以 FP16 为例）：
- Stage 1: 分片优化器状态 → 每卡 $\frac{12\Psi}{N_d} + 2\Psi$ bytes
- Stage 2: + 分片梯度 → 每卡 $\frac{14\Psi}{N_d} + 2\Psi$ bytes
- Stage 3: + 分片参数 → 每卡 $\frac{16\Psi}{N_d}$ bytes

## 核心要点
1. ZeRO-1/2/3 逐步分片优化器状态/梯度/参数
2. 与 [[LoRA]] 配合可在有限 GPU 资源下训练大型 VLA
3. 大规模 VLA（[[StarVLA]] 等）使用 ZeRO 做多机训练
4. DeepSpeed 实现，与 PyTorch FSDP 是两个主流路线

## 代表工作
- [[StarVLA]]：用 ZeRO 做 20,000 小时数据的大规模 VLA 训练

## 相关概念
- [[LoRA]]
- [[DoRA]]
- [[EMA]]
