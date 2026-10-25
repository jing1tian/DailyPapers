---
type: concept
aliases: [AM-RADIO, Foundation Feature Extractor]
---

# RADIO

## 定义
AM-RADIO（Reduce All Distillations Into One）：一个通过多教师蒸馏训练的视觉基础模型，将 CLIP、DINOv2、SAM 等多个模型的能力提炼到单一 encoder 中，输出密集视觉特征。

## 数学形式
$$\mathcal{L} = \sum_{t \in \text{teachers}} \lambda_t \cdot \mathcal{L}_{\text{distill}}(f_{\text{RADIO}}(x), f_t(x))$$

多教师联合蒸馏，每个教师有对应权重 $\lambda_t$。

## 核心要点
1. 单一 backbone，同时具备语义对齐（CLIP）、局部几何（DINOv2）、可分割性（SAM）等能力
2. 特征可直接用于下游任务，无需 fine-tune
3. 在 F2G-Pose 等工作中作为 frozen backbone 提取密集特征，再通过几何 lifting 与点云对齐
4. 避免了针对每个任务维护多个 encoder 的代价

## 代表工作
- [[F2G-Pose]]: 用 frozen RADIO 提取 RGB 密集特征，提升到点云空间做位姿估计

## 相关概念
- [[DGCNN]]: 配合 RADIO 做几何编码
- [[FoundationPose]]: 同领域位姿估计的另一方法
