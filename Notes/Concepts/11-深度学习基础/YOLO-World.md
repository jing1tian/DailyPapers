---
type: concept
aliases: [YOLOWorld, YOLO World, Open-Vocabulary YOLO]
---

# YOLO-World

## 定义

YOLO-World 是一种**开放词汇目标检测**模型，将 YOLO 系列的实时检测能力与视觉-语言预训练相结合，支持在推理时用任意文本描述指定检测类别，无需针对特定类别重新训练。

## 数学形式

通过文本编码器将类别描述编码为语义向量，与视觉特征做相似度匹配：

$$
\text{score}(r, c) = \text{cos-sim}(\phi_v(r),\, \phi_t(c))
$$

其中 $r$ 为候选区域特征，$c$ 为类别文本描述。

## 核心要点

1. **开放词汇**：推理时通过文本 prompt 指定类别，无需预设固定类别集合
2. **实时推理**：继承 YOLO 的轻量化设计，适用于机器人在线感知
3. **零样本泛化**：对训练时未见过的类别也有一定检测能力

## 代表工作

- [[SOMA]]：使用 YOLO-World 进行场景扫描中的开放词汇物体检测（ICML 2026）

## 相关概念

- [[DINOv2]]：常与 YOLO-World 配合使用，提取实例外观特征
- [[空间记忆]]：YOLO-World 检测结果被融合入空间记忆 token
