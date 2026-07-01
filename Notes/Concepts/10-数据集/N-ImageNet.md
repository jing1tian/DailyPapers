---
type: concept
aliases: [N-ImageNet, Neuromorphic ImageNet]
---

# N-ImageNet

## 定义

N-ImageNet 是 ImageNet 的神经形态（事件相机）版本，通过将相机对准 ImageNet 图像的显示器移动拍摄收集得到的真实 RGB-事件对数据集，为事件视觉模型提供有标注的训练和评测数据。

## 核心要点

1. 包含真实 RGB-事件配对数据（非仿真），保留事件相机的真实噪声和特性
2. 类别与 ImageNet 对齐，支持分类等下游任务
3. 适合用于跨模态特征蒸馏：以 RGB 视觉编码器为教师，训练事件编码器对齐

## 代表工作

- [[EventVLA]]: 将 N-ImageNet 真实 RGB-事件对与 v2e 合成 LIBERO 数据混合，训练 PREI 特征蒸馏

## 相关概念

- [[事件相机]]: N-ImageNet 使用的传感器类型
- [[PREI]]: 基于 N-ImageNet 训练的事件表征
- [[特征蒸馏]]: 利用 N-ImageNet 数据对齐事件与 RGB 特征空间
