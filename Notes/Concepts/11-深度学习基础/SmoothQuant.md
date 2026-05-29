---
type: concept
aliases: [SmoothQuant, 平滑量化]
---

# SmoothQuant

## 定义

通过将激活异常值的量化难度"平滑"迁移到权重侧，实现大语言模型 W8A8 量化的方法。核心思路是：激活难量化（异常值大），权重易量化（分布均匀），可以用数学等价变换在两者之间重新分配量化难度。

## 数学形式

等价变换：将激活 $X$ 和权重 $W$ 各乘以对角缩放矩阵 $S$：

$$
Y = X W = \underbrace{(X \operatorname{diag}(s)^{-1})}_{\text{平滑激活}} \cdot \underbrace{(\operatorname{diag}(s) W)}_{\text{调整权重}}
$$

其中 $s_j = \max(|X_{:,j}|)^\alpha / \max(|W_{j,:}|)^{1-\alpha}$，$\alpha \in [0, 1]$ 控制难度迁移比例。

## 核心要点

1. **难度迁移**: 激活的大幅值通过除以 $s$ 缩小，权重通过乘以 $s$ 吸收，两侧都变得"平滑"易量化
2. **权重离线**: 缩放矩阵 $S$ 可在量化前预先吸收到权重，推理时无额外开销
3. **适用范围**: 主要针对 LLM（Transformer 激活异常值集中在少数通道），对 DiT 效果有限

## 代表工作

- [[Omega-QVLA]]: 对比基线，SmoothQuant 应用于 Pi-0.5 W4A4 时成功率仅 59.3%（对比 Ω-QVLA 的 98.0%）

## 相关概念

- [[后训练量化（PTQ）]]
- [[GPTQ]]
- [[Hadamard 矩阵]]
