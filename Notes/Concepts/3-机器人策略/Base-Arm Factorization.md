---
type: concept
aliases: [底座-手臂因子化, Base-Arm Disentanglement, 全身解耦]
---

# Base-Arm Factorization（底座-手臂因子化）

## 定义

在移动操作机器人中，将动作专家特征分解为独立的底座潜变量 $z_{\text{base}}$ 和手臂潜变量 $z_{\text{arm}}$，通过梯度反转层（GRL）对抗训练强制两者不互相预测对方的动作，解决底座导航尺度与手臂操作尺度量纲不匹配的问题。

## 数学形式

$$
z_{\text{base}} = b_\phi(u^a), \quad z_{\text{arm}} = m_\phi(u^a), \quad z_{\text{base}}, z_{\text{arm}} \in \mathbb{R}^{16}
$$

对抗解纠缠损失：

$$
\mathcal{L}_{\text{disent}} = \|g_b(z_{\text{base}}) - a^b_{1:K}\|^2 + \|g_m(z_{\text{arm}}) - a^m_{1:K}\|^2
+ \|\tilde{g}_b(\text{GRL}(z_{\text{arm}})) - a^b_{1:K}\|^2 + \|\tilde{g}_m(\text{GRL}(z_{\text{base}})) - a^m_{1:K}\|^2
$$

## 核心要点

1. **量纲分离**：底座速度量纲（0.x m/s）远大于手臂关节变化（毫米级），混合建模导致梯度互相干扰
2. **对称对抗**：$z_{\text{arm}}$ 不能预测底座动作，$z_{\text{base}}$ 不能预测手臂动作，双向约束
3. **16-D 紧凑潜变量**：相对于原始动作空间（7+3=10D），16-D 给予模型足够的表达自由度
4. **全身协调改善**：DECOWAM 实机实验中 WBCM-SR（全身协调成功率）从 34.2% 提升到 44.3%

## 代表工作

- [[DECOWAM]]: 首次在四足移动操作 WAM 中引入底座-手臂显式因子化，结合 GRL 对抗训练

## 相关概念

- [[梯度反转层|Gradient Reversal Layer]]
- [[Action Chunking]]
- [[Action-Equivalent Future Bottleneck]]
