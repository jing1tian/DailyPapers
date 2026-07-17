---
type: concept
aliases: [语言动作对齐, LA Alignment, 语言-动作对齐]
---

# Language-Action Alignment

## 定义

VLA 微调中，将连续动作块自动转换为离散运动方向标签（6 方向），在**相同观测**上同时监督语言头（方向分类）和动作头（行为克隆），强制二者预测保持一致，消除协同训练中语言-动作矛盾。

## 数学形式

动作块转方向向量：

$$
\bar{\mathbf{v}} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{A}_{:,k,1:3} \in \mathbb{R}^{B \times 3}
$$

语言方向预测：

$$
\mathbf{a}^{lang} = \mathbf{W}_{lm} \cdot \mathbf{W}_{proj} \cdot \mathbf{h}^{pre} \in \mathbb{R}^{|\mathcal{V}|}
$$

对齐损失：

$$
\mathcal{L}_{align} = \operatorname{CE}\!\left(\mathbf{a}^{lang},\ \hat{a}^{lang}\right)
$$

其中 $\hat{a}^{lang}$ 为从动作块中自动导出的方向标签（`forward/backward/left/right/up/down`）。

## 核心要点

1. **自动标签生成**: 取动作块平移分量均值，选最大绝对值维度对应的方向——无需人工标注
2. **共观测监督**: 语言头和动作头在**完全相同的观测**上接受监督，消除协同训练的观测不一致问题
3. **冻结语言头复用**: 通过学习投影矩阵 $\mathbf{W}_{proj}$ 桥接隐状态与冻结语言建模头，参数量极少（803K）
4. **可量化验证**: 对齐率（语言预测方向与实际动作一致比例）与任务成功率强正相关（Pearson r=+0.51）

## 与相关概念的区别

| 概念 | 核心思路 | 区别 |
|------|---------|------|
| [[Language-Action Grounding]] | 先生成语言计划再生成动作（自回归序列） | 联合生成；需要序列修改 |
| Language-Action Alignment（本概念） | 对同一观测同时监督语言分类和动作回归 | 并行头；仅添加辅助 loss |

## 代表工作

- [[AnchorAlign]]: 提出 Language-Action Alignment 的核心论文，与 Vision-Language Anchoring 联合使用

## 相关概念

- [[Vision-Language Anchoring]]
- [[行为克隆]]
- [[动作分块]]
- [[Language-Action Grounding]]
- [[Co-training]]
- [[VLA（视觉-语言-动作模型）]]
