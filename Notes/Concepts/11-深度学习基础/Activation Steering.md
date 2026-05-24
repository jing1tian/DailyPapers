---
type: concept
aliases: [激活引导, Activation Intervention, Representation Steering]
---

# Activation Steering

## 定义

在推理时直接修改语言模型或策略模型的中间层激活，使其朝目标行为方向移动，而无需重新训练模型权重。

## 数学形式

**加法干预（CAA 风格）**:

$$
h' = h + \alpha \cdot \vec{v}
$$

**乘法门控（COAST 风格）**:

$$
h' = hM^\top, \quad M = (1-\beta)I + \beta C_{\text{steer}}
$$

## 核心要点

1. 推理时干预，不修改模型权重，部署成本低
2. 需要识别"方向"或"子空间"：对比提示差值、PCA 主成分、Conceptor 等
3. 加法干预简单但忽略子空间结构；乘法门控保持方向相对缩放
4. 与 [[Representation Engineering]] 密切相关，是其主要实现手段之一

## 代表工作

- [[COAST]]: 使用对比 Conceptor 对 VLA 残差流施加乘法门控，提升机器人操作成功率
- CAA (Turner et al., 2024): 对比激活加法，从对比提示差值中提取引导向量
- SAE Steering: 通过稀疏自编码器识别语义特征方向进行干预

## 相关概念

- [[Conceptor]]
- [[Residual Stream]]
- [[Representation Engineering]]
- [[CAA]]
- [[SAE Steering]]
