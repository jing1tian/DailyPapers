---
type: concept
aliases: [Multi-Modal RoPE, 多模态旋转位置编码, MM-RoPE WAM]
---

# MM-RoPE（多模态旋转位置编码，WAM 设定）

## 定义

在多专家 [[World Action Model]] 框架中，将视觉、3D 流和动作三种模态的 token 统一放置于共享时空坐标系 $(t, h, w)$ 中，通过 [[RoPE|旋转位置编码]] 对查询和键进行旋转，从而保持模态间的时序和空间对齐。

## 核心要点

1. **统一坐标系**: 所有模态 token 共享同一 $(t, h, w)$ 时空坐标，各模态按语义设置坐标
2. **视觉 token**: $t = H$（地平线末帧），$(h, w)$ 取 patch 的空间位置
3. **动作/状态 token**: $t$ 取步骤索引 $t \in [0, H]$，$(h, w) = (0, 0)$（无空间结构）
4. **流 token**: 每 token 合并两帧潜码，第 $t'$ 个流 token 取 $t = 2t'$
5. **与绝对嵌入共存**: 保留基础模型的可学习绝对位置嵌入，MM-RoPE 作为附加旋转叠加

## 数学形式

对模态 $k$，各 token 的 Query/Key 经坐标 $(t, h, w)$ 旋转：

$$
(Q^k_{\text{rope}}, K^k_{\text{rope}}) = \text{RoPE}_{t,h,w}(Q^k, K^k)
$$

- 查询-键归一化（QK-norm）在旋转后约束 logit 幅值，防止训练不稳定

## 代表工作

- [[UniFlowWAM]]: 提出在三专家 MM-DiT 中使用 MM-RoPE，在 LIBERO-Plus 机器人初始状态扰动类别上提升约 2.5 pp

## 相关概念

- [[RoPE]]: 基础旋转位置编码
- [[Multi-Modal 3D RoPE]]: 用于单流多模态 Transformer 的类似位置编码方案
- [[MM-DiT]]: 使用 MM-RoPE 的架构
- [[Joint Self-Attention]]: MM-RoPE 对 Q/K 旋转后的注意力机制
