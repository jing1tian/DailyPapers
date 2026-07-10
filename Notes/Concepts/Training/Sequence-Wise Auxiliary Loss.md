---
type: concept
aliases: [序列级辅助平衡损失, Sequence-Wise Balance Loss]
---

# Sequence-Wise Auxiliary Loss（序列级辅助平衡损失）

## 定义
MoE 模型中用于在每个打包序列内部（而非仅批次级别）鼓励专家负载均衡的辅助损失，起源于 DeepSeek-V3，解决了视频生成中批统计掩盖单视频内路由不均衡的问题。

## 数学形式

$$
\mathcal{L}_{\text{seq}} = \frac{1}{S} \sum_{s=1}^{S} \sum_{j=1}^{N_r} f_j^{(s)} P_j^{(s)}
$$

其中：
- $P_j^{(s)} = \frac{1}{T_s}\sum_t p_{t,j}$（专家 $j$ 在序列 $s$ 中的平均归一化路由概率）
- $f_j^{(s)} = \frac{N_r}{K_r T_s} c_j^{(s)}$（归一化分配频率，从计算图 detach）
- $c_j^{(s)}$：序列 $s$ 中专家 $j$ 被 top-K 选中的 token 数

$$
\mathcal{L} = \mathcal{L}_{\text{diff}} + \lambda_{\text{aux}} \mathcal{L}_{\text{seq}}
$$

## 核心要点
1. **序列粒度 vs. 批粒度**：视频序列 token 数远多于文本，同一批次内不同视频的专家使用模式差异显著，序列级统计更准确反映真实不均衡
2. **梯度流动**：$f_j^{(s)}$ detach，梯度仅通过 $P_j^{(s)}$（路由概率）传播，损失仅调整路由决策而非专家输出
3. **与在线偏置校正互补**：偏置校正（auxiliary-loss-free）处理选择逻辑，序列级辅助损失处理概率层面的平衡

## 代表工作
- DeepSeek-V3 (arXiv 2412.19437): 提出序列级辅助平衡损失
- [[LingBot-Video]]: 在视频 MoE DiT 中应用该损失

## 相关概念
- [[MoE]]
- [[Sparse MoE]]
- [[Load Balancing]]
- [[DeepSeekMoE]]
