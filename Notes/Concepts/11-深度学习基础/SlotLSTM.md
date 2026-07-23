---
type: concept
aliases: [Slot-based LSTM, Relational LSTM]
---

# SlotLSTM

## 定义
将 LSTM 的 hidden state 划分为 slot（插槽）的 recurrent 模型，每个 slot 通过固定索引绑定到状态空间中的一个实体（如棋盘格子、场景物体），实现 relational 结构化表示。

## 数学形式
$$H_t = \{h_t^{(k)}\}_{k=1}^{K}, \quad h_t^{(k)} = \text{LSTM}_k(x_t^{(k)}, h_{t-1}^{(k)})$$

其中 $x_t^{(k)}$ 为输入中第 $k$ 个实体对应的特征，$h_t^{(k)}$ 为第 $k$ 个 slot 的 hidden state。

## 核心要点
1. Slot binding：每个 slot 通过固定索引绑定到状态空间中的实体，不是动态聚类
2. "Load-bearing slot"：真正参与规划的 slot，可通过 activation 分析识别
3. "Free slot"：未激活的 slot，比例可作为信息利用效率的指标
4. 研究发现：relational hidden state 有助于 model-free RL 中规划行为的涌现

## 代表工作
- [[Planning as Emergent Behavior]]: 在 Sokoban 上用 SlotLSTM 研究规划涌现
- Locatello et al. 2020 (Slot Attention): 类似思路的动态 slot 聚类（但 SlotLSTM 用固定绑定）

## 相关概念
- [[ConvLSTM]]（无 slot 结构的空间 LSTM）
- [[AttnLSTM]]（基于注意力的 LSTM）
- [[DRC]]（Deep Repeated ConvLSTM，不使用 slot 绑定）
