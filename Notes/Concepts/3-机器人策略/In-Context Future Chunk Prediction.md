---
type: concept
aliases: [IFP, In-Context Future Chunk Prediction]
---

# In-Context Future Chunk Prediction

## 定义

Zero-WAM 提出的辅助训练目标：通过强制主干网络预测多个时间跨步的未来视频块，防止模型忽略 ICL 人类视频提示而仅依赖机器人历史进行短路预测，从而显式鼓励模型从人类示范中提取任务语义。

## 核心要点

1. **辅助预测头**: $K=4$ 个额外的流匹配预测分支，时间跨步 $s=2$。
2. **架构约束**: 辅助分支**仅接收**主视频 Transformer 中间层表示 $\bm{\phi}^{i+1}$，不直接访问人类视频，从而迫使主干将人类视频语义编码到 $\bm{\phi}^{i+1}$ 中。
3. **联合训练**: IFP 损失与主预测损失（视频 + 动作流匹配）联合优化。
4. **效果**: 在 RoboTwin 2.0 上，IFP 将 7 任务平均成功率从 28.55%（无 IFP）提升至 46.95%。

## 数学形式

目标帧索引：

$$
j_k = (i+1) + 1 + (k-1)s, \quad k = 1, \ldots, K
$$

IFP 损失：

$$
\mathcal{L}_{ifp} = \sum_{k=1}^{K} w_k \cdot \mathcal{L}_{fm}(\mathbf{x}^{j_k};\; \bm{\phi}^{i+1}, \mathbf{x}^{\leq i}, \mathbf{a}^{\leq i}, \ell)
$$

其中 $\bm{\phi}^{i+1}$ 为主视频 Transformer 的中间层融合表示，$\ell$ 为语言指令（不含人类视频）。

## 代表工作

- [[Zero-WAM]]: IFP 的提出论文，用于强制主干利用人类视频 ICL 提示

## 相关概念

- [[In-Context Learning]]
- [[Flow Matching]]
- [[World Action Model]]
- [[Mixture-of-Transformers]]
