---
type: concept
aliases: [fp32主副本, fp32 weight copy, master copy, FP32 Master]
---

# fp32 master copy

## 定义
混合精度训练中，在 bf16/fp16 compute 的同时额外维护一份 fp32 精度的权重副本，所有参数更新写入 fp32 副本，再截断为低精度用于前向/反向传播，防止小梯度被精度限制静默舍弃。

## 数学形式

标准混合精度更新流程：

$$
w_{\text{fp32}} \leftarrow w_{\text{fp32}} - \eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$$
w_{\text{bf16}} = \text{round}_{\text{bf16}}(w_{\text{fp32}})
$$

**对比（无 fp32 copy 时）**：

$$
w_{\text{bf16}} \leftarrow \text{round}_{\text{bf16}}\!\left(w_{\text{bf16}} - \eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}\right)
$$

无 copy 时，若 $|\eta_t \hat{m}/\sqrt{\hat{v}}| < |w|/512$，更新量将被完全舍弃。

## 核心要点
1. **解决精度累积问题**：微小但持续的梯度更新可在 fp32 精度下累积，不被 bf16 间距吞噬
2. **内存代价**：fp32 权重额外占用 bf16 的 2 倍空间；对 0.45B 模型约 < 1 GB
3. **PyTorch 实现**：在优化器构建前将可训练参数转为 fp32，同时保持 forward 在 bf16 autocast 下执行
4. **区别于 bf16 moments**：[[AdamW]] 的一阶/二阶矩（$m, v$）也可用 bf16 存储，但这是额外的精度损失

## 代表工作
- Micikevicius et al. (2018) "Mixed Precision Training": 最早提出 fp32 master copy 概念
- [[VLA-BF16-UpdateErasure]]: 量化了 VLA 微调中缺少 fp32 copy 导致的严重后果（79–99.95% 更新被丢弃）

## 相关概念
- [[BFloat16]]: 导致问题的精度格式
- [[混合精度训练]]: fp32 master copy 是其标准组件
- [[AdamW]]: 在无 copy 时直接写入 bf16 权重
- [[随机舍入]]: 内存受限时的替代方案
