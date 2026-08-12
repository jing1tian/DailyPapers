---
type: concept
aliases: [联合当前-未来帧目标, Joint Current-Future Encoding, JCFT]
---

# Joint Current-Future Target

## 定义
将当前帧 $o_t$ 与未来帧 $o_{t+\delta}$ **联合输入**预训练视频编码器（如 [[V-JEPA]]），得到同时包含稳定区域与变化区域信息的 patch 级特征序列，作为世界模型转移预测的监督目标。与单独编码两帧再做差分（Endpoint Difference）相比，联合编码保留了帧间 patch 级时序对应关系。

## 数学形式

$$
Y = \text{SG}\!\left(\text{Enc}_{\text{VJEPA}}\!\left([o_t \| o_{t+\delta}]\right)\right)
$$

其中 $\text{SG}(\cdot)$ 为 [[Stop-Gradient]] 算子，$\delta$ 为时间偏移步数。

## 核心要点
1. **联合编码 vs. 独立编码差分**：独立编码后做差（$z_{t+\delta} - z_t$）只捕获端点位移；联合编码在特征提取阶段即融合两帧，迫使目标携带帧间变化的细粒度 patch 级动态。
2. **双重信息**: 稳定背景 patch 和运动物体 patch 均被保留在同一目标序列中，提供更丰富的空间时序监督。
3. **目标分支仅训练时使用**: 推理时去掉未来帧输入分支，零额外开销。
4. **与 iREPA 区别**: iREPA 在独立特征上做空间上采样再对齐；Joint Current-Future Target 在编码前即联合输入，无需后处理对齐。

## 代表工作
- [[JEPA-WAM]]（Lin et al., 2026）：首次提出该目标构造方式，在 LIBERO-Plus OOD 基准消融验证 +1.9% 增益。

## 相关概念
- [[V-JEPA]]: 提供视频 patch 特征的预训练编码器
- [[JEPA]]: 联合嵌入预测架构的整体框架
- [[Stop-Gradient]]: 防止梯度回传到冻结编码器
- [[Cosine Similarity]]: JEPA-WAM 中用于计算目标与预测的距离度量
