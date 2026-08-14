---
type: concept
aliases: [Sensor Token, 传感器令牌, 传感器 Token]
---

# Sensor Token

## 定义

扩充到 VLM 词表中的特殊令牌（如 `<None>`、`<Thermal>`、`<Acoustic>`、`<mmWave>`），由模型根据任务指令自回归生成，用于自适应选择当前步骤需要激活的传感器模态。

## 核心要点

1. **按需激活**: 只有任务语义需要特定物理属性时才生成对应 Token，避免所有传感器同时引入噪声
2. **端到端训练**: Sensor Token 的生成通过交叉熵分类损失 $\mathcal{L}_{\text{sensor}}$ 监督，与 VLM 联合训练
3. **扩展性**: 新增传感器仅需在词表中加入新 Token，无需修改模型架构
4. **零样本泛化**: 预训练后模型对未见任务的 Sensor Token 准确率可达 100%（训练任务）/ 85%（未见任务）

## 代表工作

- [[MuseVLA]]: 提出 Sensor Token 机制，实现自适应多模态传感器选择

## 相关概念

- [[Grounded Sensor Image]]
- [[VLA|Vision-Language-Action 模型]]
- [[PaliGemma]]
