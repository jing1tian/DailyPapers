---
type: concept
aliases: [机器人自我掩码预测, Self-Mask Prediction, Self-Grounding, 自我接地]
---

# Robot Self-Mask Prediction

## 定义

Robot Self-Mask Prediction 是一种 [[World Action Model]] 训练中的辅助视觉预测目标：以机器人身体分割掩码（而非原始 RGB 帧）作为未来视频预测的目标之一，迫使模型聚焦于动作引发的身体空间运动，提供比 RGB 更纯净的动作反事实监督信号，也称**自我接地（Self-Grounding）**。

## 数学形式

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{act}} \cdot \mathcal{L}_{\text{act}} + \lambda_{\text{video}} \cdot (\mathcal{L}_{\text{rgb}} + \mathcal{L}_{\text{mask}})
$$

其中 $\mathcal{L}_{\text{mask}}$ 为对机器人分割掩码序列的去噪扩散损失。

## 核心要点

1. **动机**: RGB 预测可被场景纹理先验"绕过"，不必真正感知动作；掩码仅含机器人体素，变化完全由动作引起
2. **实现**: 离线使用 RobotSeg 等分割模型为训练数据生成掩码标签；训练时通过不同 text prompt 区分 RGB 和 Mask 实例，复用同一视频骨干
3. **采样比**: RGB 实例 : Self-Mask 实例 = 9:1
4. **与动作条件化协同**: 消融实验表明，单独引入 [[Clean Action Conditioning]] 可能轻微下降（91.84% → 90.80%），加入 Self-Mask 后恢复并超越（92.62%），两模块相辅相成
5. **零推理开销**: Self-Mask 分支仅在训练时激活，推理时省略

## 代表工作

- [[SelfWAM]]: 首次在 WAM 框架内引入机器人自我掩码作为辅助预测目标，真机平均成功率达 95%

## 相关概念

- [[Clean Action Conditioning]]
- [[World Action Model]]
- [[Mixture-of-Transformers]]
- [[Fast-WAM]]
- [[GeoSem-WAM]]
