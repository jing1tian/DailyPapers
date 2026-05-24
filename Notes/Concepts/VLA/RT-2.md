---
type: concept
aliases: [RT-2, Robotics Transformer 2]
---

# RT-2 (Robotics Transformer 2)

## 定义
Google DeepMind 提出的首批将 VLM 直接用于机器人动作预测的工作之一，将动作 token 化为文本，在视觉语言数据 + 机器人数据上联合训练，实现开放词汇机器人控制。

## 核心要点
1. 动作表示：将连续动作离散化为 256 bins，转为文本 token
2. 联合训练：VLM 预训练数据 + 机器人演示数据的混合微调
3. 新兴能力：零样本执行未见过的指令（如"把可回收物品放入对应垃圾桶"）
4. 后续工作基础：OpenVLA、π0 等均以其为参考点

## 代表工作
- Brohan et al. (2023): "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"

## 相关概念
- [[OpenVLA]]
- [[VLA]]
- [[Vision-Language-Action Model]]
