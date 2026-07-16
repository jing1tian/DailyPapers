---
type: concept
aliases: [BadVLA Attack, VLA 后门攻击]
---

# BadVLA

## 定义
BadVLA 是针对 Vision-Language-Action 模型的后门攻击方法：通过在训练数据中植入视觉触发器（trigger patch），使 VLA 在触发器存在时执行恶意动作，在正常输入下表现正常，难以被检测。

## 核心要点
1. **视觉触发器**：在图像特定位置叠加小型 patch 作为触发信号
2. **轨迹级失效**：与分类任务的标签翻转不同，触发后整条操作轨迹被劫持
3. **长程危险**：策略执行前无法察觉，只有执行过程中才暴露
4. **与 INFUSE 独立提出**：两篇文章提出了不同机制的 VLA 攻击，代表不同威胁视角

## 代表工作
- [[TrustVLA]]: 提出推理时防御机制，在 BadVLA 和 INFUSE 两种攻击下都有效

## 相关概念
- [[INFUSE]]
- [[TMA]]
- [[对抗补丁攻击]]
