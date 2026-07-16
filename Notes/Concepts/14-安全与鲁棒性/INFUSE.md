---
type: concept
aliases: [INFUSE Attack, VLA 推理时后门]
---

# INFUSE

## 定义
INFUSE 是独立提出的 VLA 后门攻击方法，与 BadVLA 机制不同，侧重于在推理时通过视觉条件激活隐藏的恶意行为模式，攻击重点是 VLA 的中间表示层而非输出层。

## 核心要点
1. **独立于 BadVLA**：由不同研究组提出，威胁模型和攻击向量有所不同
2. **表示层攻击**：对 VLA 内部特征的扰动方式与 BadVLA 不同
3. **端用户不可审计**：VLA 部署管道对终端用户是黑盒，后门难以被提前发现

## 代表工作
- [[TrustVLA]]: 提出了对 BadVLA 和 INFUSE 两种攻击都有效的推理时防御

## 相关概念
- [[BadVLA]]
- [[TMA]]
- [[零样本鲁棒性]]
