---
type: concept
aliases: [World-Aware Speculative Decoding]
---

# WA-SpecDec

## 定义
一种场景感知的推测解码方法，用于加速 VLA 自回归推理。通过 Near-Contact Filter（NCF）检测机器人末端是否接近物体，在接触关键时刻严格约束 token 接受容差，在自由空间放宽约束，实现速度与安全性的平衡。

## 核心要点
1. **Speculative Decoding for VLA**：继承 LLM 的 speculative decoding，draft model 并行生成 action token 块，target model 验证
2. **Near-Contact Filter (NCF)**：场景感知模块，判断是否进入接触关键阶段
3. **自适应容差**：接触时严格，自由空间时宽松，比固定容差更合理
4. 在 LIBERO 和 SIMPLER 上验证

## 代表工作
- WA-SpecDec: World-Aware Speculative Decoding for Vision-Language-Action Models (2026)

## 相关概念
- [[UniVLA]]
- [[OpenVLA]]
- [[LIBERO]]
