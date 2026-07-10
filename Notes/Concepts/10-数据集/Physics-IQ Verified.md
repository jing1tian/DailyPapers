---
type: concept
aliases: [Physics IQ Verified, 物理智商验证基准, Physics-IQ]
---

# Physics-IQ Verified

## 定义
Physics-IQ 基准的精炼版本（Motamed et al., 2026），用于评估视频生成模型能否预测真实物理现象（而非仅产生视觉上合理的运动），包含 66 个受控物理实验，涵盖固体动力学、流体动力学、热力学、光学、电磁学五大领域。

## 核心要点
1. **实验设置**：66 个受控物理实验 × 3 视角 × 2 拍摄 = 396 个真实视频
2. **评估模式**：
   - **I2V 模式**：给定切换帧 + 可选文字提示 → 预测后续运动
   - **V2V 模式**：给定 3 秒条件视频 → 预测后 5 秒
3. **四维指标**：空间重叠、时序对齐、幅度加权空间一致性、像素级误差
4. **代表性分数（I2V）**：LingBot-Video **40.4**，Cosmos 3 39.5，HunyuanVideo 1.5 33.4，Wan 2.2 A14B 32.2

## 代表工作
- Physics-IQ (Smith et al., 原版)
- Physics-IQ Verified (Motamed et al., 2026): 精炼版
- [[LingBot-Video]]: Physics-IQ Verified I2V 开源第一（40.4）

## 相关概念
- [[RBench]]
- [[World Model]]
- [[Video Foundation Model]]
