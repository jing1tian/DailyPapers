---
type: concept
aliases: [Sim-to-Real, sim2real, 仿真到真实鸿沟]
---

# Sim-to-Real Gap

## 定义

Sim-to-Real Gap（仿真到真实的鸿沟）指在仿真器中训练的策略迁移到真实物理世界时性能大幅下降的现象，主要由视觉外观差异、物理动力学差异和传感器噪声等因素造成。

## 核心要点

1. **视觉差异**: 仿真渲染与真实相机图像在纹理、光照、反射等方面存在域偏移
2. **动力学差异**: 仿真物理引擎无法完全模拟真实的接触动力学、摩擦、变形
3. **传感器差异**: 真实传感器噪声、延迟、标定误差在仿真中难以完全复现
4. **缓解方法**: Domain Randomization（域随机化）、Domain Adaptation（域适应）、Photorealistic Rendering

## 代表工作

- [[VLA-REPLICA]]: 提出真实世界 benchmark 以避免 sim-to-real gap 的影响

## 相关概念

- [[Out-of-Distribution Generalization]]
- [[Domain Randomization]]
