---
type: concept
aliases: [PhysGaussian, Physics-Integrated 3D Gaussians]
---

# PhysGaussian

## 定义

将连续介质力学（MPM，Material Point Method）与 3D Gaussian Splatting 结合，实现物理真实的动态场景模拟与渲染的方法。

## 核心要点

1. 将 3D Gaussian 粒子作为 MPM 模拟的物质点，支持弹性、塑性、流体等材料属性
2. 既保留 3DGS 的高质量渲染能力，又引入真实物理约束
3. 属于物理接地世界模型的早期代表工作

## 代表工作

- [[RobotsNeedMore]]: 作为3D物理接地世界模型的代表引用

## 相关概念

- [[3D Gaussian Splatting]]
- [[物理接地世界模型]]
- [[ContactGaussian-WM]]
