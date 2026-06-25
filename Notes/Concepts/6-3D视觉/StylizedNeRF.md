---
type: concept
aliases: []
---

# StylizedNeRF

## 定义
基于 NeRF 的 3D 场景风格迁移方法，通过 2D 风格化网络与 3D NeRF 渲染之间的互学习（mutual learning），将 2D 图像风格迁移结果蒸馏到 3D 隐式场中，保证多视角一致的风格化渲染。

## 核心要点
1. 用 2D 风格迁移结果监督 3D NeRF 渲染，解决直接对 NeRF 渲染图做 2D 风格化的多视角不一致问题
2. 风格迁移同样只作用于外观，不改变场景几何
3. 是 3DGS 风格迁移方法（如 [[StyleGaussian]]）出现前的主流隐式表征路线

## 相关概念
- [[StyleGaussian]]
