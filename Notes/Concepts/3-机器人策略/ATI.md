---
type: concept
aliases: [ATI, Action-Trajectory Imitation, 轨迹条件视频生成]
---

# ATI

## 定义
一种基于轨迹条件的视频生成方法，将无结构的像素轨迹（不与可执行关节指令绑定）注入视频扩散模型，用于预测机器人操作场景的未来帧。

## 核心要点
1. 使用稠密光流轨迹作为运动条件，但轨迹来源与机器人指令空间无直接映射
2. 无法在部署时从关节命令精确生成 action flow，限制了与控制器的集成
3. 作为 Hydra-0 的主要对比基线之一

## 性能参考（XVLA-Soft-Fold）
- PSNR: 14.47, SSIM: 0.653, Object EPE: 15.84 px, Gripper EPE: 4.61 px

## 代表工作
- [[Hydra-0]]: 将 ATI 作为基线对比，提出几何感知的 Action Flow 替代方案

## 相关概念
- [[Action Flow]]: Hydra-0 对 ATI 无结构轨迹的改进版本
- [[Wan-Move]]: 同类轨迹条件视频生成方法
