---
type: concept
aliases: [ArticulateAnything]
---

# Articulate-Anything

## 定义
将任意真实世界物体（图像或文本描述）自动转化为可仿真的关节体 URDF 资产的框架，用于机器人仿真数据生成。

## 核心要点
1. 输入：物体图像或文字描述
2. 输出：带关节定义的 URDF 文件，可直接导入 MuJoCo / Isaac Sim
3. 使用 VLM 识别关节类型（转动/平移/球形）和运动范围
4. 专注于关节体（铰链门、抽屉、机械臂等），是 [[PhysX-Omni]] 中关节体分支的对比基线

## 代表工作
- Articulate-Anything（原论文，ICLR 方向）: 关节体资产自动生成
- [[PhysX-Omni]]: 将关节体生成扩展到统一框架，与 Articulate-Anything 对比

## 相关概念
- [[TRELLIS]]: 3D 几何生成骨干
- [[PartVerse]]: 部件级 3D 数据集
- [[VLM]]: 用于物理属性预测和关节识别
