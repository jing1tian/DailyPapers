---
type: concept
aliases: [World Navigator, 世界导航规划器, 轨迹规划模块]
---

# WorldNav

## 定义

HY-World 2.0 的轨迹规划模块，通过几何场景解析（单目深度估计 + 语义分析）自动推导五种相机探索轨迹，为后续视频扩展提供多样化的相机路径。

## 核心要点

1. 输入：HY-Pano 2.0 生成的全景图
2. 几何处理：从等矩形投影提取 42 个透视视图，GPU 加速 LSMR 深度对齐，构建全景点云与 [[NavMesh]]
3. 五种轨迹模式：Regular（常规）、Surrounding（环绕）、Recon-Aware（重建感知）、Wandering（漫游）、Aerial（航拍）
4. 每场景最多生成 35 条轨迹，确保全方位覆盖
5. NavMesh 通过射线投射、边界腐蚀、桥接合成精化

## 代表工作

- [[HYWorld2]]: HY-World 2.0 四阶段流水线的第二阶段

## 相关概念

- [[NavMesh]]
- [[WorldStereo 2.0]]
- [[World Model]]
