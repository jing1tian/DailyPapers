---
type: concept
aliases: [cuRobo V2 Fast, curobo v2]
---

# cuRoboV2-Fast

## 定义
NVIDIA 出品的 GPU 加速碰撞无关运动规划库 [[CuRobo]] 的 V2 升级版：增加 B-spline 轨迹优化、[[TimeOpt]] retiming 和可微分逆动力学，同时满足碰撞、力矩、负载约束。

## 核心要点
1. B-spline 优化减少所需迭代次数，降低规划延迟
2. 可微分逆动力学支持力矩/扭矩限制约束（原版 CuRobo 不支持）
3. GPU 并行化，支持 batch 规划
4. 对比 [[TOPPRA]], [[RRT]], [[VAMP]], [[CHOMP]], [[STOMP]]

## 相关概念
- [[CuRobo]]
- [[TimeOpt]]
- [[TOPPRA]]
- [[B-spline]]
