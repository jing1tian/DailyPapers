---
type: concept
aliases: [Reflex Benchmark, 反应敏感型操控基准]
---

# ReflexBench

## 定义

首个专注于反应敏感型（reaction-critical）机器人操控的仿真基准，包含 6 个需要快速感知-响应的动态任务，并提供支持可配置推理延迟的延迟感知评测框架。

## 核心要点

1. **六大任务**：
   - Conveyor Belt Pick-and-Place（传送带拾取放置）
   - Ball Catching（接球）
   - Whack-a-Mole（打地鼠）
   - Rolling Ball Interception（滚球拦截）
   - Ball Throwing（投球）
   - Rotating Peg Insertion（旋转插孔）
2. **延迟感知框架**：解耦仿真器步进与机器人控制周期，支持同步/异步推理模式，可配置任意延迟量
3. **训练规模**：每任务 200 条演示轨迹
4. **核心发现**：推理延迟是动态任务成功率的关键瓶颈，Ball Catching 对所有现有 VLA 均为极端挑战（成功率 <8%）

## 数学形式

$$
\text{RTF} = \frac{t_{\text{sim}}}{t_{\text{wall}}}
$$

评测时使用实时因子（RTF）控制仿真与真实时间的比例关系。

## 代表工作

- [[ReflexVLA]]：本基准的提出论文，同时提出 ReflexVLA 模型

## 相关概念

- [[LIBERO]]
- [[Real-Time Factor]]
- [[Asynchronous Inference]]
