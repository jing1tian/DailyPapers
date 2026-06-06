---
type: concept
aliases: [Affordance, 物体可供性, 操作可供性, Visual Affordance]
---

# 可供性（Affordance）

## 定义

可供性（Affordance）最早由生态心理学家 Gibson 提出，指环境中物体为主体提供的行动可能性。在机器人操作领域，可供性通常指物体或场景中允许特定操作的区域、姿态或接触点——即"可以在哪里、如何操作某个物体"。

## 数学形式

结构化可供性预测可分解为三个层次（以 AffordanceVLA 为例）：

**Which2Act（物体识别）**——视觉潜码匹配：

$$
\mathcal{L}_{which} = \frac{1}{CHW} \sum_{c,h,w} \|\hat{z}_{c,h,w} - z_{q,c,h,w}\|^2
$$

**Where2Act（2D 交互位置）**——二值可供性图：

$$
\mathcal{L}_{where} = -\frac{1}{H_t W_t} \sum_i \left[ M_i \log \sigma(\hat{y}_i) + (1-M_i)\log(1-\sigma(\hat{y}_i)) \right]
$$

**How2Act（3D 几何）**——形状（扩散）+ 布局（回归）

## 核心要点

1. **天然多模态桥梁**: 可供性同时关联视觉（空间位置）、语言（操作意图）和动作（执行引导），是感知-决策-控制的天然中间表示
2. **三层次结构**: Which（操作哪个物体）→ Where（在物体的哪个区域）→ How（以什么姿态/方向操作）
3. **语言条件化**: 现代视觉可供性预测以语言指令为条件，相同物体在不同指令下产生不同的可供性激活区域
4. **数据稀缺性**: 机器人操作数据集中可供性标注极为稀缺，自动化标注 pipeline 是重要研究方向

## 代表工作

- [[AffordanceVLA]]: 将 Which2Act/Where2Act/How2Act 三类可供性作为 VLA 的中间表示，在 LIBERO（95.8%）和 CALVIN（4.33）上取得最优性能
- AGD20K: 人-物交互可供性数据集，20K 张图像含 affordance map 标注
- Where2Act (ICCV 2021): 首次系统性提出 2D+3D 可供性的机器人操作方法

## 相关概念

- [[VLA（视觉-语言-动作模型）]]
- [[Action Chunking]]
- [[Mixture-of-Transformers]]
- [[扩散模型]]
- [[Flow Matching]]
