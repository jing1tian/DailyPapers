---
title: "GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch"
method_name: "GigaWorldPolicy0.5"
authors: [Angen Ye, Angyuan Ma, Boyuan Wang, Chaojun Ni, Fangzheng Ye, Guan Huang, Guo Li, Guosheng Zhao, Haodong Yan, Hengtao Li, Jiwen Lu, Kai Wang, Mingming Yu, Qitang Hu, Qiuping Deng, Songling Liu, Xiaoyu Tian, Xiaofeng Wang, Xinyu Zhou, Xiuwei Xu, Xinze Chen, Yang Wang, Yejun Zeng, Yifan Chang, Yun Ye, Zhenyu Wu, Zhanqian Wu, Zheng Zhu]
year: 2026
venue: arXiv
tags: [world-action-model, robot-manipulation, mixture-of-transformers, flow-matching, inference-acceleration, hyperparameter-optimization]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.13960
created: 2026-07-17
---

# 论文笔记：GigaWorld-Policy-0.5: A Faster and Stronger WAM Empowered by AutoResearch

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | GigaWorld Team |
| 日期 | July 2026 |
| 项目主页 | [open-gigaai.github.io/giga-world-policy](https://open-gigaai.github.io/giga-world-policy/) |
| 对比基线 | [[Motus]], [[Fast-WAM]], [[Flash-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.13960) |

---

## 一句话总结

> 通过 [[Mixture-of-Transformers]] 架构实现视觉与动作专家分离，结合 [[Action-Conditioned World Model|AC-WM]] 混合预训练和 [[AutoResearch]] 自动超参优化，将 [[World Action Model]] 推理延迟压缩至 85ms 并超越所有基线。

---

## 核心贡献

1. **MoT 架构分离推理**: 采用 [[Mixture-of-Transformers]] 将视觉专家与动作专家解耦，训练时联合预测视觉未来帧与动作块，推理时仅走动作路径，避免像素级未来预测开销
2. **AC-WM 混合预训练**: 在 2K 小时机器人数据上联合训练 [[Action-Conditioned World Model|Action-Conditioned World Modeling (AC-WM)]] 和 WAM 目标，使动作表示学习到更可迁移的世界动力学先验，收敛更快
3. **AutoResearch 自动化**: [[AutoResearch]] 智能体流水线通过 1K-step 试探性扫描系统探索超参空间（学习率、批大小），选出 lr=6×10⁻⁵ 并延伸训练至 30K 步，替代手动调参

---

## 问题背景

### 要解决的问题
[[World Action Model]]（WAM）在训练时利用未来视觉预测提供密集监督信号，但大多数现有 WAM 在推理时也需要生成未来视频帧，造成巨大计算开销（如 [[Motus]] 推理延迟达 3231ms），无法满足实时机器人控制需求。

### 现有方法的局限
- **[[Motus]]**：统一的潜空间动作世界模型，推理时需生成未来帧，A100 延迟 3231ms
- **[[Fast-WAM]]**：尝试解耦训练与推理，但单专家架构 A100 延迟仍有 229ms，且难以利用视频预训练权重
- **[[Flash-WAM]]**：侧重训练效率优化，推理延迟问题未完全解决
- 所有方法均缺乏系统性超参优化流程，依赖人工经验设置

### 本文的动机
"世界建模的收益不必依赖推理时昂贵的像素级未来预测"——只需在训练阶段保留视觉监督信号，推理阶段走轻量级动作专用路径，即可兼顾训练质量与部署效率。

---

## 方法详解

### 模型架构

GigaWorldPolicy0.5 采用 **[[Mixture-of-Transformers]] (MoT)** 架构：

- **输入**: 语言指令 $l$ + 多视角复合观测 $o_t^{\text{comp}}$ + 本体感知状态 $s_t$
- **视觉专家（Visual Expert）**: 处理当前帧与未来预测帧的视觉 token，建模场景演化；hidden dim=3072，FFN dim=14336；从 GigaWorld-1 视频生成模型（10000+ 小时视频数据）初始化
- **动作专家（Action Expert）**: 轻量级结构，专注 [[Action Chunking|动作块]] token 去噪；hidden dim=1024，FFN dim=4096；从视觉专家权重截断初始化（维度不匹配部分裁剪）
- **多模态自注意力（Multi-Modal Self-Attention）**: 连接两个专家，保留动作中心因果掩码（[[Action Masking|causal masking]]）——动作 token 可注意当前帧但不注意未来帧，未来视觉 token 可注意动作 token
- **输出**: 动作块 $\hat{a}_{t:t+p-1}$ 和可选未来观测 $\hat{o}_{t+\Delta:t+K\Delta}^{\text{comp}}$

### 核心模块

#### 模块 1：动作中心 [[Mixture-of-Transformers|MoT]] 因果掩码

**设计动机**: 利用[[Action-Conditioned World Model|动作条件世界模型]]原理，在训练时让未来视觉预测隐式依赖动作 token（"未来视觉 token 可以注意动作 token"），同时防止动作预测被未来帧信息污染，从而支持推理时跳过未来帧生成。

**具体实现**:
- 动作 token → 注意当前视觉 + 语言 token（不注意未来视觉）
- 未来视觉 token → 注意动作 token + 当前视觉 token
- 推理时直接丢弃未来视觉解码路径，仅运行动作专家

#### 模块 2：多视角复合观测编码

**具体实现**:
- 将前视图（front view）置于上方，左右视图（left/right views）拼接于下方，构成单张复合图像 $o^{\text{comp}}$
- 统一输入尺寸，兼容预训练视频模型的图像编码器

#### 模块 3：C++ 统一运行时

**具体实现**:
- 原生 C++ 流水线集成：预处理 → tensor 构建 → 模型执行（含 [[KV-Cache Conditioning|KV Cache]] 复用）→ 后处理
- 联合 `torch.compile` 图编译（算子融合 + 内存优化），在 RTX 4090 实现 85ms 延迟

---

## 关键公式

### 公式 1：[[World Action Model|WAM 联合生成目标]]

$$
(\hat{\mathbf{a}}_{t:t+p-1},\; \hat{\mathbf{o}}_{t+\Delta:t+K\Delta}^{\text{comp}}) \;\sim\; g_\theta\!\left(\cdot \mid o_t^{\text{comp}}, s_t, l\right)
$$

**含义**: 模型 $g_\theta$ 在当前复合观测 $o_t^{\text{comp}}$、本体状态 $s_t$、语言指令 $l$ 条件下，联合采样长度为 $p$ 的动作块和 $K$ 步未来观测序列。

**符号说明**:
- $\hat{\mathbf{a}}_{t:t+p-1} = (\hat{a}_t, \hat{a}_{t+1}, \ldots, \hat{a}_{t+p-1})$：预测动作块
- $\hat{\mathbf{o}}_{t+\Delta:t+K\Delta}^{\text{comp}}$：$K$ 步未来复合观测，间隔 $\Delta$
- $g_\theta$：MoT 参数化的联合生成模型

---

### 公式 2：[[Action-Conditioned World Model|复合观测构成]]

$$
o^{\text{comp}} = \text{Compose}(o^{\text{left}}, o^{\text{front}}, o^{\text{right}})
$$

**含义**: 将三路相机图像通过空间拼接策略合并为单张复合图像，作为统一视觉输入。

**符号说明**:
- $o^{\text{left}}, o^{\text{front}}, o^{\text{right}}$：左、前、右三视角原始帧
- $\text{Compose}(\cdot)$：前视图在上、左右视图拼于下方的空间拼接函数

---

### 公式 3：[[Flow Matching|模态独立流时间步采样]]

$$
\tau^a = \frac{\gamma_a\, r^a}{1 + (\gamma_a - 1)\, r^a}, \qquad \tau^v = \frac{\gamma_v\, r^v}{1 + (\gamma_v - 1)\, r^v}
$$

联合时间步向量：

$$
\tau = \begin{bmatrix} \tau^a \\ \tau^v \end{bmatrix}
$$

**含义**: 对动作和视觉模态分别采样独立的流匹配时间步，通过 flow-shift 因子 $\gamma_a, \gamma_v$ 控制各模态噪声调度，避免单一时间步对两种模态都次优。

**符号说明**:
- $r^a, r^v \sim \mathcal{U}(0,1)$：动作/视觉模态的均匀随机数
- $\gamma_a, \gamma_v$：flow-shift 超参，控制噪声重心偏移
- $\tau^a, \tau^v$：动作/视觉模态各自的插值时间步 $\in [0,1]$

---

### 公式 4：[[Flow Matching|带噪联合 token 构建]]

$$
x_\tau = \tau \begin{bmatrix} x_1^a \\ x_1^v \end{bmatrix} + (1 - \tau) \begin{bmatrix} x_0^a \\ x_0^v \end{bmatrix}
$$

**含义**: 将干净样本 $x_1$（动作/视觉）与纯噪声 $x_0$ 按 $\tau$ 线性插值，得到送入模型的中间带噪 token。

**符号说明**:
- $x_1^a, x_1^v$：动作/视觉干净数据
- $x_0^a, x_0^v$：对应模态的高斯噪声
- $\tau$：分模态时间步向量

---

### 公式 5：[[Flow Matching|目标速度场]]

$$
\nu_\tau = \frac{dx_\tau}{d\tau} = \begin{bmatrix} x_1^a - x_0^a \\ x_1^v - x_0^v \end{bmatrix}
$$

**含义**: 流匹配的目标是拟合噪声到数据的直线速度场，即"干净数据减去噪声"，方向明确且无需时间步相关系数。

---

### 公式 6：[[Flow Matching|联合训练损失]]

$$
\mathcal{L} = \mathbb{E}_{x_1, x_0, \tau} \left[ \left\| g_\theta\!\left(x_\tau, \tau \mid o_t^{\text{comp}}, s_t, l\right) - \nu_\tau \right\|_2^2 \right]
$$

**含义**: 用 $L_2$ 距离监督模型预测速度场 $\nu_\tau$，同时优化动作预测（主目标）和未来视觉预测（辅助监督），共享梯度更新视觉专家和动作专家。

**符号说明**:
- $g_\theta(\cdot)$：MoT 模型输出的速度场预测
- $\nu_\tau$：目标速度（公式 5）
- 期望对随机噪声 $x_0$、真实数据 $x_1$、时间步 $\tau$ 取均值

---

## 关键图表

### Figure 1：推理频率 vs 成功率对比

![Figure 1](https://arxiv.org/html/2607.13960v2/x1.png)

**说明**: GigaWorldPolicy0.5（RTX 4090 + C++）以 85ms 延迟（约 11.8 Hz）达到 0.85 成功率，在"推理速度—任务性能"帕累托前沿上全面优于所有基线。[[Motus]] 成功率相当但延迟超 3000ms；[[Fast-WAM]] 速度较快但性能明显更低。

---

### Figure 2：GigaWorldPolicy0.5 MoT 架构总览

![Figure 2](https://arxiv.org/html/2607.13960v2/x2.png)

**说明**: 左侧为训练模式——视觉专家处理当前帧与 $K$ 步未来帧，动作专家处理 [[Action Chunking|动作块]] token；多模态自注意力连接两者，动作 token 不注意未来帧（因果掩码）。右侧为推理模式——仅激活动作专家路径，跳过未来视觉生成。

---

### Figure 3：Tableware Arrangement 长时域任务演示

![Figure 3](https://arxiv.org/html/2607.13960v2/x3.png)

**说明**: 展示碗碟摆放任务的多步骤执行过程，GigaWorldPolicy0.5 成功率达 0.80，较最强基线 [[Motus]] (0.60) 提升 33%。

---

### Figure 4：Food Heating 长时域任务演示

![Figure 4](https://arxiv.org/html/2607.13960v2/x4.png)

**说明**: 展示食物加热任务的连续操作，成功率同样为 0.80，较 [[Motus]] (0.60) 和 π₀.₅ (0.50) 均有显著提升。

---

### Figure 5：AC-WM 消融—不同训练步数成功率

![Figure 5](https://arxiv.org/html/2607.13960v2/figures/ablation_acwm.png)

**说明**: 混合 [[Action-Conditioned World Model|AC-WM]] 预训练（蓝线）在训练全程均优于纯 WAM 预训练（橙线），最终收敛到 0.85 成功率，证明显式动作-视觉转换建模带来更可迁移的动作表示。

---

### Figure 6：[[AutoResearch]] 超参搜索与训练过程

![Figure 6](https://arxiv.org/html/2607.13960v2/figures/autoresearch.png)

**说明**: 左图为 1K-step 试探性学习率扫描（4 个候选值），右图为选定 lr=6×10⁻⁵ 后延伸训练至 30K 步的验证集 MSE 曲线，峰值性能出现在约 30K 步。

---

### Table 1：文本跟随能力——水果抓取任务

| 任务 | 语言指令 | π₀.₅ | Motus | FastWAM | GigaWorld-Policy | **GigaWorldPolicy0.5** |
|------|----------|------|-------|---------|-----------------|------------------------|
| Pick the Fruit | Pick banana, place in basket | 0.88 | 0.93 | 0.83 | 0.90 | **0.95** |
| | Pick apple, place in basket | 0.73 | 0.78 | 0.75 | 0.73 | **0.80** |
| | Pick lemon, place in basket | 0.68 | 0.75 | 0.73 | 0.78 | **0.83** |
| | Pick grape, place in basket | 0.85 | 0.83 | 0.88 | 0.88 | **0.93** |
| | Pick avocado, place in basket | 0.68 | 0.70 | 0.75 | 0.73 | **0.78** |
| | Pick strawberry, place in basket | 0.78 | 0.80 | 0.73 | 0.78 | **0.85** |
| **平均** | – | 0.76 | 0.80 | 0.78 | 0.80 | **0.85** |

**关键发现**: GigaWorldPolicy0.5 在全部 6 种水果上均排第一，平均成功率超次优基线 [[Motus]] 0.05。

---

### Table 2：文本跟随能力——物体摆放任务

| 任务 | 语言指令 | π₀.₅ | Motus | FastWAM | GigaWorld-Policy | **GigaWorldPolicy0.5** |
|------|----------|------|-------|---------|-----------------|------------------------|
| Place Objects | Bowl on plate | 0.87 | 0.88 | 0.73 | 0.80 | **0.93** |
| | Fork on plate | 0.78 | 0.83 | **0.85** | 0.80 | 0.88 |
| | Spoon on plate | 0.73 | 0.83 | 0.80 | 0.75 | **0.85** |
| | Bowl into basket | 0.75 | 0.83 | 0.80 | 0.88 | **0.95** |
| | Fork into basket | 0.65 | 0.75 | 0.68 | 0.78 | **0.85** |
| | Spoon into basket | 0.80 | 0.85 | 0.75 | 0.83 | **0.88** |
| **平均** | – | 0.76 | 0.83 | 0.77 | 0.81 | **0.89** |

**关键发现**: 平均成功率超最强基线 [[Motus]] 0.06。

---

### Table 3：长时域任务成功率

| 任务 | π₀.₅ | Motus | FastWAM | GigaWorld-Policy | **GigaWorldPolicy0.5** |
|------|------|-------|---------|-----------------|------------------------|
| Food Heating | 0.50 | 0.60 | 0.50 | 0.60 | **0.80** |
| Tableware Arrangement | 0.70 | 0.60 | 0.50 | 0.60 | **0.80** |
| **平均** | 0.60 | 0.60 | 0.50 | 0.60 | **0.80** |

**关键发现**: 长时域任务提升最显著，相对最强基线提升 33%。

---

### Table 4：推理效率对比

| 方法 | A100 延迟 (ms) | RTX 4090 延迟 (ms) | 成功率 |
|------|----------------|-------------------|--------|
| π₀.₅ | 225 | 110 | 0.76 |
| [[Motus]] | 3231 | — | 0.80 |
| [[Fast-WAM]] | 229 | 182 | 0.78 |
| GigaWorld-Policy | 360 | 293 | 0.80 |
| **GigaWorldPolicy0.5** | **189** | **110** | **0.85** |
| GigaWorldPolicy0.5 + C++ | **140** | **85** | **0.85** |

**关键发现**: 相比 [[Fast-WAM]]，A100 延迟降低 17.5%；C++ 部署后 RTX 4090 延迟仅 85ms（比 π₀.₅ 快 23%，比 FastWAM 快 53%），同时成功率更高。

---

### Table 5：[[AutoResearch]] 学习率扫描结果

| 学习率 | 训练 Action Loss | 训练 Visual Loss | 评估 Action MSE |
|--------|-----------------|-----------------|----------------|
| 3×10⁻⁵ | 0.257300 | 0.172330 | 0.416832 |
| 4.316×10⁻⁵ | 0.256593 | 0.172893 | 0.449387 |
| **6×10⁻⁵** | **0.252476** | 0.173318 | **0.409764** |
| 8×10⁻⁵ | 0.261832 | 0.175986 | 0.461381 |

**关键发现**: lr=6×10⁻⁵ 在训练 Action Loss 和评估 MSE 上均最优，被 [[AutoResearch]] 自动选出用于完整训练。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 开源 + 内部机器人数据 | 2K 小时 | 经筛选的多场景机器人操作视频 | 预训练 |
| GigaWorld-1 视频 | 10,000+ 小时 | 大规模通用视频，用于视觉专家初始化 | 骨干预训练 |
| 目标机器人轨迹 | 真实操作演示 | 带对齐的观测、指令、状态、动作 | 后训练 |

### 实现细节

- **Backbone**: [[Mixture-of-Transformers]]（视觉专家 hidden=3072/FFN=14336，动作专家 hidden=1024/FFN=4096）
- **优化器**: AdamW，学习率 6×10⁻⁵（[[AutoResearch]] 选定）
- **Batch Size**: 16
- **预训练步数**: 2K 小时机器人数据
- **后训练步数**: 约 30K steps（[[AutoResearch]] 确定）
- **推理优化**: [[KV-Cache Conditioning|KV Cache]] + `torch.compile` 图编译 + C++ 统一运行时
- **硬件**: A100 / RTX 4090

### 可视化结果

- Tableware Arrangement：机械臂按语言指令（"bowl on plate"等）逐一完成多步摆放，全程无手动干预
- Food Heating：连续多阶段操作（抓取食物→放入加热区→取出）成功率 0.80

---

## 批判性思考

### 优点
1. **效率与性能双赢**: 85ms 延迟（RTX 4090 + C++）在所有 WAM 中最优，且成功率超过所有对比方法
2. **架构设计简洁**: MoT 解耦视觉/动作处理，推理时自然跳过未来帧路径，无需特殊 inference 逻辑
3. **自动化调参**: [[AutoResearch]] 减少人工介入，1K-step 试探性扫描经济高效
4. **视频模型迁移**: 视觉专家直接从 GigaWorld-1（万小时视频）初始化，充分利用大规模视觉先验

### 局限性
1. **硬件依赖**: 85ms 延迟仅在高端 GPU（RTX 4090）上实现，边缘部署场景未评估
2. **评估任务有限**: 仅在 3 类桌面操作任务上测试，泛化到更复杂场景（移动操作、非结构化环境）未验证
3. **[[AutoResearch]] 细节不足**: 论文未详述 AutoResearch 的搜索空间设计和 agent 架构，复现难度较高
4. **Future Frame 的边际收益未量化**: 虽有 AC-WM 消融，但未明确量化未来帧辅助监督对最终成功率的贡献曲线

### 潜在改进方向
1. 将 [[AutoResearch]] 扩展到架构超参（专家维度、层数），进一步自动化 NAS
2. 探索更轻量的动作专家（如 SSM / linear attention）以进一步降低推理延迟
3. 在移动操作和足式机器人等更广泛场景验证方法泛化性

### 可复现性评估
- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（超参、数据规模均在论文中说明）
- [ ] 数据集可获取（内部数据未公开）

---

## 关联笔记

### 基于
- [[GigaWorld]]: GigaWorld-1 视频生成模型，为视觉专家提供预训练权重
- [[World Action Model]]: WAM 范式基础，GigaWorldPolicy0.5 是其加速改进版
- [[Flow Matching]]: 训练框架，使用模态独立流时间步

### 对比
- [[Motus]]: 同为 MoT 结构的 WAM，但推理时需生成未来帧，延迟极高
- [[Fast-WAM]]: 同样尝试解耦训练/推理，但单专家 A100 延迟 229ms，且本文性能更优
- [[Flash-WAM]]: WAM 训练效率工作，侧重数据/训练优化而非推理加速

### 方法相关
- [[Mixture-of-Transformers]]: 核心架构，视觉/动作专家分离
- [[Action-Conditioned World Model]]: AC-WM 预训练目标，提供动作-视觉转换先验
- [[Flow Matching]]: 联合训练优化框架
- [[Action Chunking]]: 动作输出形式
- [[KV-Cache Conditioning]]: 推理加速中的 KV 复用
- [[AutoResearch]]: 自动超参优化 agent 流水线

### 硬件/数据相关
- GigaWorld-1: 视觉专家的视频预训练基础模型

---

## 速查卡片

> [!summary] GigaWorld-Policy-0.5
> - **核心**: MoT 视觉/动作专家分离 + AC-WM 混合预训练 + AutoResearch 自动调参
> - **方法**: 训练时联合预测动作块与未来帧；推理时仅走动作专家路径（85ms @ RTX 4090 C++）
> - **结果**: 水果抓取 SR=0.85、物体摆放 SR=0.89、长时域任务 SR=0.80，全面超越 π₀.₅/Motus/FastWAM
> - **代码**: 未开源（项目主页: https://open-gigaai.github.io/giga-world-policy/）

---

*笔记创建时间: 2026-07-17*
