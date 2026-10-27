---
title: "WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time"
method_name: "WAM-TTT"
authors: [Yusen Feng, Bingchen Han, Jiangran Lyu, Kai Liu, Yixin Zheng, Yuxuan Wan, Weiheng Liu, Sun Han, Ruiqin Li, Yulong Zhang, Fangfu Liu, Xuesong Shi, Libin Liu, Yizhou Wang, Zhizheng Zhang, He Wang]
year: 2026
venue: arXiv
tags: [world-action-model, test-time-training, robot-manipulation, meta-learning, fast-weights]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.06988
created: 2026-10-27
---

# 论文笔记：WAM-TTT

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University, BIGAI |
| 日期 | July 2026 |
| 项目主页 | [wam-ttt.github.io](https://wam-ttt.github.io) |
| 对比基线 | [[LDA]], [[π₀.₅]], [[EgoScale]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.06988) |

---

## 一句话总结

> WAM-TTT 通过在部署时用无标注人类视频做[[测试时训练|测试时训练（TTT）]]更新[[Fast Weights|快权重]]，将冻结的[[World Action Model|世界动作模型]]引导到新任务，无需额外机器人示范或微调。

---

## 核心贡献

1. **无机器人数据的部署时适应**: 仅用无标注人类玩耍视频，在测试时自监督更新[[Fast Weights|快权重]]，无需机器人示范、任务专用微调或长上下文条件化。
2. **键值记忆重建目标（KVM）**: 设计了一个将人类帧写入、机器人帧读取的[[Key-Value Memory|键值记忆]]重建损失，使人机行为在元训练阶段对齐。
3. **元训练框架**: 用 [[MAML]] 风格的双层优化，内循环在人类视频上适应[[Fast Weights|快权重]]，外循环优化 WAM 和慢投影参数，使快权重收敛到能帮助机器人的状态。

---

## 问题背景

### 要解决的问题

机器人基础模型（如 [[World Action Model|WAM]]）难以在部署时被引导到新任务或用户偏好的行为，现有方案都需要额外的机器人示范、任务专用微调或超长上下文条件化。

### 现有方法的局限

- **同步训练（Co-training / Cotrain）**: 需要在训练期就混入人类视频，新场景下无法即时适应。
- **上下文内学习（ICL）**: WAM-ICL 在分布外（New）场景下性能崩溃（留存率仅 0.15），说明单纯上下文化人类视频不可靠。
- **需要额外监督的人类视频方法（EgoScale 等）**: 依赖手部姿态估计、重定向轨迹或潜在计划，监督获取成本高。

### 本文的动机

人类玩耍视频是廉价的、无标注的，且包含丰富的操作先验。关键洞见是：把人类视频当作**部署时的记忆**而非轨迹来模仿——用自监督视频预测把人类动态写入[[Fast Weights|快权重]]，让机器人的 WAM 在推理时能读取这些快权重。

---

## 方法详解

### 模型架构

WAM-TTT 以 [[LDA]]（一个含视频专家和动作专家的[[Diffusion Transformer|扩散 Transformer]]）为骨干，**仅在视频专家侧**叠加 TTT 残差分支：

- **输入**: 多视角 RGB 观测 $\bm{o}_t$ + 语言目标 $\bm{g}$ + 人类视频剪辑 $\bm{u}_h$
- **Backbone**: [[LDA]]（Video Expert + Action Expert 双流 [[Diffusion Transformer|扩散 Transformer]]）
- **TTT 分支**: 每个视频专家层插入快权重 MLP $f_W$，通过慢投影 $\theta_{K,V,Q,O}$ 读写人类帧信息
- **输出**: 动作块 $\bm{a}_{t:t+k}$
- **冻结参数**: 测试时 WAM 主干、慢投影初始化全部冻结，只有快权重 $W$ 被更新

### 核心模块

#### 模块 1：TTT 残差分支

**设计动机**: 利用[[Fast Weights|快权重]] $W$ 存储人类视频的动态信息，使视频专家的表示能被人类先验调制，同时保持动作专家不变。

**具体实现**:
- 视频流通过残差更新：$\bm{z}^{(\ell+1)} = \hat{\bm{z}}^{(\ell+1)} + \Delta\bm{z}_{\mathrm{TTT}}^{(\ell)}$，动作流不变：$\bm{x}^{(\ell+1)} = \hat{\bm{x}}^{(\ell+1)}$
- TTT 读出由慢投影 $\theta_Q$ 提取查询，经快权重 MLP $f_W$ 变换，再由 $\theta_O$ 投影回视频维度

#### 模块 2：键值记忆重建损失（KVM）

**设计动机**: 桥接人类帧与机器人帧的分布差异——让快权重 $W$ 学习将人类键（Keys）映射到人类值（Values），使机器人查询能通过已适应的 $W$ 检索人类语义。

**具体实现**:
- 人类帧经慢投影得到 $\bm{K}_h^{(\ell)}, \bm{V}_h^{(\ell)}$，机器人帧得到 $\bm{Q}_r^{(\ell)}$
- KVM 损失要求 $f_W$ 在人类键上重建人类值，作为辅助目标约束快权重

#### 模块 3：归一化 Episode 相位对齐

**设计动机**: 人类和机器人 episode 时长不同，无法直接对齐帧索引。

**具体实现**:
- 定义归一化相位 $\phi = t / T$，将人类帧和机器人帧按 $\phi$ 配对，实现技能阶段同步。

---

## 关键公式

### 公式 1：[[Fast Weights|TTT 残差流]]更新

$$
\bm{z}^{(\ell+1)} = \hat{\bm{z}}^{(\ell+1)} + \Delta\bm{z}_{\mathrm{TTT}}^{(\ell)}, \qquad \bm{x}^{(\ell+1)} = \hat{\bm{x}}^{(\ell+1)}
$$

**含义**: 视频流叠加 TTT 残差，动作流保持不变，实现对视频专家的轻量调制。

**符号说明**:
- $\bm{z}^{(\ell)}$: 第 $\ell$ 层视频 token
- $\bm{x}^{(\ell)}$: 第 $\ell$ 层动作 token
- $\hat{\cdot}$: LDA 原始块输出（无 TTT）
- $\Delta\bm{z}_{\mathrm{TTT}}^{(\ell)}$: TTT 残差（见下式）

### 公式 2：[[Fast Weights|TTT 读出]]

$$
\Delta\bm{z}_{\mathrm{TTT}}^{(\ell)} = \theta_O^{(\ell)}\, f_{W^{(\ell)}}\!\left(\theta_Q^{(\ell)}(\bm{z}^{(\ell)})\right)
$$

**含义**: 视频 token 经慢查询投影 $\theta_Q$，输入快权重 MLP $f_W$，再经慢输出投影 $\theta_O$ 得到残差。

**符号说明**:
- $\theta_Q^{(\ell)}, \theta_O^{(\ell)}$: 可学习的慢投影矩阵
- $f_{W^{(\ell)}}$: 由快权重 $W^{(\ell)}$ 参数化的 MLP，在测试时被更新

### 公式 3：[[Key-Value Memory|键值投影]]

$$
\bm{K}_h^{(\ell)} = \theta_K^{(\ell)}(\bm{h}_\phi^{(\ell)}), \quad \bm{V}_h^{(\ell)} = \theta_V^{(\ell)}(\bm{h}_\phi^{(\ell)}), \quad \bm{Q}_r^{(\ell)} = \theta_Q^{(\ell)}(\bm{z}_r^{(\ell)})
$$

**含义**: 人类帧 $\bm{h}_\phi$ 分别经键/值投影，机器人帧 $\bm{z}_r$ 经查询投影，为 KVM 损失提供输入。

**符号说明**:
- $\bm{h}_\phi^{(\ell)}$: 相位 $\phi$ 对齐的人类帧表示
- $\bm{z}_r^{(\ell)}$: 机器人帧表示

### 公式 4：[[Key-Value Memory|键值记忆重建损失]]（KVM）

$$
\mathcal{L}_{\mathrm{KVM}}^{(\ell)}(W_i) = \frac{1}{BL_h d}\left\|\,f_{W_i^{(\ell)}}(\bm{K}_h^{(\ell)}) - \bm{V}_h^{(\ell)}\right\|_2^2
$$

**含义**: 要求快权重 MLP 在人类键上精确重建人类值，确保 $W$ 编码了人类动态。

**符号说明**:
- $B$: batch size；$L_h$: 人类序列长度；$d$: 特征维度
- $f_{W_i^{(\ell)}}$: 当前快权重 MLP

### 公式 5：[[MAML|内循环适应目标]]

$$
\mathcal{L}_{\mathrm{adapt}}(W_i) = \mathcal{L}_{\mathrm{vg}}^{\mathrm{human}}(\bm{u}_h;\Theta_{\mathrm{WAM}},\theta_{\mathrm{TTT}},W_i) + \lambda\sum_\ell \mathcal{L}_{\mathrm{KVM}}^{(\ell)}(W_i)
$$

**含义**: 内循环同时优化人类视频生成损失和 KVM 重建损失，每个样本后重置快权重。

**符号说明**:
- $\mathcal{L}_{\mathrm{vg}}^{\mathrm{human}}$: 人类视频预测（视频生成）损失
- $\lambda$: KVM 权重系数

### 公式 6：[[MAML|内循环 SGD 更新]]

$$
W_{i+1}^{(\ell)} = W_i^{(\ell)} - \eta\,\nabla_{W_i^{(\ell)}}\mathcal{L}_{\mathrm{adapt}}(W_i)
$$

**含义**: 快权重通过单步 SGD 在人类视频适应目标上更新。

**符号说明**:
- $\eta$: 内循环学习率
- $W_i^{(\ell)}$: 第 $i$ 步迭代时第 $\ell$ 层快权重

### 公式 7：[[MAML|外循环元训练目标]]

$$
\mathcal{L}_{\mathrm{meta}} = \mathcal{L}_{\mathrm{WAM}}^{\mathrm{robot}} + \lambda\sum_\ell \mathcal{L}_{\mathrm{KVM}}^{(\ell)}(W_N)
$$

**含义**: 外循环用机器人数据优化 WAM 标准损失，同时在内循环收敛后的快权重上追加 KVM 损失，使慢参数（$\Theta_{\mathrm{WAM}}, \theta_{\mathrm{TTT}}$）学会配合快权重工作。

**符号说明**:
- $\mathcal{L}_{\mathrm{WAM}}^{\mathrm{robot}}$: 标准 WAM 机器人多任务损失
- $W_N$: 内循环 $N$ 步后的快权重

### 公式 8：测试时更新目标

$$
\mathcal{L}_{\mathrm{TTT}}(W_i) = \frac{1}{|\mathcal{B}_h|}\sum_{\bm{u}\in\mathcal{B}_h}\left[\mathcal{L}_{\mathrm{vg}}(\bm{u};\Theta_{\mathrm{WAM}},\theta_{\mathrm{TTT}},W_i) + \lambda\sum_\ell \mathcal{L}_{\mathrm{KVM}}^{(\ell)}(\bm{u};W_i)\right]
$$

**含义**: 测试时对一批无标注人类视频 $\mathcal{B}_h$ 最小化同样的目标，更新快权重，主干和慢参数保持冻结。

**符号说明**:
- $\mathcal{B}_h$: 部署时可用的人类视频 batch
- 测试时所有参数中**只有** $W^{(\ell)}$ 被更新

### 公式 9：机器人动作采样（推理）

$$
\bm{a}_{t:t+k} \sim p_{\Theta_{\mathrm{WAM}},\theta_{\mathrm{TTT}},W_N}(\bm{a}_{t:t+k}\mid\bm{o}_t,\bm{g})
$$

**含义**: 快权重适应完成并固定后，使用完整模型（WAM + TTT 分支）进行机器人动作采样，不再更新参数。

**符号说明**:
- $W_N$: TTT 收敛后固定的快权重
- $\bm{o}_t$: 当前观测；$\bm{g}$: 语言目标

### 公式 10：理论分析——快权重闭式解（附录 A）

$$
W^* = V_h^\top K_h \left(K_h^\top K_h\right)^{-1}
$$

**含义**: 线性情形下 KVM 损失的解析解是人类键值的外积，等价于对人类键值做[[线性注意力|线性注意力]]的 Hebbian 记忆。

**符号说明**:
- $K_h, V_h$: 所有人类帧的键和值矩阵（堆叠）

---

## 关键图表

### Figure 1：系统概览（Teaser）

![Figure 1](https://arxiv.org/html/2607.06988v3/teaser.png)

**说明**: WAM-TTT 的核心思路——人类玩耍视频在测试时通过 TTT 快权重写入世界动作模型，机器人在推理时读取这些记忆完成操作任务，整个过程无需机器人示范。

### Figure 2：方法流水线

![Figure 2](https://arxiv.org/html/2607.06988v3/pipeline.png)

**说明**: 左侧为元训练阶段，人类帧（绿色）通过 $\theta_K, \theta_V$ 写入快权重，机器人帧（蓝色）通过 $\theta_Q$ 读取；右侧为测试时，仅人类视频更新快权重，随后机器人利用冻结的适应权重执行动作。

### Figure 3：实验配置

![Figure 3](https://arxiv.org/html/2607.06988v3/fig_setup.jpg)

**说明**: 三种机器人本体（Unitree G1 人形、Galbot 夹爪、Galbot Sharpa 灵巧手）和 9 项操作任务，分为标准化方舱（Orig.）和未见家庭场景（New）两种评测设置。

### Figure 4：定性结果 Gallery

![Figure 4](https://arxiv.org/html/2607.06988v3/fig_gallery.jpg)

**说明**: WAM-TTT 在未见家庭场景中的定性执行结果，各行展示一次完整推演的 5 个关键帧，覆盖全部 9 项基准任务。

### Figure 5：泛化设置

![Figure 5](https://arxiv.org/html/2607.06988v3/sections/generalization.jpg)

**说明**: 对比 WAM-TTT 与 LDA 在光照扰动和空间扰动（Deliver Drink 任务）下的性能，TTT 分支通过快权重吸收了新场景分布，鲁棒性显著提升。

### Figure B.1：机器人数据采集（方舱）

![Figure B.1](https://arxiv.org/html/2607.06988v3/sections/corl_cubic.jpg)

**说明**: 标准化方舱中的机器人数据采集环境，用于训练 WAM 基础能力（2286 个配对人-机 episode）。

### Figure B.2：人类数据采集（家庭场景）

![Figure B.2](https://arxiv.org/html/2607.06988v3/sections/corl_human_new_scene.jpg)

**说明**: 多样家庭场景中的人类无标注视频采集，仅用于测试时 TTT 更新。

### Figure E.1：定性 Gallery（完整）

![Figure E.1](https://arxiv.org/html/2607.06988v3/sections/corl_main_result.jpg)

**说明**: WAM-TTT 在家庭、办公室、厨房未见场景的完整定性推演 gallery，每行 5 帧，包含额外野外演示。

### Figure E.2：压力测试——零场景人类数据

![Figure E.2](https://arxiv.org/html/2607.06988v3/corl_appendix_stress_test.jpg)

**说明**: 直接泛化到未见实验室场景（光照、桌布、物体、目标位置同时改变），无任何场景专属人类数据，验证 WAM-TTT 的极端泛化能力。

### Figure E.3：六维扰动鲁棒性

![Figure E.3](https://arxiv.org/html/2607.06988v3/sections/corl_generalization_full.jpg)

**说明**: 六个场景内分布偏移轴（物体、光照、高度、底盘/背景、全新印章/可供性、物体位置）下的性能，WAM-TTT 快权重仅更新视频专家分支，动作专家不受干扰。

### Figure E.4：伪动作标注（MANO）

![Figure E.4](https://arxiv.org/html/2607.06988v3/sections/annotated_row5.jpg)

**说明**: 用 MediaPipe 关键点 + MANO 网格拟合得到的人类单视角手部姿态标注，用于消融实验中对比伪动作前向动力学损失的效果（添加后性能反而下降，最终不采用）。

### Figure E.5：FastWAM 视频预测（Pyramid Stacking）

![Figure E.5](https://arxiv.org/html/2607.06988v3/fastwam_cup_gallery.png)

**说明**: WAM-TTT 在 Pyramid Stacking 任务上的视频预测结果，展示左头、左腕、右腕三路相机的预测帧（上）和真实帧（下），验证快权重有效编码了人类操作动态。

### Table 1：主结果（完整 9 任务）

| 方法 | 设置 | Transfer Bottle | Table Bussing | Deliver Drink | Swap Place | Pour Water | Stamp Paper | Flip Steak | Pyramid Stacking | Multi-step Steak | 均值 |
|------|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| π0.5 | Orig. | 55.5 | 80.0 | 70.0 | 12.2 | 25.0 | 31.1 | 4.4 | 6.1 | 2.0 | 31.8 |
| π0.5 | **New** | 33.4 | 36.0 | 15.0 | 7.4 | 10.0 | 24.4 | 2.0 | 4.7 | 0.3 | 14.8 |
| LDA | Orig. | 72.0 | 90.0 | 80.0 | 66.7 | 33.6 | 50.0 | 33.3 | 6.7 | 19.2 | 50.2 |
| LDA | **New** | 56.0 | 70.0 | 55.0 | 44.4 | 20.0 | 33.3 | 10.0 | 0.6 | 3.0 | 32.5 |
| EgoScale | Orig. | 69.4 | 80.0 | 69.6 | 10.0 | 30.0 | 2.7 | 5.0 | 2.0 | 2.0 | 30.1 |
| EgoScale | **New** | 34.4 | 44.0 | 33.3 | 6.0 | 7.5 | 1.1 | 5.0 | 2.0 | 2.0 | 15.0 |
| WAM-Cotrain | Orig. | 11.6 | 40.0 | 59.4 | 74.1 | 26.3 | 10.0 | 21.0 | 9.4 | 16.8 | 29.8 |
| WAM-Cotrain | **New** | 10.0 | 10.0 | 44.3 | 48.1 | 24.0 | 21.7 | 34.2 | 12.0 | 23.8 | 25.3 |
| WAM-ICL | Orig. | 60.0 | 89.0 | 70.3 | 68.7 | 55.3 | 36.0 | 33.3 | 4.7 | 18.2 | 48.4 |
| WAM-ICL | **New** | 10.0 | 10.0 | 14.2 | 10.0 | 0.0 | 5.0 | 10.0 | 2.0 | 2.5 | 7.1 |
| **WAM-TTT** | **Orig.** | **77.8** | **100.0** | **90.0** | **88.9** | **63.3** | **35.0** | **44.0** | **10.0** | **40.6** | **61.1** |
| **WAM-TTT** | **New** | **55.6** | **100.0** | **66.7** | **66.7** | **30.0** | 8.3 | **34.3** | **10.4** | **43.8** | **46.2** |

**关键发现**: WAM-TTT 在 New 设置下均值 46.2%，领先第二名 LDA（32.5%）13.7 个百分点，9 个任务中赢 7 个。WAM-ICL 在 New 下崩溃（留存率 0.15），证明上下文化人类视频不可靠，而 TTT 写入快权重才是关键。

### Table 2：消融实验

| 配置 | Table Bussing (New) | 说明 |
|------|:-------------------:|------|
| 无元训练（仅 TTT） | —（大幅下降） | 快权重无法从人类键值中有效读取 |
| 无 TTT（替换为 [[LoRA]]） | —（大幅下降） | LoRA 无法快速适应，需要更多数据 |
| 无 KVM 损失 | 66.7% | 缺少记忆重建约束，对齐变差 |
| **完整 WAM-TTT** | **100.0%** | — |

**关键发现**: 元训练和 TTT 机制缺一不可；KVM 损失对 Table Bussing 贡献 33.3 个百分点。

### Table 3：数据比例消融（三任务子集，New）

| (机器人, 人类) 数据量 | Transfer Bottle | Table Bussing | Deliver Drink | 均值 |
|----------------------|:-:|:-:|:-:|:-:|
| (100, 0) | 44.1 | 90.0 | 44.4 | 59.5 |
| (10, 190) | 42.1 | 68.0 | 44.2 | 51.4 |
| **(100, 100) [ours]** | **55.6** | **100.0** | **66.7** | **74.1** |
| (200, 0) | 47.9 | 100.0 | 73.2 | 73.7 |
| (100, 200) | 58.9 | 100.0 | 61.0 | 73.3 |

**关键发现**: 在相同总数据预算下，配对人类数据与等量机器人数据性能相当（74.1 vs 73.7），证明人类视频可作为机器人数据的廉价替代。

### Table 4：模型架构消融（Table Bussing，New）

| 配置 | 进度 (%) |
|------|:---:|
| 仅 DiT（无 VLM 后端） | 72.0 |
| DiT + VLM（无 VLM 预训练） | 80.0 |
| DiT + VLM（VLM 冻结） | 54.0 |
| **DiT + VLM（VLM 开放）[ours]** | **100.0** |

**关键发现**: VLM 后端开放微调是 Table Bussing 满分的关键，冻结 VLM 反而比无 VLM 更差。

### Table 5：伪动作标注消融（四任务子集，New）

| 配置 | Transfer Bottle | Table Bussing | Deliver Drink | Swap Place | 均值 |
|------|:-:|:-:|:-:|:-:|:-:|
| **VG 仅（ours）** | **55.6** | **100.0** | **66.7** | **66.7** | **72.3** |
| VG + FD（MANO 伪动作） | 14.2 | 33.3 | 26.8 | 41.2 | 28.9 |

**关键发现**: 加入 MANO 重定向伪动作和前向动力学损失后性能暴跌 43.4 个百分点，说明动作无关的纯视频预测设计反而更优，伪动作噪声有害。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 配对人-机 episodes | 2,286 个 | 三种本体，9 任务，方舱采集 | 元训练 |
| Orig. 设置 | 25 trials/任务 | 标准化方舱 | 分布内评测 |
| New 设置 | 25 trials/任务 | 未见家庭/办公/厨房场景 | 分布外评测 |

### 实现细节

- **Backbone**: [[LDA]]（LDA diffusion transformer，Video Expert + Action Expert 双流）
- **TTT 分支**: 残差 MLP，插入视频专家每层，快权重通过内循环 SGD 更新
- **内循环步数**: 多步 SGD，每个人类 episode 后快权重重置
- **相位对齐**: 归一化 episode 相位 $\phi = t/T$ 配对人机帧
- **机器人本体**: [[Unitree G1]] 人形机器人、Galbot 夹爪、Galbot Sharpa 灵巧手
- **任务**: Transfer Bottle, Table Bussing, Deliver Drink, Swap Place, Pour Water, Stamp Paper, Flip Steak, Pyramid Stacking, Multi-step Steak
- **评测指标**: 加性里程碑进度（权重和为 1.0），自动从末端执行器和物体位姿计算

### 可视化结果

WAM-TTT 在未见家庭场景中展示出稳定的多步操作能力，包括需要精确抓取的 Transfer Bottle（55.6%）和高难度多步 Multi-step Steak（43.8%）。零场景人类数据的压力测试（Figure E.2）表明快权重已编码了可跨场景迁移的操作先验。

---

## 批判性思考

### 优点

1. **无监督部署适应**: 无需机器人数据、手部姿态标注或轨迹重定向，人类视频直接使用。
2. **解耦设计清晰**: 快权重只修改视频专家，动作专家完全不动，避免遗忘和干扰。
3. **理论支撑**: 附录 A 的线性分析证明 KVM 损失在等方性假设下等价于 Hebbian 线性注意力记忆，有理论基础。
4. **伪动作消融结论反直觉但重要**: 明确指出 MANO 伪动作有害，动作无关设计是正确选择。

### 局限性

1. **相位对齐假设强**: 人类和机器人 episode 必须在技能阶段分布上相似，新场景如果动作顺序差异很大可能对齐失败。
2. **快权重容量瓶颈**: 适应能力受限于快权重 MLP 大小和元训练任务集的覆盖范围，大分布偏移下可能不足。
3. **仅用 egocentric RGB**: 不使用手部姿态、3D 线索，导致遮挡场景下人类帧信息利用不完整。
4. **需要配对元训练数据**: 尽管测试时不需要机器人数据，元训练阶段仍需人机配对 episode，数据获取有一定成本。

### 潜在改进方向

1. 将快权重扩展到动作专家，探索部分解冻的联合适应。
2. 用多模态对齐（视觉-语言）替代纯相位对齐，处理动作顺序差异。
3. 增大快权重容量或改用注意力形式的快权重（如 Titans 风格）。

### 可复现性评估

- [ ] 代码开源（尚未，2026.10 时）
- [ ] 预训练模型
- [x] 训练细节完整（论文附录详细描述）
- [ ] 数据集可获取（方舱数据集未公开）

---

## 关联笔记

### 基于

- [[LDA]]: WAM-TTT 的骨干模型，提供视频专家 + 动作专家双流架构
- [[World Action Model]]: 论文所针对的基础模型类别
- [[MAML]]: 元训练的双层优化框架

### 对比

- [[π₀.₅]]: 通用 VLA 基线，New 设置 14.8%，缺乏场景适应机制
- [[EgoScale]]: 人类 ego 视频监督方法，需要额外人工标注，New 设置 15.0%
- [[LoRA]]: 消融替代快权重的参数高效微调方法，性能远差于 TTT

### 方法相关

- [[测试时训练]]: 核心方法，在测试时用自监督目标更新部分参数
- [[Fast Weights]]: TTT 分支的核心机制，快速适应人类视频分布
- [[Key-Value Memory]]: KVM 损失设计的基础，人类写入/机器人读取的记忆范式
- [[Diffusion Transformer]]: LDA 的架构基础

### 硬件/数据相关

- [[Unitree G1]]: 实验中使用的人形机器人平台
- [[EgoScale]]: 用于对比的人类 ego 视频数据集 / 方法

---

## 速查卡片

> [!summary] WAM-TTT
> - **核心**: 用无标注人类玩耍视频在测试时更新快权重，引导冻结 WAM 适应新场景
> - **方法**: LDA 骨干 + TTT 残差分支 + KVM 损失 + MAML 元训练
> - **结果**: New 设置均值 46.2%，领先 LDA（32.5%）和 π0.5（14.8%）
> - **项目主页**: [wam-ttt.github.io](https://wam-ttt.github.io)

---

*笔记创建时间: 2026-10-27*
