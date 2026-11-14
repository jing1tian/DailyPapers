---
title: "Bind-WAM: Binding-Uncertainty-Guided Selective Future Imagination in World Action Models"
method_name: "Bind-WAM"
authors: [Chenhao Qiu, Shuyi Chen, Songen Gu, Shufeng Nan, Tianyu Wang, Siyu Lin, Yifan Li, Bingwen Zhu, Yanwei Fu]
year: 2026
venue: arXiv
tags: [world-action-model, selective-imagination, binding-uncertainty, robotic-manipulation, flow-matching, bimanual-robot]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2611.15368v1
created: 2026-11-14
---

# 论文笔记：Bind-WAM: Binding-Uncertainty-Guided Selective Future Imagination in World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University, Tongji University, Shanghai Innovation Institute, ROBOTOPIA |
| 日期 | November 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.15368) / Code 未公开 |

---

## 一句话总结

> Bind-WAM 通过预测语言-目标绑定不确定性，选择性地触发 [[WAM|世界动作模型]] 中的未来想象，在保持成功率的同时将想象调用降至 40%，显著加快推理速度。

---

## 核心贡献

1. **绑定不确定性路由框架**: 将 WAM 中选择性未来想象问题建模为状态依赖路由问题，以 [[Language-Object Binding|语言-目标绑定不确定性]] 作为可解释的触发信号。
2. **轻量级不确定性预测器**: 仅用离线标注数据训练约 6.65M 参数的 [[ViT]] 风格预测器，推理时仅需当前帧与语言指令，无需任何额外监督。
3. **统一 WAM 双路径推理**: 一个共享 WAM 参数，通过切换注意力掩码在 "直接推理" 和 "联合想象" 之间路由，消除了两套独立模型的开销。

---

## 问题背景

### 要解决的问题

[[WAM|世界动作模型]] 在每个重规划步骤均生成未来视觉状态（future imagination），计算代价高昂。对于已经稳定、任务相关物体清晰可见的帧，这种想象是冗余的，会拖慢策略执行速度。

### 现有方法的局限

- **[[Fast-WAM]]**：完全跳过未来想象，速度快但在物体遮挡/外观漂移时性能下降；
- **Fast-WAM-Joint（全量想象）**：每步均想象，成功率略高但耗时约 15.7 s/任务（LIBERO），远高于无想象的 9.6 s；
- **[[AdaWAM]]**：自适应想象，但信号来源较为隐式，缺乏可解释性。

### 本文的动机

语言-目标绑定（Language-Object Binding）的准确性直接决定动作生成质量。当绑定不确定（物体被遮挡、特征漂移、指称歧义），策略难以生成稳定动作，需要想象辅助；绑定确定时则无需想象。该信号可解释、可预测、可在单张图像上实时估计。

---

## 方法详解

### 模型架构

Bind-WAM 采用 **双路径共享参数** 架构：

- **输入**: 语言指令 $\ell$ + 当前观测 $o_t$ + 历史视频帧 $e_t$
- **Backbone**: [[Wan 2.2-5B]] 视频生成模型（Wan2.2-5B）
- **核心模块**: [[Binding Uncertainty|绑定不确定性预测器]] 控制 [[Selective Future Imagination|选择性未来想象]] 路由
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H}$，以及（联合模式下）未来视频潜变量 $y^v$
- **预测器参数**: ~6.65M

### 核心模块

#### 模块1: 离线绑定标注流水线（Offline Binding Labeling）

**设计动机**: 自动生成语言-目标绑定不确定性的帧级标签，用于有监督训练预测器。

**具体实现**:

- 使用 [[Qwen3-VL]]（32B）对示例轨迹中的关键目标进行指称定位（Reference Grounding），生成 $N$ 个角色（role）的 bounding box；
- 使用 [[SAM3]]（SAM 3.1）生成并追踪每个角色的实例分割掩码；
- 使用 [[DINOv3]] 在连续帧间测量 IoU 和特征相似度；
- Qwen3-VL 在时序帧上进行多轮投票验证；
- 每条示例轨迹抽样 32 帧进行标注。

#### 模块2: 绑定不确定性预测器（Binding Uncertainty Predictor）

**设计动机**: 仅用当前帧实时预测绑定是否不确定，无需完整历史。

**具体实现**:

- 以 [[ViT]] 为图像编码器，提取 $o_t$ 视觉特征；
- 语言指令 $\ell$ 编码为 query token，与视觉特征交叉注意力融合；
- 输出单个标量 $U_t \in [0,1]$，表示当前帧绑定不确定度。

#### 模块3: 不确定性路由器（Uncertainty Router）

**设计动机**: 利用不确定性绝对值和逐帧变化率双重信号，捕捉 "进入不确定状态" 和 "持续不确定状态" 两种场景。

**具体实现**:

- 计算当前帧不确定度 $U_t$ 和前一帧变化量 $\Delta U_t = |U_t - U_{t-1}|$；
- 若超过阈值则触发 `joint` 路径（联合想象），否则走 `direct` 路径（直接动作生成）；
- 两条路径共享同一 WAM 权重，仅注意力掩码不同。

---

## 关键公式

### 公式1: [[WAM|策略函数]]

$$
\pi(a_{t:t+H} \mid o_t, e_t, \ell)
$$

**含义**: WAM 策略接收当前观测 $o_t$、历史视频 $e_t$ 和语言指令 $\ell$，输出动作块 $a_{t:t+H}$。

**符号说明**:
- $o_t$: 当前时刻 RGB 观测帧
- $e_t$: 历史视频帧序列
- $\ell$: 语言任务指令
- $a_{t:t+H}$: 时间步 $t$ 至 $t+H$ 的动作块

### 公式2: [[Language-Object Binding|角色级不确定性标签]]

$$
y_t^r = \begin{cases} 1, & \mathrm{IoU}_r < 0.3 \;\lor\; v_t^r = 0 \;\lor\; s_t^r < 0.9 \\ 0, & \text{otherwise} \end{cases}
$$

**含义**: 若角色 $r$ 的指称 IoU 过低、目标不可见或 DINO 特征相似度过低，则该帧该角色标记为不确定。

**符号说明**:
- $y_t^r$: 时刻 $t$、角色 $r$ 的不确定性二值标签
- $\mathrm{IoU}_r$: 角色 $r$ 的指称定位 IoU（阈值 0.3）
- $v_t^r$: 目标可见性标志（0 = 不可见）
- $s_t^r$: [[DINOv3]] 特征余弦相似度（阈值 0.9）

### 公式3: [[Binding Uncertainty|帧级不确定性标签]]

$$
y_t = \max_r \; y_t^r
$$

**含义**: 任意一个角色不确定则整帧标记为不确定（worst-case 聚合）。

**符号说明**:
- $y_t$: 帧级不确定性标签
- $r$: 任务相关角色集合

### 公式4: [[Binding Uncertainty|预测器输出]]

$$
U_t = f_\theta(o_t, \ell), \quad U_t \in [0, 1]
$$

**含义**: 轻量 ViT 预测器以观测和指令为输入，输出当前帧绑定不确定概率。

**符号说明**:
- $f_\theta$: 参数量约 6.65M 的 [[ViT]] 风格网络
- $\theta$: 可训练参数

### 公式5: [[Binding Uncertainty|预测器损失]]

$$
\mathcal{L}_{\mathrm{bind}} = \mathrm{BCE}_w(\hat{U}_t, y_t) + \lambda (\hat{U}_t - y_t)^2
$$

**含义**: 类别加权交叉熵（BCE）加 Brier 评分（MSE 项），前者应对类别不平衡，后者促进概率校准。

**符号说明**:
- $\mathrm{BCE}_w$: 类别加权二值交叉熵
- $\hat{U}_t$: 预测不确定概率
- $y_t$: 真实标签
- $\lambda$: Brier 权重系数

### 公式6: [[Selective Future Imagination|直接注意力掩码]]

$$
M_{\mathrm{direct}}(Z^a, Z_h^v) = \begin{cases} 1, & h = 0 \\ 0, & h > 0 \end{cases}
$$

**含义**: 直接路径中，动作 token 仅与当前帧视频 token（$h=0$）交叉注意力，屏蔽历史帧，跳过未来想象。

**符号说明**:
- $Z^a$: 动作 token 序列
- $Z_h^v$: 时刻 $h$ 的视频 token
- $h=0$: 当前帧

### 公式7: [[Selective Future Imagination|联合注意力掩码]]

$$
M_{\mathrm{joint}}(Z^a, Z_h^v) = 1, \quad h = 0, \ldots, H
$$

**含义**: 联合路径中，动作 token 对所有历史和未来视频 token 完全可见，激活完整未来想象。

**符号说明**:
- $H$: 未来视频帧数量

### 公式8: [[Selective Future Imagination|不确定性变化率]]

$$
\Delta U_t = |U_t - U_{t-1}|
$$

**含义**: 当前帧与前一帧不确定性之差的绝对值，捕捉"进入不确定状态"的瞬间。

### 公式9: [[Selective Future Imagination|路由规则]]

$$
\pi(a_{t:t+H} \mid o_t, e_t, \ell) = \begin{cases} \pi_{\mathrm{joint}}, & U_t \geq \tau_U \;\lor\; \Delta U_t \geq \tau_\Delta \\ \pi_{\mathrm{direct}}, & \text{otherwise} \end{cases}
$$

**含义**: 双重条件触发联合想象：绝对不确定度超阈值，或不确定度变化超阈值（即使绝对值未超）。

**符号说明**:
- $\tau_U$: 绝对不确定性阈值
- $\tau_\Delta$: 不确定性变化率阈值
- $\pi_{\mathrm{joint}}$: 联合想象策略路径
- $\pi_{\mathrm{direct}}$: 直接动作生成路径

### 公式10: [[Flow Matching|流匹配插值]]

$$
y^d_{t_d} = (1 - t_d)\, y^d + t_d\, \epsilon^d, \quad v^d = \epsilon^d - y^d
$$

**含义**: 流匹配训练时，在数据样本 $y^d$ 和噪声 $\epsilon^d$ 之间插值构造训练轨迹，速度场目标为 $v^d$。

**符号说明**:
- $d \in \{a, v\}$: 动作（action）或视频（video）模态
- $t_d$: 时间步（$t_d \in [0,1]$）
- $y^d$: 真实数据样本
- $\epsilon^d$: 高斯噪声

### 公式11: [[WAM|共享 WAM 前向传播]]

$$
(\hat{v}^a, \hat{v}^v) = F_\phi(y^a_{t_a}, y^v_{t_v}, t_a, t_v, c; M_g)
$$

**含义**: 共享 WAM 网络 $F_\phi$ 以动作噪声、视频噪声、时间步和条件信息为输入，输出速度场估计；$M_g$ 为路由决策的注意力掩码。

**符号说明**:
- $F_\phi$: 共享 WAM（Wan2.2-5B 为骨干）
- $c$: 条件信息（观测 + 语言指令）
- $M_g \in \{M_{\mathrm{direct}}, M_{\mathrm{joint}}\}$: 路由掩码

### 公式12: [[Flow Matching|WAM 总训练损失]]

$$
\mathcal{L}_{\mathrm{WAM}} = \lambda_a \mathcal{L}_a + \lambda_v \mathcal{L}_v
$$

**含义**: 动作流匹配损失与视频流匹配损失的加权组合。

**符号说明**:
- $\lambda_a, \lambda_v$: 动作与视频损失权重
- $\mathcal{L}_a, \mathcal{L}_v$: 各模态流匹配损失

### 公式13: [[Flow Matching|模态损失]]

$$
\mathcal{L}_d = \mathbb{E}\!\left[w(t_d)\,\mathrm{MSE}(\hat{v}^d, \epsilon^d - y^d)\right]
$$

**含义**: 对时间步加权 MSE，测量预测速度场与真实速度场之差，$w(t_d)$ 为时间步权重函数。

**符号说明**:
- $w(t_d)$: 时间步权重函数
- $\hat{v}^d$: 网络预测的速度场
- $\epsilon^d - y^d$: 真实速度场目标

---

## 关键图表

### Figure 1: 方法对比概览

> 注：论文 HTML 版中 Fig. 1 为矢量/SVG 格式，无可用外链图片。

**说明**: 对比 Fast-WAM（无想象）、Fast-WAM-Joint（全量想象）和 Bind-WAM（选择性想象）在成功率、想象比例、完成时间三个维度的表现，直观展示 Bind-WAM 的帕累托优势。

### Figure 2: Bind-WAM 整体架构

![Figure 2: Bind-WAM overview](https://arxiv.org/html/2611.15368v1/bind_wam_method.png)

**说明**: Bind-WAM 推理流水线。左侧绑定不确定性预测器输出 $U_t$，路由器据此选择直接路径或联合路径；共享 WAM（Wan2.2-5B）在两条路径下参数相同，仅注意力掩码不同；离线标注流水线（Qwen3-VL、SAM 3.1、DINOv3）仅在训练时使用。

### Figure 3: 离线绑定标注流水线示例

![Figure 3: Binding annotation pipeline](https://arxiv.org/html/2611.15368v1/binding_annotation_pipeline_example_labeled.png)

**说明**: 以炉灶和摩卡壶为例展示标注流程：Qwen3-VL 定位目标框，SAM 3.1 追踪掩码，DINOv3 计算特征相似度，Qwen3-VL 对时序帧投票验证；最终输出每帧每角色的不确定性标签（IoU、DINO 相似度、可见性）。

### Figure 4: 绑定不确定性预测示例

![Figure 4: Binding uncertainty predictions](https://arxiv.org/html/2611.15368v1/binding_uncertainty_examples_editable.png)

**说明**: 展示跨场景的低不确定性（物体清晰可见、指称明确）与高不确定性（遮挡、外观漂移、指称歧义）帧，验证预测器输出的可解释性。

### Figure 5: 单 Episode 不确定性曲线

![Figure 5: Episode uncertainty over time](https://arxiv.org/html/2611.15368v1/three_episode_overview.png)

**说明**: 三条 LIBERO 任务 episode 的不确定性随时间变化曲线。不确定性在抓取、物体转移等关键阶段出现峰值，与人工判断相符；路由器在峰值处触发联合想象。

### Figure 6: 真实世界执行演示

![Figure 6: Real-world demonstrations](https://arxiv.org/html/2611.15368v1/realworld_demonstrations.png)

**说明**: 双臂机器人 HEXFELLOW ARCHER-D6Y 在三项真实任务（Towel-Fold、Handover-Place、Shape-Alignment）上的成功执行关键帧，展示 Bind-WAM 在真实噪声和多相机环境下的鲁棒性。

### Figure 7: 想象比例阈值扫描

> 注：Fig. 7 为图表（threshold sweep plot），论文 HTML 中无独立外链图片。

**说明**: 在 LIBERO 上扫描联合去噪步骤比例（0% 至 100%）对应的成功率曲线。纯直接执行达 96.45%，在约 40% 联合步骤时达到峰值 98.6%，更高比例不再带来提升，说明选择性想象 ≈ 全量想象的关键帧收益。

### Figure 8: LIBERO 酒架任务关键帧对比

![Figure 8: Wine-rack task comparison](https://arxiv.org/html/2611.15368v1/imagination_evidence.png)

**说明**: 以 LIBERO wine-rack 任务为例的定性对比：Fast-WAM 因视觉不确定性在关键抓取时刻失败；Fast-WAM-Joint 过度执行想象导致撞倒瓶子；Bind-WAM 在绑定不确定时精准触发想象，成功抓取并放置。

### Table I: 绑定不确定性预测器性能

| 指标 | LIBERO | RoboTwin 2.0 |
|------|--------|--------------|
| 平衡准确率 | 87.59% | 96.75% |
| AUROC | 0.9653 | 0.9984 |
| AUPRC | 0.9917 | 0.9999 |
| Brier Score | 0.0504 | 0.0070 |
| 不确定召回（@0.5） | 97.32% | 99.27% |
| 确定召回（@0.5） | 77.86% | 94.23% |

**关键发现**: 预测器在两个基准上均达到高 AUROC（≥0.965），不确定类别召回率极高（≥97%），避免漏判关键帧。

### Table II: LIBERO 成功率对比

| 方法 | Spatial | Object | Goal | Long | 均值 | 时间（s） | 想象比例 |
|------|---------|--------|------|------|------|----------|----------|
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | – | – |
| π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 | – | – |
| π0.5 | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 | – | – |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 | – | – |
| Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 | – | – |
| AdaWAM | 98.0 | 99.6 | 97.1 | 99.1 | 98.5 | – | – |
| Fast-WAM† | 97.6 | 99.4 | 96.6 | 94.8 | 97.1 | 9.6 | 0.0% |
| Fast-WAM-Joint† | 99.8 | 98.8 | 98.6 | 96.4 | 98.4 | 15.7 | 100.0% |
| **Bind-WAM（Ours）** | **99.8** | **99.2** | **98.8** | **96.6** | **98.6** | **11.7** | **40.6%** |

**关键发现**: Bind-WAM 在 LIBERO 上以 98.6% 均值超越 Fast-WAM-Joint（98.4%），仅需 40.6% 的想象调用和 11.7 s 完成时间（vs 15.7 s）。

### Table III: RoboTwin 2.0 成功率对比

**Clean 环境：**

| 任务 | GO-1 | π0 | π0.5 | X-VLA | LingBot-VA | Motus | AdaWAM | Fast-WAM† | Joint† | **Ours** |
|------|------|----|-------|-------|------------|-------|--------|-----------|--------|---------|
| Hanging Mug | 0.0 | 14.0 | 18.0 | 23.0 | 40.0 | 38.0 | 59.0 | 57.0 | 72.0 | **71.0** |
| Pick Diverse Bottles | 61.0 | 69.0 | 81.0 | 58.0 | 89.0 | 90.0 | 87.0 | 82.0 | 86.0 | **87.0** |
| Put Object Cabinet | 60.0 | 85.0 | 80.0 | 46.0 | 80.0 | 88.0 | 96.0 | 94.0 | 95.0 | **95.0** |
| Rotate QR Code | 22.0 | 74.0 | 89.0 | 34.0 | 96.0 | 89.0 | 94.0 | 93.0 | 91.0 | **92.0** |
| Scan Object | 1.0 | 55.0 | 72.0 | 14.0 | 96.0 | 67.0 | 92.0 | 89.0 | 92.0 | **89.0** |
| Stack Bowls Three | 4.0 | 77.0 | 77.0 | 76.0 | 100.0 | 79.0 | 100.0 | 77.0 | 82.0 | **87.0** |
| Stamp Seal | 19.0 | 46.0 | 79.0 | 76.0 | 96.0 | 93.0 | 91.0 | 90.0 | 96.0 | **97.0** |
| **均值** | 23.9 | 60.0 | 70.9 | 46.7 | 85.3 | 77.7 | 88.4 | 83.1 | 87.7 | **88.3** |
| 完成时间（s） | – | – | – | – | – | – | – | 23.2 | 25.9 | **24.4** |
| 想象比例 | – | – | – | – | – | – | – | 0.0% | 100.0% | **44.5%** |

**Random 环境：**

| 任务 | GO-1 | π0 | π0.5 | X-VLA | LingBot-VA | Motus | AdaWAM | Fast-WAM† | Joint† | **Ours** |
|------|------|----|-------|-------|------------|-------|--------|-----------|--------|---------|
| Hanging Mug | 0.0 | 11.0 | 17.0 | 27.0 | 28.0 | 38.0 | 60.0 | 62.0 | 58.0 | **60.0** |
| Pick Diverse Bottles | 56.0 | 31.0 | 71.0 | 36.0 | 82.0 | 91.0 | 86.0 | 85.0 | 87.0 | **88.0** |
| Put Object Cabinet | 43.0 | 87.0 | 79.0 | 48.0 | 79.0 | 71.0 | 91.0 | 87.0 | 91.0 | **89.0** |
| Rotate QR Code | 9.0 | 70.0 | 87.0 | 33.0 | 91.0 | 73.0 | 94.0 | 88.0 | 90.0 | **93.0** |
| Scan Object | 2.0 | 42.0 | 65.0 | 36.0 | 91.0 | 66.0 | 91.0 | 90.0 | 93.0 | **93.0** |
| Stack Bowls Three | 7.0 | 75.0 | 71.0 | 86.0 | 98.0 | 87.0 | 98.0 | 84.0 | 88.0 | **87.0** |
| Stamp Seal | 13.0 | 33.0 | 55.0 | 82.0 | 97.0 | 92.0 | 86.0 | 94.0 | 96.0 | **99.0** |
| **均值** | 18.6 | 49.9 | 63.6 | 49.7 | 80.9 | 74.0 | 86.6 | 84.3 | 86.1 | **87.0** |
| 完成时间（s） | – | – | – | – | – | – | – | 24.0 | 26.4 | **25.3** |
| 想象比例 | – | – | – | – | – | – | – | 0.0% | 100.0% | **36.2%** |

**关键发现**: RoboTwin 2.0 Clean/Random 均值分别为 88.3%/87.0%，均优于 Fast-WAM-Joint（87.7%/86.1%），想象比例降至 44.5%/36.2%。

### Table IV: 真实世界双臂机器人成功率

| 任务 | Motus | Fast-WAM | Fast-WAM-Joint | **Bind-WAM** |
|------|-------|----------|----------------|-------------|
| Towel-Fold | 54.5 | 45.5 | 63.6 | **72.7** |
| Handover-Place | 63.6 | 54.5 | 72.7 | **72.7** |
| Shape-Alignment | 36.4 | 27.3 | 36.4 | **54.5** |
| **均值** | 51.5 | 42.4 | 57.6 | **66.6** |
| 完成时间（s） | 50.1 | 37.1 | 41.9 | **39.0** |
| 想象比例 | 100.0% | 0.0% | 100.0% | **28.4%** |

**关键发现**: 真实世界中 Bind-WAM 以 66.6% 均值大幅领先 Fast-WAM-Joint（57.6%），且完成时间（39.0 s）更接近 Fast-WAM（37.1 s），仅想象 28.4% 步骤。

### Table V: LIBERO 路由策略消融

| 路由策略 | 成功率 | 时间（s） | 想象比例 |
|----------|--------|----------|----------|
| 不确定性路由（Ours） | **98.60** | **11.71** | 40.58% |
| 随机路由（相似比例） | 97.35 | 14.58 | 40.89% |
| 均匀路由（相似比例） | 97.50 | 14.64 | 40.58% |

**关键发现**: 在想象比例相同的情况下，不确定性路由比随机/均匀路由成功率高 1.1–1.25 个百分点，且执行时间短约 20%，证明路由信号而非想象频率本身是关键。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] (Spatial/Object/Goal/Long) | 4 子任务套件 | 多场景桌面操作，标准化测试 | 仿真测试 |
| [[RoboTwin 2.0]] | 7 任务，Clean+Random | 细粒度物体多样性，随机化物体颜色/形状 | 仿真测试 |
| 真实 Bimanual | 3 任务，每任务 75 条示例 | HEXFELLOW ARCHER-D6Y 双臂，3 相机 | 真实世界测试 |

### 实现细节

- **Backbone**: [[Wan 2.2-5B]] 视频生成模型（Wan2.2-5B）作为共享 WAM 基础
- **预测器架构**: [[ViT]] 风格图像-指令交叉注意力网络（~6.65M 参数）
- **标注工具**: [[Qwen3-VL]]（32B）+ [[SAM3]]（3.1）+ [[DINOv3]]（特征相似度）
- **标注密度**: 每条示例轨迹采样 32 帧
- **损失权重**: BCE + Brier 双项，类别加权
- **重规划频率**: LIBERO 每 10 步重规划一次
- **真实测试**: 每任务 11 次 trials，每任务 75 条训练示例

### 可视化结果

- 不确定性预测器在遮挡、物体转移、外观漂移等关键时刻输出峰值，与人工判断高度一致（Fig. 4、5）；
- 真实世界中路由器能在 Towel-Fold 的布料形变、Handover-Place 的双臂交接时刻精准触发想象（Fig. 6）；
- wine-rack 对比（Fig. 8）直观展示选择性想象对抓取成功率的决定性作用。

---

## 批判性思考

### 优点

1. **可解释信号**: 绑定不确定性有清晰的物理含义（指称定位、可见性、特征漂移），路由决策可解释；
2. **计算效率**: 预测器轻量（6.65M），共享 WAM 无双份权重，真实世界仅想象 28.4% 步骤；
3. **强泛化性**: 离线标注在仿真上训练的预测器迁移至真实世界，仅靠 RGB 和语言输入即可工作；
4. **消融验证充分**: 路由策略消融（Table V）直接证明信号质量而非比例是关键。

### 局限性

1. **标注成本高**: 离线标注依赖 Qwen3-VL 32B + SAM 3.1 + DINOv3，每条轨迹 32 帧，扩展到大规模数据时计算成本显著；
2. **阈值超参敏感**: $\tau_U, \tau_\Delta$ 需在每个新场景/任务上调优，尚无自适应方法；
3. **真实任务覆盖有限**: 仅测试 3 个双臂任务，75 条示例/任务，泛化性有待扩展；
4. **路由反馈缺失**: 路由器不利用任务执行结果反馈，无法在线自适应调整。

### 潜在改进方向

1. 引入执行结果反馈（task-outcome feedback）对路由器进行在线微调；
2. 探索其他可解释不确定性信号（如深度不确定性、语言歧义度）；
3. 扩展至更多机器人平台和复杂接触任务（精细装配、柔性体操作）。

### 可复现性评估

- [ ] 代码开源（arXiv 页面未列出）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文中有完整描述）
- [x] 数据集可获取（LIBERO、RoboTwin 2.0 均公开）

---

## 关联笔记

### 基于

- [[Fast-WAM]]: 直接基础模型，Bind-WAM 在其双路径设计上添加路由
- [[WAM|世界动作模型（WAM）]]: 核心范式
- [[Flow Matching]]: 训练动作和视频生成的损失框架
- [[Wan 2.2-5B]]: 视频生成骨干网络

### 对比

- [[Fast-WAM]]: 无想象基线
- [[AdaWAM]]: 另一种自适应想象方案

### 方法相关

- [[Binding Uncertainty]]: 本文核心信号
- [[Language-Object Binding]]: 绑定不确定性的来源概念
- [[Selective Future Imagination]]: 本文解决的核心问题
- [[ViT]]: 预测器架构
- [[Qwen3-VL]]: 离线标注 VLM
- [[SAM3]]: 实例分割与追踪
- [[DINOv3]]: 特征相似度计算

### 硬件/数据相关

- [[LIBERO]]: 仿真测试基准
- [[RoboTwin 2.0]]: 仿真测试基准
- [[Action Chunking]]: 动作输出格式

---

## 速查卡片

> [!summary] Bind-WAM
> - **核心**: 语言-目标绑定不确定性路由，选择性触发 WAM 未来想象
> - **方法**: 离线标注（Qwen3-VL+SAM3+DINOv3）→ 轻量 ViT 预测器 → 双条件路由（$U_t$ 或 $\Delta U_t$ 超阈值触发联合路径）
> - **结果**: LIBERO 98.6%（vs Joint 98.4%），想象 40.6%，11.7 s；真实 66.6%（vs Joint 57.6%），想象 28.4%
> - **代码**: 未公开

---

*笔记创建时间: 2026-11-14*
