---
title: "GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation"
method_name: "GAF"
authors: [Ying Chai, Litao Deng, Ruizhi Shao, Jiajun Zhang, Kangchen Lv, Liangjun Xing, Xiang Li, Hongwen Zhang, Yebin Liu]
year: 2026
venue: arXiv
tags: [gaussian-action-field, world-model, robotic-manipulation, 3d-gaussian-splatting, diffusion-policy, 4d-representation, dynamic-scene-modeling]
zotero_collection: 3-Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2506.14135v5
created: 2026-05-26
---

# 论文笔记：GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, BAAI |
| 日期 | May 2026 |
| 项目主页 | [ChaiYing1.github.io/projects/GAF](https://ChaiYing1.github.io/projects/GAF/) |
| 对比基线 | [[ManiGaussian]], [[扩散策略\|Diffusion Policy]], [[Act3D]] |
| 链接 | [arXiv](https://arxiv.org/abs/2506.14135) |

---

## 一句话总结

> GAF 将 [[3D Gaussian Splatting]] 扩展为含运动属性的 4D 表示，统一建模当前场景重建、未来状态预测与初始动作估计，并通过动作-视觉对齐去噪框架提升操作精度。

---

## 核心贡献

1. **V-4D-A 范式**: 提出 Vision-to-4D-to-Action 新范式，超越静态 V-3D-A 方法，将时序动态显式建模纳入机器人操作框架。
2. **Gaussian Action Field 表示**: 为每个高斯原语附加可学习运动位移 $\Delta\mu$，从单一前向传播中同时输出场景重建、未来预测和初始动作估计。
3. **动作-视觉对齐去噪框架**: 将估计的夹爪位姿渲染叠加到多视角 RGB 图像上，提供可视化条件引导扩散式动作精化，显著提升操作成功率。

---

## 问题背景

### 要解决的问题

机器人操作策略需要对动态场景进行精确的三维时序理解，以推断物体运动和生成可执行动作。现有方法要么只处理静态几何，要么直接从像素到动作映射，缺乏对场景动态的显式建模。

### 现有方法的局限

- **V-A（Vision-to-Action）**: 直接从 RGB 预测动作，缺乏三维推理能力，空间泛化性差（代表：[[扩散策略|Diffusion Policy]]、RT-2）。
- **V-3D-A（Vision-to-3D-to-Action）**: 引入静态 3D 表示（如点云、Voxel），但忽略时序动态，无法预测物体运动（代表：[[Act3D]]、PerAct、[[ManiGaussian]]）。

### 本文的动机

[[3D Gaussian Splatting]] 提供了高质量的显式场景表示，若为每个高斯原语附加运动属性 $\Delta\mu$，就能同时建模当前状态和未来状态。运动场本身即隐含了物体如何响应操作动作的信息，从而可以直接用于动作估计。

---

## 方法详解

### 模型架构

GAF 采用 **Vision Transformer + 多头预测** 架构，实现端到端的 4D 场景理解：

- **输入**: 稀疏多视角 RGB 图像 $\{I_v^t\}_{v=1}^{V}$ + 相机内外参 $\{k_v^t\}_{v=1}^{V}$
- **Backbone**: [[Vision Transformer|ViT]]（初始化权重来自 [[MASt3R]]），提取混合场景特征
- **三个预测头**:
  - [[Gaussian Center Head]]: 预测 3D 高斯位置 $\mu$
  - [[Gaussian Param Head]]: 预测外观参数 $\{c, \sigma, r, s\}$
  - [[Motion Prediction Head]]: 预测每点位移向量 $\Delta\mu^{t \to t+\Delta t}$
- **动作精化**: [[动作-视觉对齐去噪框架]] 通过 [[DDIM]] 扩散迭代精化动作
- **输出**: 可执行末端执行器动作序列

### 核心模块

#### 模块1：GAF 4D 场景表示

**设计动机**: 利用 [[3D Gaussian Splatting]] 的显式可微渲染特性，同时建模几何结构和时序动态，避免隐式潜空间表示带来的信息损失。

**具体实现**:
- 每个高斯原语参数化为 $(\mu, \Delta\mu, f)$，其中 $f = \{c, \sigma, r, s\}$ 为颜色、不透明度、旋转、尺度
- 当前帧 Gaussians：$\{\mu^t, f^t\}$，通过 [[Alpha-Blending 渲染]] 重建当前场景
- 未来帧 Gaussians：$\{\mu^t + \Delta\mu^{t \to t+\Delta t}, f^t\}$，预测 $\Delta t$ 步后场景状态
- ViT 提取跨视角注意力特征后，分别由三个解码头并行输出

#### 模块2：ICP 初始动作估计

**设计动机**: 利用 [[Iterative Closest Point|ICP]] 点云配准，从预测的运动场直接计算夹爪的刚体变换，作为动作估计的初值，减少扩散去噪所需迭代步数。

**具体实现**:
- 选取夹爪对应的 Gaussian 点集，利用 ICP 求解最优刚体变换 $T^{t \to t+\Delta t} \in SE(3)$
- 该变换直接对应末端执行器的位移和旋转动作
- 作为 init action 传入后续去噪框架作为初始估计

#### 模块3：动作-视觉对齐去噪框架

**设计动机**: 初始动作估计可能存在误差，通过将夹爪位姿渲染到多视角图像上，提供人类可理解的视觉反馈作为扩散条件，显著提升动作精度。

**具体实现**:
- 将估计的初始夹爪位姿 $T_g^w$ 叠加相对动作 $a$，通过相机参数渲染到每个视角图像
- 得到 "Actionable Multiview RGB Guidance"（可行动多视角 RGB 引导）
- 将渲染叠加图作为条件，与 [[扩散策略|Diffusion Policy]] 范式结合，进行有限步 [[DDIM]] 去噪（仅 3 步）
- 损失函数约束去噪方向 $D$、末端执行器噪声 $\varepsilon$ 和夹爪开闭状态 $g$

---

## 关键公式

### 公式1：[[Gaussian Action Field|GAF 函数映射]]

$$
\mathcal{F}_\Theta: \{g(x), t\} \mapsto \{\mu, \Delta\mu, f\}
$$

**含义**: GAF 网络将场景几何输入 $g(x)$ 和时间步 $t$ 映射到三元组输出，同时预测当前几何、运动位移和外观参数。

**符号说明**:
- $\mathcal{F}_\Theta$: GAF 神经网络，参数为 $\Theta$
- $g(x)$: 场景几何表示（由多视角图像提取）
- $\mu \in \mathbb{R}^3$: 3D 高斯中心位置
- $\Delta\mu \in \mathbb{R}^3$: 时间 $\Delta t$ 内的运动位移向量
- $f = \{c, \sigma, r, s\}$: 外观参数（颜色、不透明度、旋转、尺度）

### 公式2：[[Vision Transformer|高斯重建头]]

$$
\mathcal{H}_{Gauss}\left(\text{ViT}\left(\{I_v^t, k_v^t\}\right)_{v=1}^{V}\right) = \{\mu_j^t, c_j^t, \sigma_j^t, r_j^t, s_j^t\}_{j=1}^{V \times H \times W}
$$

**含义**: ViT 处理所有视角图像及相机参数，高斯头解码输出每像素对应的 3D 高斯原语完整参数。

**符号说明**:
- $V$: 视角数量
- $H, W$: 图像高度和宽度
- $\mu_j^t$: 第 $j$ 个高斯在时刻 $t$ 的 3D 位置
- $k_v^t$: 第 $v$ 个相机的内外参数

### 公式3：[[Motion Prediction Head|运动预测头]]

$$
h_{motion}\left(\text{ViT}\left(\{I_v^t, k_v^t\}\right)_{v=1}^{V}\right) = \{\Delta\mu_j^{t \to t+\Delta t}\}
$$

**含义**: 与高斯重建头共享 ViT 特征，运动头单独输出每个高斯原语在未来 $\Delta t$ 步的运动位移。

**符号说明**:
- $\Delta\mu_j^{t \to t+\Delta t}$: 第 $j$ 个高斯原语从 $t$ 到 $t+\Delta t$ 的位移向量
- $\Delta t$: 预测的未来时间间隔

### 公式4：[[Alpha-Blending 渲染|体渲染颜色累积]]

$$
C(\mathbf{p}) = \sum_{i=1}^{N} \alpha_i c_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

**含义**: 标准 3DGS 体渲染公式，沿射线 $\mathbf{p}$ 对所有高斯原语进行 alpha-blending，得到最终渲染像素颜色。

**符号说明**:
- $C(\mathbf{p})$: 像素射线 $\mathbf{p}$ 的渲染颜色
- $N$: 参与渲染的高斯数量
- $\alpha_i$: 第 $i$ 个高斯在 2D 投影上的密度（由协方差矩阵 $\Sigma_i$ 决定）
- $c_i$: 第 $i$ 个高斯的颜色

### 公式5：[[Gaussian Action Field|GAF 训练损失]]

$$
\mathcal{L}_{GAF} = \mathcal{L}_{LPIPS}^t + \mathcal{L}_{MSE}^t + \mathcal{L}_{LPIPS}^{t+\Delta t} + \mathcal{L}_{MSE}^{t+\Delta t}
$$

**含义**: 同时监督当前帧和未来帧的渲染质量，用感知损失 LPIPS 和像素级 MSE 损失联合约束几何和外观重建。

**符号说明**:
- $\mathcal{L}_{LPIPS}^t$: 当前帧感知相似度损失（[[LPIPS]]）
- $\mathcal{L}_{MSE}^t$: 当前帧像素均方误差
- $\mathcal{L}_{LPIPS}^{t+\Delta t}$: 未来帧感知损失，监督运动预测
- $\mathcal{L}_{MSE}^{t+\Delta t}$: 未来帧像素均方误差，监督运动预测

### 公式6：[[Iterative Closest Point|ICP 初始动作计算]]

$$
T^{t \to t+\Delta t} = \arg\min \sum_{k \in \text{gripper}} \|T(\mu_k) - (\mu + \Delta\mu)_k\|^2
$$

**含义**: 对夹爪区域高斯点集，求解使当前位置变换后与预测未来位置最接近的刚体变换，直接得到末端执行器的动作。

**符号说明**:
- $T^{t \to t+\Delta t} \in SE(3)$: 末端执行器的刚体变换（位移 + 旋转）
- $\mu_k$: 夹爪区域第 $k$ 个高斯原语当前位置
- $(\mu + \Delta\mu)_k$: 该高斯原语的预测未来位置

### 公式7：[[动作-视觉对齐去噪框架|动作视觉渲染]]

$$
R^c = \text{Render}(T_{w2c} \times T_g^w \times a, K^c)
$$

**含义**: 将世界坐标系下的夹爪位姿与相对动作变换叠加，通过相机内参投影渲染到第 $c$ 个视角的图像上，生成可视化条件。

**符号说明**:
- $R^c$: 第 $c$ 个视角下渲染的夹爪叠加图像
- $T_{w2c}$: 世界到相机坐标系的外参变换
- $T_g^w$: 世界坐标系下夹爪位姿
- $a$: 相对动作变换
- $K^c$: 第 $c$ 个相机的内参矩阵

### 公式8：[[扩散策略|动作精化损失]]

$$
\mathcal{L}_{refine} = L_1(D, D^{gt}) + L_1(\varepsilon, \varepsilon^{gt}) + \text{BCE}(g, g^{gt})
$$

**含义**: 动作精化阶段的多任务损失，分别约束扩散去噪方向、末端执行器噪声预测和夹爪开闭状态分类。

**符号说明**:
- $D$: 预测的夹爪去噪方向，$D^{gt}$: 对应真值
- $\varepsilon$: 末端执行器动作上的预测噪声，$\varepsilon^{gt}$: 真值噪声
- $g$: 夹爪开闭预测（二值），$g^{gt}$: 真值
- $\text{BCE}$: 二值交叉熵损失

---

## 关键图表

### Figure 1: V-A / V-3D-A / V-4D-A 范式对比

![Figure 1](https://arxiv.org/html/2506.14135v5/x1.png)

**说明**: 展示三种机器人操作范式的演进。V-A 范式直接从视觉预测动作，缺乏 3D 推理；V-3D-A 引入静态 3D 表示但忽略时序动态；GAF 提出的 V-4D-A 范式通过 [[3D Gaussian Splatting]] 扩展的 4D 表示，统一建模当前重建、未来预测和动作估计。

### Figure 2: GAF 重建架构概览

![Figure 2](https://arxiv.org/html/2506.14135v5/x2.png)

**说明**: 稀疏多视角图像输入 [[Vision Transformer|ViT]]，提取混合场景特征后由三个解码头并行输出：Gaussian Center Head 预测高斯位置、Gaussian Param Head 预测外观参数、Motion Prediction Head 输出每点位移向量 $\Delta\mu$。三个头共享 ViT 特征，实现高效联合预测。

### Figure 3: 操作执行流水线

![Figure 3](https://arxiv.org/html/2506.14135v5/x3.png)

**说明**: GAF 输出的 4D 场景表示（当前高斯 + 未来高斯）作为条件，输入[[动作-视觉对齐去噪框架]]。初始动作经 [[Iterative Closest Point|ICP]] 估计，再通过 3 步 [[DDIM]] 去噪精化，生成可执行的末端执行器轨迹。整个流程迭代执行直至任务完成。

### Figure 4: 当前帧重建与未来帧预测对比

![Figure 4](https://arxiv.org/html/2506.14135v5/x4.png)

**说明**: 从新视角展示当前场景重建（Ours/Now）和未来场景预测（Ours/Future）质量，与 [[ManiGaussian]] 的 baseline 对比。GAF 在 PSNR 上提升 +11.54 dB，SSIM 提升 +0.39，LPIPS 降低 -0.56，视觉质量显著更优。

### Figure 5: 去噪框架消融实验

![Figure 5](https://arxiv.org/html/2506.14135v5/x5.png)

**说明**: 上图为无动作去噪精化的失败案例（夹爪位置偏差导致操作失败），下图为加入[[动作-视觉对齐去噪框架]]后的成功案例。证明视觉对齐条件对动作精度的关键作用。

### Figure 6: 空间泛化能力测试

![Figure 6](https://arxiv.org/html/2506.14135v5/x6.png)

**说明**: 在 RLBench 的 22 个任务上，用 20 条演示（紫色星号标注采样位置）训练后，测试工作空间网格采样点的泛化能力。红色代表失败，蓝色代表成功。GAF 在边界和角落位置的泛化显著优于 [[扩散策略|Diffusion Policy]] baseline。

### Figure 7: 真实世界实验装置

![Figure 7](https://arxiv.org/html/2506.14135v5/x7.png)

**说明**: 实验平台为 [[Franka Panda]] 机械臂，配备两个静态相机和一个腕部相机，提供多视角视觉输入。在 55 个真实世界任务上评估。

### Figure 8: 真实世界 GAF 渲染结果

![Figure 8](https://arxiv.org/html/2506.14135v5/x8.png)

**说明**: 展示 GAF 在真实场景的当前高斯渲染（Current Gaussian）和未来高斯渲染（Future Gaussian），验证 4D 表示在真实硬件上的有效性。

### Table I: RLBench 操作任务成功率对比（%）

| Method | Toilet Seat Down | Open Grill | Close Grill | Close Fridge | Phone On Base | Lift Lid Up | Close Microwave | Push Button | Close Laptop | Avg. |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [[扩散策略\|Diffusion Policy]] | 39 | 20 | 36 | 16 | 23 | 81 | 62 | 61 | 64 | 44.7 |
| [[Act3D]] | 60 | 22 | 41 | 47 | 26 | 91 | 73 | 60 | 58 | 53.1 |
| [[ManiGaussian]] | 34 | 24 | 38 | 41 | 30 | 97 | 67 | 63 | 57 | 50.1 |
| Ours w/o GAF | 57 | 16 | 49 | 27 | 31 | 94 | 79 | 58 | 51 | 51.3 |
| **Ours (GAF)** | **71** | **26** | **55** | **42** | **35** | **100** | **85** | **61** | **69** | **60.4** |

**关键发现**: GAF 以 60.4% 平均成功率大幅领先，比 Act3D（+7.3%）、ManiGaussian（+10.3%）、Diffusion Policy（+15.7%）均有显著提升。Lift Lid Up 任务达到 100% 成功率。

### Table II: 新视角合成质量对比（当前帧 + 未来帧）

| 方法 | Close Microwave PSNR↑ | SSIM↑ | LPIPS↓ | Toilet Seat PSNR↑ | SSIM↑ | LPIPS↓ | Lift Lid PSNR↑ | SSIM↑ | LPIPS↓ | Avg PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ManiGaussian/Now | 16.43 | 0.375 | 0.763 | 16.56 | 0.398 | 0.681 | 16.15 | 0.414 | 0.622 | 16.38 | 0.396 | 0.688 |
| ManiGaussian/Future | 16.14 | 0.357 | 0.790 | 15.80 | 0.369 | 0.716 | 15.37 | 0.397 | 0.657 | 15.77 | 0.374 | 0.721 |
| **Ours/Now** | **27.10** | **0.798** | **0.129** | **28.17** | **0.778** | **0.135** | **28.49** | **0.771** | **0.129** | **27.92** | **0.782** | **0.131** |
| **Ours/Future** | **24.59** | **0.765** | **0.149** | **27.30** | **0.766** | **0.146** | **27.02** | **0.748** | **0.141** | **26.30** | **0.760** | **0.145** |

**关键发现**: GAF 当前帧重建 PSNR 平均提升 +11.54 dB，未来帧预测也大幅领先，证明 4D 表示的质量优势。

### Table III: 多任务评估成功率（%，括号为 delta）

| Method | Toilet Seat Down | Close Microwave | Lift Lid Up | Close Laptop | Average |
|--------|:---:|:---:|:---:|:---:|:---:|
| [[扩散策略\|Diffusion Policy]] | 55 (+16) | 50 (-12) | 39 (-42) | 18 (-46) | 40.5 (-21.0) |
| **Ours (GAF)** | **59 (-12)** | **85 (+0)** | **57 (-43)** | **79 (+10)** | **70.0 (-11.25)** |

**关键发现**: 多任务训练（单网络同时学 4 个任务）时 GAF 平均下降 11.25%，而 Diffusion Policy 下降 21%，GAF 的多任务扩展性更好。

### Table IV: 真实世界任务成功率

| Task | Success Rate |
|------|:---:|
| Push Button | 10/10 (100%) |
| Close Door | 8/10 (80%) |
| Open Door | 7/10 (70%) |
| Pick Cup | 7/10 (70%) |
| Place Apple | 6/10 (60%) |

**关键发现**: 在 5 类真实任务共 55 次测试中，GAF 表现稳定，Push Button 达到满分，Place Apple 等精细任务也有 60% 以上成功率。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RLBench]] | 9 / 22 / 4 个任务变体 | 多任务仿真，支持多视角输入 | 仿真训练/测试 |
| Real-World (自采) | 55 次试验 | Franka Panda，真实物理场景 | 真实部署测试 |

### 实现细节

- **Backbone**: [[Vision Transformer|ViT]]，初始化权重来自 [[MASt3R]]
- **视角数**: $V = 3$（仿真），$V = 3$（真实：2 静态 + 1 腕部）
- **训练轮数**: 80,000 次迭代，约 24 小时
- **推理**: 每 8 步预测 < 0.3 秒，去噪仅 3 步 [[DDIM]] 迭代
- **硬件**: NVIDIA RTX A800
- **演示数据**: 每任务约 100 条演示（仿真）；真实世界使用少量真实演示

### 可视化结果

- 当前帧高斯渲染质量远超 ManiGaussian（视觉锐利，纹理清晰）
- 未来帧预测能准确捕捉物体运动方向和位移量
- 空间泛化测试显示 GAF 在工作空间边缘和角落也有较高成功率

---

## 批判性思考

### 优点

1. **统一表示**: 单一前向传播同时完成重建、预测和动作估计，推理高效（<0.3s/8步）
2. **显式时序建模**: 相比隐式潜空间世界模型，GAF 的显式 4D 高斯表示可解释性强，且可以直接用于可视化验证
3. **渲染质量突破**: PSNR 提升 +11.54 dB 是一个极大幅度的提升，证明架构设计（MASt3R 权重 + 多头设计）的有效性
4. **真实部署验证**: 不仅限于仿真，在真实 Franka 机械臂上取得了 60-100% 的成功率

### 局限性

1. **缺乏语义/任务理解**: 作者承认系统无任务级语义理解，无法处理复杂的任务分解和语言指令
2. **抓取阶段鲁棒性不足**: 真实世界主要失败发生在抓取阶段，缺少力触觉反馈
3. **多任务性能下降**: 多任务训练相比单任务仍有 11.25% 的性能下降，网络容量可能需要扩展
4. **未建模外力**: 系统未建模外部力/接触力，对依赖力反馈的精细操作有局限

### 潜在改进方向

1. 集成语言模型实现语言指令条件的任务规划
2. 引入力/触觉传感器信号改善接触建模
3. 探索 Hierarchical Gaussian（层次高斯）支持长视野多步骤任务

### 可复现性评估

- [ ] 代码开源（项目主页存在但未见公开代码）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文描述了 80k 迭代、MASt3R 初始化等细节）
- [x] 数据集可获取（RLBench 公开可用）

---

## 关联笔记

### 基于

- [[3D Gaussian Splatting]]: GAF 的核心表示基础，扩展了运动属性
- [[MASt3R]]: ViT 权重初始化来源，提供强大的多视角特征提取
- [[扩散策略|Diffusion Policy]]: 动作精化阶段采用的扩散框架

### 对比

- [[ManiGaussian]]: 同样基于 3DGS 的操作方法，但只建模静态 3D，GAF 扩展为 4D
- [[Act3D]]: 基于点云的 3D 操作方法，V-3D-A 范式代表
- [[扩散策略|Diffusion Policy]]: V-A 范式代表，缺乏 3D 推理

### 方法相关

- [[3D Gaussian Splatting]]: 核心场景表示
- [[Iterative Closest Point]]: 初始动作估计方法
- [[DDIM]]: 动作精化去噪采样器
- [[Vision Transformer]]: 特征提取主干
- [[Alpha-Blending 渲染]]: 3DGS 可微渲染机制

### 硬件/数据相关

- [[RLBench]]: 主要评估 benchmark
- [[Franka Panda]]: 真实部署机器人平台

---

## 速查卡片

> [!summary] GAF: Gaussian Action Field
> - **核心**: 将 3DGS 扩展为含运动属性的 4D 表示，统一建模场景重建、未来预测与动作估计
> - **方法**: V-4D-A 范式；ViT 三头预测 + ICP 初始动作 + 视觉对齐扩散精化
> - **结果**: RLBench 60.4% 成功率（+7.3% vs Act3D），重建 PSNR +11.54 dB
> - **代码**: 未公开（项目主页：https://ChaiYing1.github.io/projects/GAF/）

---

*笔记创建时间: 2026-05-26*
