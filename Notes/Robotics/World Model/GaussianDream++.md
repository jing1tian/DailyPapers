---
title: "GaussianDream++: Efficient 3D Gaussian World Modeling for Robotic Manipulation"
method_name: "GaussianDream++"
authors: [Yuqing Jiang, Zijian Zhang, Weitao Zhou, Jiawei Wang, Junjie He, Lei Yang, Haifang Qing, Si Liu, Ding Zhao, Ping Luo, Haibao Yu]
year: 2026
venue: arXiv
tags: [3d-gaussian-world-model, vla, robot-manipulation, world-model, gaussian-splatting, policy-native, efficient-deployment]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.25659
created: 2026-08-28
---

# 论文笔记：GaussianDream++: Efficient 3D Gaussian World Modeling for Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tuojing Intelligence, Chinese Academy of Sciences, Tsinghua University, HKUST, CMU, University of Hong Kong |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[GaussianDream]], [[pi0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.25659) / [Code](https://github.com/TuojingAI/GaussianDream) |

---

## 一句话总结

> GaussianDream++ 将 [[3DGS|3D Gaussian Splatting]] 世界监督压缩进 20 个策略原生 World Token，训练时通过轻量 World Representation Head 提供当前重建和未来预测的稠密几何监督，推理时丢弃解码头和渲染器，仅保留 token 驱动动作生成，实现 1.15× 延迟开销的高效部署。

---

## 核心贡献

1. **策略原生世界表示（Policy-Native World Representation）**: 将 [[World State Tokens|世界状态 Token]]（16 个）和 [[World Prediction Tokens|世界预测 Token]]（4 个）直接嵌入 [[PaliGemma]] 骨干，替代 [[GaussianDream]] 中基于 [[VGGT]]/[[TGE]] 的 1024-token 密集前缀，使 Gaussian 监督通过原生注意力内化为动作表征，而无需额外的几何前向路径。
2. **耦合当前-未来 Gaussian 预测（[[耦合未来预测|Coupled Future Prediction]]）**: 基于共享 Gaussian 原语进行当前世界重建与短视野未来预测，未来帧仅预测中心位移，尺度、旋转、透明度和外观全部继承自当前世界，结合[[静态-动态分解|静态-动态门控]]消除背景漂移，最大化预测容量用于交互区域运动。
3. **训练-推理路径解耦（Efficient Deployment）**: 训练时使用完整的 World Representation Head、Gaussian 渲染器和辅助监督分支；推理时完全移除，仅保留 PaliGemma 骨干 + 20 个 World Token + Action Expert，延迟仅 330 ms（vs. [[GaussianDream]] 531 ms），开销仅 +44 ms（1.15×）。

---

## 问题背景

### 要解决的问题

当前 [[VLA（视觉-语言-动作模型）|VLA]] 策略在复杂操作任务中面临三大缺陷：
1. **几何欠定（Spatial Underspecification）**: 预训练 [[VLA（视觉-语言-动作模型）|VLM]] 在 2D 像素空间运行，缺乏对 3D 度量几何的显式建模；
2. **动态缺失（Temporal Blindness）**: 稀疏的动作模仿目标无法捕捉场景演化和物理约束；
3. **推理效率瓶颈**: 已有几何增强方法（如 [[GaussianDream]]）引入的密集前缀和运行时几何路径导致推理延迟过高。

### 现有方法的局限

- **几何增强 VLA**（[[GeoVLA]]、Spatial Forcing）：仅关注当前帧空间结构，缺少时序预测；
- **预测策略/世界模型**（[[ManiGaussian]]、Fast-WAM）：在 RGB / 视频隐空间建模未来，视觉一致性不等同于度量物理变化；
- **[[GaussianDream]]（直接前身）**: 使用 [[VGGT]] / [[TGE]] 构建的 1024-token 前缀携带纠缠的状态、动态和动作信息，需要专用运行时几何路径，推理延迟 531 ms。

### 本文的动机

训练时的 Gaussian 监督（当前重建 + 未来预测）能有效为 VLA 提供 3D 结构约束，但无需在推理时保留完整世界模型。若能将这一监督信号**内化**为策略骨干的紧凑 Token 表示，就能同时兼顾几何精度和部署效率。

---

## 方法详解

### 模型架构

GaussianDream++ 采用**策略原生 Gaussian 监督**架构：

- **输入**: 语言指令 $\ell$ + 多视角观测 $o_t$ + 机器人状态 $r_t$
- **Backbone**: [[PaliGemma]]（处理视觉-语言 token）+ [[π₀.₅]] Action Expert（流匹配动作解码）
- **核心模块**: [[World State Tokens]]（16 个，编码当前场景）+ [[World Prediction Tokens]]（4 个，编码短视野演化）
- **训练辅助**: World Representation Head → [[3DGS|Gaussian 渲染器]]（推理时移除）
- **输出**: [[Action Chunking|动作块]] $A_t$（流匹配 Action Expert 生成）

```
训练路径:
[o_t, ℓ, Z^S, Z^P] → PaliGemma → [H^VL, H^S, H^P]
                                         ↓              ↓
                               Action Expert     World Representation Head
                                    ↓                    ↓
                                   A_t           G_t, Ĝ_{t+h} → 渲染监督

推理路径（移除 World Representation Head 和渲染器）:
[o_t, ℓ, Z^S, Z^P] → PaliGemma → [H^VL, H^S, H^P] → Action Expert → A_t
```

### 核心模块

#### 模块 1: Policy-Native World Representation（策略原生世界表示）

**设计动机**: 利用 [[PaliGemma]] 的原生注意力机制，让 World Token 直接参与视觉-语言上下文化，消除额外投影层。

**具体实现**:
- [[World State Tokens]] $Z^S \in \mathbb{R}^{N_s \times d}$（$N_s = 16$）：4×4 空间排布，附有经过标定的空间嵌入和射线嵌入，维持粗粒度空间组织；
- [[World Prediction Tokens]] $Z^P \in \mathbb{R}^{N_p \times d}$（$N_p = 4$）：与预测视野（horizon）槽关联，每个槽对应一个预测步；
- PaliGemma 通过标准 Transformer 将 $Z^S, Z^P$ 与视觉 token 和语言 token **联合上下文化**，输出 $(H_t^{VL}, H_t^S, H_t^P)$；
- Action Expert 直接以**完整的世界 Token 增强前缀**为条件，无中间投影器。

#### 模块 2: World Representation Head（世界表示解码头）

**设计动机**: 在训练时将 World Token 隐状态解码为显式 Gaussian 基元，提供稠密的几何-外观-运动监督；推理时完全丢弃，零部署开销。

**具体实现**:
- 解码器 $D_\text{state}$：将 $H_t^S$ 的 Grid 展开为稠密 Gaussian 特征场 $F_t^G$；
- [[几何-外观分解|几何-外观分离]]:
  - 几何分支 $H_\text{geo}$：预测度量深度 $\bar{D}_t$、几何属性 $\Theta_t^\text{geo}$（尺度、旋转、透明度）和几何特征 $F_t^\text{geo}$；
  - 外观分支 $H_\text{app}$：在 stop-gradient 几何特征上融合当前图像特征 $f_t^\text{img}$ 得到外观属性 $\Theta_t^\text{app}$（球谐系数），防止光度捷径污染度量几何结构；
- Gaussian 中心由多视角度量反投影 $\mathcal{U}(\bar{D}_t; K_t, T_t)$ 确定，锚定在物理 3D 空间；
- 可微分 [[3DGS|Gaussian Splatting]] 渲染 RGB、度量深度和覆盖率 $(\hat{I}_t, \hat{D}_t, \hat{A}_t)$，提供密集监督。

#### 模块 3: Coupled Future Prediction（耦合未来预测）

**设计动机**: 在保留当前世界几何模板的同时，仅预测短视野内的度量中心位移，避免重复生成静态场景元素，将预测容量集中于交互区域。

**具体实现**:
- 对每个预测视野 $h$，融合 $F_{t,i}^G$（当前 Gaussian 特征）、$H_{t,h}^P$（预测 Token 隐状态）和视野嵌入 $e_h$，预测中心位移 $\Delta\mu_i^h$；
- [[静态-动态分解|静态-动态门控]]：门控系数 $m_i^h \in [0,1]$ 对背景/静态区域置零，仅允许动态区域发生位移；
- 未来 Gaussian $G_{t+h}$ 的中心由 $\mu_i^{t+h} = \mu_i^t + m_i^h \Delta\mu_i^h$ 更新，尺度、旋转、透明度、外观**全部继承**自 $G_t$，确保当前-未来一致性；
- 使用视野专用相机参数渲染，防止视点变化被吸收进物理运动预测；
- **静态一致性监督** $\mathcal{L}_\text{stat}$：惩罚持续区域的残余运动，抑制背景漂移。

---

## 关键公式

### 公式 1: [[World State Tokens|世界 Token 维度定义]]

$$
Z^{S} \in \mathbb{R}^{N_s \times d}, \quad Z^{P} \in \mathbb{R}^{N_p \times d}
$$

**含义**: 定义 $N_s=16$ 个世界状态 Token 和 $N_p=4$ 个世界预测 Token 的形状，两者为可学习参数，直接嵌入 PaliGemma 前缀。

**符号说明**:
- $Z^S$: 世界状态 Token，编码当前帧物理场景结构
- $Z^P$: 世界预测 Token，编码短视野演化意图
- $N_s = 16$: 状态 Token 数量（4×4 空间排布）
- $N_p = 4$: 预测 Token 数量（对应多个预测视野槽）
- $d$: Token 隐维度（与 PaliGemma 骨干一致）

### 公式 2: [[PaliGemma|PaliGemma 联合上下文化]]

$$
(H_t^{VL}, H_t^S, H_t^P) = F_\theta(o_t, \ell;\, Z^S, Z^P)
$$

**含义**: PaliGemma 对观测、语言和 World Token 进行联合上下文化，输出视觉-语言隐状态和世界 Token 隐状态。

**符号说明**:
- $F_\theta$: PaliGemma Transformer 参数化函数
- $o_t$: 当前时刻多视角观测
- $\ell$: 语言任务指令
- $H_t^{VL}$: 视觉-语言隐状态
- $H_t^S, H_t^P$: 被上下文化后的状态/预测 Token 隐状态

### 公式 3: [[Action Chunking|动作生成]]

$$
A_t = \pi_\phi(r_t;\, H_t^{VL}, H_t^S, H_t^P)
$$

**含义**: 流匹配 Action Expert 以机器人状态和世界 Token 增强前缀为条件生成动作块，世界 Token 通过原生注意力直接约束动作生成，无需中间投影。

**符号说明**:
- $\pi_\phi$: 流匹配 Action Expert
- $r_t$: 机器人本体状态（关节位置等）
- $A_t$: 预测动作块（action chunk）

### 公式 4: [[World Representation Head|世界表示解码]]

$$
(G_t,\, \{\widehat{G}_{t+h}\}_{h \in \mathcal{H}}) = W_\psi(H_t^S, H_t^P)
$$

**含义**: 训练时专用解码头将 World Token 隐状态解码为当前 Gaussian 世界 $G_t$ 和一系列未来 Gaussian 预测 $\widehat{G}_{t+h}$，推理时完全丢弃。

**符号说明**:
- $W_\psi$: World Representation Head（训练专用）
- $G_t$: 当前 Gaussian 世界（$N_G$ 个基元的集合）
- $\mathcal{H}$: 预测视野集合（多个时间步）

### 公式 5: [[3DGS|Gaussian 特征场展开]]

$$
F_t^G = D_{\text{state}}(\text{Grid}(H_t^S))
$$

**含义**: 将 $H_t^S$ 展开为空间网格后，通过多层解码器扩展为稠密 Gaussian 特征场，为后续几何-外观分解提供中间表示。

**符号说明**:
- $D_\text{state}$: 特征场解码器（多层扩展网络）
- $\text{Grid}(\cdot)$: 将 $N_s$ 个 token 重排为 4×4 空间网格
- $F_t^G$: 稠密 Gaussian 特征场

### 公式 6: [[几何-外观分解|几何-外观分离]]

$$
\bar{D}_t,\, \Theta_t^{\text{geo}},\, F_t^{\text{geo}} = H_{\text{geo}}(F_t^G)
$$

$$
\Theta_t^{\text{app}} = H_{\text{app}}(\text{sg}[F_t^{\text{geo}}],\, f_t^{\text{img}})
$$

**含义**: 几何分支预测度量深度和 Gaussian 几何属性；外观分支在 stop-gradient 几何特征上融合图像特征预测球谐外观，防止光度捷径干扰度量几何学习。

**符号说明**:
- $\bar{D}_t$: 预测的度量深度图
- $\Theta_t^\text{geo}$: 几何属性（尺度 $s$、旋转 $q$、透明度 $\alpha$）
- $F_t^\text{geo}$: 几何特征（用于外观分支输入）
- $\text{sg}[\cdot]$: stop-gradient 算子（阻断外观梯度反传几何分支）
- $f_t^\text{img}$: 当前图像特征（可选融合）
- $\Theta_t^\text{app}$: 外观属性（球谐系数 $c$）

### 公式 7: [[3DGS|Gaussian 中心反投影]]

$$
\{\mu_i^t\}_{i=1}^{N_G} = \mathcal{U}(\bar{D}_t;\, K_t, T_t)
$$

**含义**: 利用标定相机内参 $K_t$ 和外参 $T_t$ 将预测深度图反投影到 3D 空间，确定每个 Gaussian 基元的初始中心位置，实现度量 3D 锚定。

**符号说明**:
- $\mathcal{U}$: 多视角度量反投影函数
- $K_t, T_t$: 相机内参矩阵和外参矩阵（多视角标定值）
- $N_G$: Gaussian 基元数量

### 公式 8: 当前世界 Gaussian 集合

$$
G_t = \{(\mu_i^t,\, s_i^t,\, q_i^t,\, \alpha_i^t,\, c_i^t)\}_{i=1}^{N_G}
$$

**含义**: 当前世界由 $N_G$ 个显式 Gaussian 基元组成，每个基元包含位置、尺度、旋转、透明度和球谐外观系数五元组。

**符号说明**:
- $\mu_i^t$: 第 $i$ 个基元的 3D 中心位置
- $s_i^t$: 各向异性尺度（3D 椭球轴长）
- $q_i^t$: 旋转四元数
- $\alpha_i^t$: 不透明度
- $c_i^t$: 球谐外观系数

### 公式 9: [[3DGS|可微分 Gaussian 渲染]]

$$
(\hat{I}_t,\, \hat{D}_t,\, \hat{A}_t) = \mathcal{R}(G_t;\, K_t, T_t)
$$

**含义**: 对当前 Gaussian 世界进行可微分 Splatting 渲染，同时输出 RGB 图像、度量深度图和累积覆盖率（alpha map），支持三种模态的密集监督。

**符号说明**:
- $\mathcal{R}$: 可微分 Gaussian Splatting 渲染器
- $\hat{I}_t$: 渲染 RGB 图像
- $\hat{D}_t$: 渲染度量深度图
- $\hat{A}_t$: 渲染累积覆盖率（alpha map）

### 公式 10: [[耦合未来预测|当前世界监督损失]]

$$
\mathcal{L}_{\text{cur}} = \lambda_{\text{rgb}}\mathcal{L}_{\text{rgb}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}} + \lambda_\alpha \mathcal{L}_\alpha + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}
$$

**含义**: 联合优化渲染 RGB（光度 + SSIM）、度量深度和覆盖率，并加入 Gaussian 基元正则项，约束当前世界的几何精度与视觉一致性。

**符号说明**:
- $\mathcal{L}_\text{rgb}$: RGB 损失（光度 + 结构相似性）
- $\mathcal{L}_\text{depth}$: 度量深度损失（仅在有效监督区域计算）
- $\mathcal{L}_\alpha$: 覆盖率损失（可见性约束）
- $\mathcal{L}_\text{reg}$: Gaussian 基元正则项
- $\lambda_*$: 各项权重系数

### 公式 11: [[World Prediction Tokens|未来中心位移预测]]

$$
\Delta \mu_i^h = H_\Delta(F_{t,i}^G,\, H_{t,h}^P,\, e_h)
$$

**含义**: 对每个 Gaussian 基元预测视野 $h$ 下的 3D 中心位移，融合当前几何特征、预测 Token 隐状态和视野嵌入，实现基元对齐的时序建模。

**符号说明**:
- $H_\Delta$: 位移预测网络
- $F_{t,i}^G$: 第 $i$ 个基元的当前几何特征
- $H_{t,h}^P$: 视野 $h$ 对应的预测 Token 隐状态
- $e_h$: 可学习视野嵌入
- $\Delta\mu_i^h$: 预测的 3D 中心位移

### 公式 12: [[静态-动态分解|静态-动态门控位移]]

$$
\mu_i^{t+h} = \mu_i^t + m_i^h \cdot \Delta \mu_i^h
$$

**含义**: 静态-动态门控系数 $m_i^h$ 对背景/持续区域置零，抑制无关漂移；仅允许动态交互区域发生位移，将预测容量集中于运动相关基元。

**符号说明**:
- $m_i^h \in [0,1]$: 静态-动态门控系数（软掩码）
- $\mu_i^t$: 当前世界中基元 $i$ 的中心
- $\mu_i^{t+h}$: 视野 $h$ 下预测的未来中心

### 公式 13: 未来 Gaussian 世界（耦合继承）

$$
G_{t+h} = \{(\mu_i^{t+h},\, s_i^t,\, q_i^t,\, \alpha_i^t,\, c_i^t)\}_{i=1}^{N_G}
$$

**含义**: 未来 Gaussian 仅更新中心位置，其余属性全部继承自 $G_t$。**耦合**设计确保当前-未来一致性，避免重复生成静态场景，预测容量专注于物理运动。

**符号说明**:
- $\mu_i^{t+h}$: 更新后的未来中心（式 12）
- $s_i^t, q_i^t, \alpha_i^t, c_i^t$: 从 $G_t$ 继承，保持不变

### 公式 14: 未来世界可微分渲染

$$
(\hat{I}_{t+h},\, \hat{D}_{t+h},\, \hat{A}_{t+h}) = \mathcal{R}(G_{t+h};\, K_{t+h}, T_{t+h})
$$

**含义**: 使用视野专用相机参数对未来 Gaussian 进行渲染，防止视角变化被混入物理运动预测，提供多模态未来监督目标。

**符号说明**:
- $K_{t+h}, T_{t+h}$: 视野 $h$ 对应的相机内外参（与当前帧独立）

### 公式 15: [[耦合未来预测|未来预测损失]]

$$
\mathcal{L}_{\text{fut}} = \sum_{h \in \mathcal{H}} w_h \left(\lambda_{\text{rgb}}\mathcal{L}_{\text{rgb}}^h + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}^h + \lambda_{\text{flow}}\mathcal{L}_{\text{flow}}^h + \lambda_{\text{stat}}\mathcal{L}_{\text{stat}}^h\right)
$$

**含义**: 对多个预测视野加权求和，每个视野包含 RGB、深度、度量 3D 场景流和静态一致性四项监督，综合约束运动的光度、几何和流场一致性。

**符号说明**:
- $w_h$: 视野 $h$ 的时间权重
- $\mathcal{L}_\text{flow}^h$: 度量 3D [[场景流]] 损失（约束运动物理一致性）
- $\mathcal{L}_\text{stat}^h$: 静态一致性损失（惩罚持续区域的残余运动）

### 公式 16: [[耦合未来预测|完整训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda_{\text{cur}}\mathcal{L}_{\text{cur}} + \lambda_{\text{fut}}\mathcal{L}_{\text{fut}}
$$

**含义**: 联合优化动作生成、当前世界重建和未来预测三个目标，Gaussian 监督防止 Token 表示坍塌为纯动作特征，维持物理结构约束。

**符号说明**:
- $\mathcal{L}_\text{act}$: 流匹配动作损失（Action Expert 主目标）
- $\lambda_\text{cur}, \lambda_\text{fut}$: 当前/未来 Gaussian 监督权重

---

## 关键图表

### Figure 1: 机器人策略范式对比

![Figure 1](https://arxiv.org/html/2608.25659v1/assets/comparation.png)

**说明**: 横向对比四类策略范式。**Reactive VLA** 仅隐式视觉表示直接映射动作；**Geometry-enhanced** 增加当前帧空间建模；**Predictive/World-Model** 引入未来时序监督；**GaussianDream++**（右下）结合当前重建和未来预测于紧凑 [[World State Tokens|策略原生 Gaussian 表示]]，训练时提供结构化 3D 监督，推理时无需在线世界解码。

### Figure 2: GaussianDream++ 整体框架

![Figure 2](https://arxiv.org/html/2608.25659v1/assets/worlddream_framework.png)

**说明**: 完整的训练-推理双路径架构。**[[World State Tokens]]**（蓝色，16 个）和 **[[World Prediction Tokens]]**（橙色，4 个）插入 [[PaliGemma]] 前缀与视觉/语言 token 联合上下文化，结果直接条件化 Action Expert。训练时，World Representation Head 从相同 Token 隐状态解码为当前 Gaussian $G_t$ 和未来预测 $\hat{G}_{t+h}$，进行稠密渲染监督；未来观测仅作为监督目标，**从不进入**策略前向传播。推理时，解码头、渲染器和辅助分支全部移除，仅保留 20 个 World Token 条件化动作。

### Figure 3: 真实世界任务与分布迁移

![Figure 3](https://arxiv.org/html/2608.25659v1/assets/Realworld.png)

**说明**: 双臂机器人平台的两个任务设置及三种测试条件。**Bowl-Proximity**（左）：将白碗精准移至目标位置旁侧（测试精细对象相对定位）；**Eggplant-to-Pink-Plate**（右）：抓取茄子放入粉色盘（测试长视野操作与语义干扰物辨别）。每个任务分别在 Standard（标准排布）、Layout Shift（物体配置改变）、Camera Shift（全局相机迁移）三种条件下进行 20 次试验。

### Figure 4: 定性分析——当前重建与未来预测

![Figure 4](https://arxiv.org/html/2608.25659v1/assets/vis.png)

**说明**: 可视化 Ground Truth、**当前重建**（World State Token 解码）和**未来预测**（World State + World Prediction Token 联合解码）三列。当前重建恢复主要机器人配置、工作空间几何和物体布局（精细纹理较模糊，符合代理目标而非光照重建）；未来预测保留相同粗粒度场景结构，同时表征短视野几何变化。耦合设计使当前世界作为持续模板，World Prediction Token 仅编码残差演化。

### Table 1a: LIBERO 标准测试集结果

| Method | Spatial | Object | Goal | Long | Average |
|--------|---------|--------|------|------|---------|
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| π₀.₅ | 97.8 | 98.8 | 97.6 | 92.4 | 96.7 |
| GeoPredict | 98.0 | 98.2 | 95.7 | 94.0 | 96.5 |
| QDepth-VLA | 97.6 | 96.6 | 95.2 | 90.0 | 94.9 |
| LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| GeoVLA | 98.4 | 99.0 | 96.6 | 96.6 | 97.7 |
| VLA-4D | 97.9 | 98.6 | 97.8 | 94.8 | 97.4 |
| 3D-CAVLA | 98.2 | 99.8 | 98.2 | 96.1 | 98.1 |
| Spatial Forcing | 98.6 | 98.4 | 98.2 | 95.4 | 97.6 |
| π₀.₅ reproduced | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| GaussianDream | 99.0 | 99.6 | 99.0 | 96.0 | 98.4 |
| **GaussianDream++** | **99.2** | **99.6** | **99.0** | **96.6** | **98.6** |

**关键发现**: GaussianDream++ 以 98.6% 平均成功率达到所有方法最高，较 π₀.₅ reproduced 提升 1.7 个百分点，较 GaussianDream 提升 0.2 个百分点（主要增益在 Long 任务 +0.6）。

### Table 1b: [[LIBERO-Plus]] 零样本鲁棒性测试

| Method | Camera | Robot | Lang. | Light | BG | Noise | Layout | Overall |
|--------|--------|-------|-------|-------|-----|-------|--------|---------|
| OpenVLA-OFT | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| RIPT-VLA | 55.2 | 31.2 | 77.6 | 88.4 | 91.6 | 73.5 | 74.2 | 68.4 |
| StarVLA-α (Spec.) | 48.7 | 63.4 | 86.8 | 95.8 | 94.6 | 75.0 | 80.2 | 77.8 |
| StarVLA-α (Gen.) | 52.5 | 64.3 | 86.2 | 97.8 | 98.1 | 80.2 | 79.1 | 79.7 |
| Cosmos Policy | 75.8 | 63.3 | 81.7 | 96.5 | 88.9 | 92.7 | 82.2 | 82.2 |
| LaMP | 64.5 | 69.6 | 88.2 | 95.3 | 97.4 | 76.9 | 73.8 | 79.3 |
| π₀.₅ (ACoT protocol) | 75.8 | 79.4 | 83.3 | 95.5 | 95.0 | 89.6 | 87.0 | 85.7 |
| ACoT-VLA | 72.6 | 82.6 | 87.5 | 97.7 | 96.5 | 87.8 | 88.1 | 86.6 |
| π₀.₅ reproduced | 73.2 | 77.9 | 84.3 | 96.3 | 95.5 | 89.9 | 87.7 | 85.5 |
| GaussianDream | 77.3 | 72.1 | 86.9 | 98.9 | 98.5 | 93.6 | 88.4 | 87.0 |
| **GaussianDream++** | **80.1** | **73.0** | **87.4** | **99.1** | **98.8** | **94.2** | **90.0** | **87.8** |

**关键发现**: GaussianDream++ 在 Overall 达到 87.8%，各方法最高。几何敏感扰动下最显著：Camera +6.9 pp（vs. π₀.₅）、Layout +2.3 pp；Noise +4.3 pp（结构约束抑制视觉降噪干扰）。Robot shift 仍低于 π₀.₅，表明最大收益集中于视角、布局和视觉扰动类别。

### Table 2: 真实机器人操作结果

| Method | Standard | Layout | Camera | Overall |
|--------|----------|--------|--------|---------|
| **Bowl-Proximity** | | | | |
| π₀.₅ reproduced | 35.0% (7/20) | 20.0% (4/20) | 20.0% (4/20) | 25.0% (15/60) |
| GaussianDream++ | 55.0% (11/20) | 50.0% (10/20) | 35.0% (7/20) | 46.7% (28/60) |
| **Eggplant-to-Pink-Plate** | | | | |
| π₀.₅ reproduced | 45.0% (9/20) | 30.0% (6/20) | 25.0% (5/20) | 33.3% (20/60) |
| GaussianDream++ | 70.0% (14/20) | 55.0% (11/20) | 50.0% (10/20) | 58.3% (35/60) |
| **All Tasks** | | | | |
| π₀.₅ reproduced | 40.0% (16/40) | 25.0% (10/40) | 22.5% (9/40) | 29.2% (35/120) |
| **GaussianDream++** | **62.5%** (25/40) | **52.5%** (21/40) | **42.5%** (17/40) | **52.5%** (63/120) |

**关键发现**: 真实机器人综合成功率从 29.2% → 52.5%（+23.3 pp），Layout 迁移从 25.0% → 52.5%（双倍），Camera 迁移从 22.5% → 42.5%，验证 Gaussian 表示在闭环执行中的空间稳定性。

### Table 3: 推理路径与延迟对比

| Method | Runtime VGGT/TGE | World Tokens | Runtime Head | Latency |
|--------|------------------|--------------|--------------|---------|
| π₀.₅ reproduced | No | 0 | No | 286 ms |
| **GaussianDream++** | **No** | **20** | **No** | **330 ms** |
| GaussianDream† | Yes | 1024 | No | 531 ms |

**关键发现**: GaussianDream++ 仅增加 +44 ms（1.15×），远低于 GaussianDream 的 531 ms（1.86×）。20 个 World Token 的上下文化是唯一额外计算，无在线 Gaussian 构建、渲染或未来推演。

### Table 4: 消融实验（LIBERO-Plus Overall）

| 变体 | LIBERO Avg. | LIBERO-Plus Overall | 分析 |
|------|-------------|---------------------|------|
| π₀.₅ reproduced | 96.9 | 85.5 | 基准线 |
| World Tokens w/o Gaussian Supervision | 97.5 | 86.3 | Token 容量本身不足 |
| Current World only | 97.9 | 86.9 | 基础空间锚定 |
| Current + Future, uncoupled | 98.1 | 87.2 | 时序贡献 |
| Coupled w/o Static Consistency | 98.4 | 87.5 | 静态一致性必要 |
| Full w/o RGB Rendering | 98.4 | 87.4 | RGB 主要影响光照/背景 |
| Full w/o Metric Depth | 98.0 | 86.9 | 最强单项约束 |
| Full w/o Alpha Supervision | 98.3 | 87.3 | 影响可见性敏感扰动 |
| Full w/o Metric 3D Flow | 98.1 | 87.2 | 弱化运动监督 |
| **Full GaussianDream++** | **98.6** | **87.8** | 互补约束综合最优 |

**关键发现**:
1. **度量深度是最强单项约束**（去除后降 0.9 pp），是 Camera/Layout 鲁棒性的主要来源；
2. **耦合优于独立**（+0.3 pp over uncoupled）：共享 Gaussian 基元避免当前-未来状态不一致；
3. **静态一致性损失不可缺**（+0.3 pp over w/o static）：显式约束背景区域零运动；
4. **Gaussian 监督必要**（w/o supervision +0.8 pp over reproduced，full model +1.5 pp）：Token 容量本身不提供几何约束能力。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| [[LIBERO]] | 4 任务套件（Spatial/Object/Goal/Long），仿真操作 | 标准训练 + 测试 |
| [[LIBERO-Plus]] | 7 类受控分布迁移（相机/机器人/语言/光照/背景/噪声/布局） | 零样本鲁棒性评估 |
| 双臂真实机器人数据 | Bowl-Proximity + Eggplant-to-Pink-Plate，20 trials × 3 条件 | 真实部署验证 |

### 实现细节

- **Backbone**: [[PaliGemma]]（视觉-语言）+ 流匹配 Action Expert
- **World Token 数量**: $N_s = 16$（状态）+ $N_p = 4$（预测）= 共 20 个
- **训练策略**: 联合优化 $\mathcal{L}_\text{act} + \lambda_\text{cur}\mathcal{L}_\text{cur} + \lambda_\text{fut}\mathcal{L}_\text{fut}$
- **对比基线**: 复现 [[pi0.5|π₀.₅]]（相同协议）作为严格对照，避免评估协议差异
- **推理硬件**: 批量大小 1，端到端延迟 330 ms（含完整动作块预测）

### 可视化结果

- 当前重建恢复主要机器人配置、工作空间几何、物体布局，但精细外观较模糊（与目标吻合：代理重建服务于策略表征，非光照一致性重建）
- 未来预测与当前重建保持一致的粗粒度场景结构，展现适度的短视野几何变化
- [[静态-动态分解|静态-动态门控]]使背景区域接近零位移，机器人-物体交互区域位移明显，体现耦合设计的信息分配效率

---

## 批判性思考

### 优点

1. **极致训练-推理分离**: 所有 Gaussian 计算仅在训练时存在，推理时仅 20 个额外 Token，延迟开销最小（+15%）
2. **共享基元耦合设计**: 未来预测复用当前 Gaussian 模板，强制当前-未来时序一致性，避免属性重复学习
3. **多模态互补监督**: RGB + 度量深度 + 3D 场景流 + 静态一致性四类信号从不同角度约束同一物理场景，消融验证各项贡献

### 局限性

1. **Robot Shift 未改善**: LIBERO-Plus Robot 维度低于复现的 π₀.₅（73.0 vs. 77.9），暗示 Gaussian 几何监督对机器人形态变化的泛化贡献有限
2. **Camera-Conditional 预测局限**: World Prediction Token 调制于观测和任务上下文，学习短视野期望演化而非反事实的动作条件仿真，限制了用于在线规划的扩展潜力
3. **小 Token 数量的代表性上限**: 16 个状态 Token 压缩完整场景，精细外观和小物体细节难以保留

### 潜在改进方向

1. **自适应 Token 分配**: 根据任务复杂度动态调整 World Token 数量，平衡效率与表达能力
2. **动作条件预测**: 将候选动作引入 World Prediction Token 调制，支持在线 model-predictive 控制
3. **跨任务分布迁移**: 将 Gaussian 监督迁移至无标定数据的通用操作场景，减少对精确相机参数的依赖

### 可复现性评估

- [x] 代码开源（GitHub: TuojingAI/GaussianDream）
- [ ] 预训练模型（未确认发布状态）
- [x] 训练细节完整（消融验证各超参贡献）
- [x] 数据集可获取（LIBERO 公开）

---

## 关联笔记

### 基于

- [[GaussianDream]]: 直接前身，建立了 Gaussian 监督 VLA 范式（VGGT+TGE 1024-token 前缀）
- [[PaliGemma]]: 视觉-语言骨干（GaussianDream++ 嵌入 Token 于其中）
- [[3DGS]]: 3D Gaussian Splatting 核心表示和可微分渲染引擎

### 对比

- [[pi0.5|π₀.₅]]: 主要基线（相同骨干，无世界建模）
- [[ManiGaussian]]: 同类 Gaussian 世界模型 VLA，但面向动态多任务控制

### 方法相关

- [[World State Tokens]]: 核心：当前场景 16 个策略原生 Token
- [[World Prediction Tokens]]: 核心：短视野演化 4 个预测 Token
- [[耦合未来预测]]: 核心：共享基元的耦合当前-未来预测设计
- [[静态-动态分解]]: 关键组件：门控位移仅作用于动态区域
- [[几何-外观分解]]: 关键组件：止梯度分离几何与外观分支
- [[场景流]]: 监督信号之一：度量 3D 场景流约束运动物理一致性
- [[VGGT]]: GaussianDream 使用的前身模块（GaussianDream++ 已移除）
- [[TGE]]: GaussianDream 使用的时序演化模块（GaussianDream++ 已移除）

### 数据集相关

- [[LIBERO]]: 主要训练和测试 benchmark
- [[LIBERO-Plus]]: 零样本分布迁移鲁棒性评估

---

## 速查卡片

> [!summary] GaussianDream++
> - **核心**: 用 20 个策略原生 World Token 内化 3D Gaussian 世界监督，训练提供几何约束，推理零开销
> - **方法**: World State Tokens（16）+ World Prediction Tokens（4）嵌入 PaliGemma，训练时 World Representation Head 解码为 Gaussian 重建+预测，推理时丢弃
> - **结果**: LIBERO 98.6%、LIBERO-Plus 87.8%（最优）、真实机器人 52.5%（vs. 29.2%），推理仅 +44ms
> - **代码**: [TuojingAI/GaussianDream](https://github.com/TuojingAI/GaussianDream)

---

*笔记创建时间: 2026-08-28*
