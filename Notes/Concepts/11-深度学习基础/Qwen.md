---
type: concept
aliases: [Qwen, Qwen-VL, Qwen3.5-VL, 通义千问]
---

# Qwen

## 定义

Qwen（通义千问）是阿里巴巴开发的大型语言模型系列，Qwen-VL 为其多模态视觉语言版本，支持图像和视频理解任务，可进行视觉定位、目标识别和场景描述等。

## 核心要点

1. Qwen3.5-VL 是较新版本，具备强大的视觉 grounding 和目标定位能力
2. 在机器人领域常用于从图像/视频中生成语义分割 mask（VLM 引导的分割）
3. 可结合 SAM-2 等分割模型，通过语言指令定位目标物体
4. GEM-4D 的 AIDS 中用于当追踪突然失败时重新 grounding 末端执行器位置

## 代表工作

- [[GEM-4D]]: 在 AIDS 系统中，当帧间变化 $\Delta s_t < -\delta$ 时调用 Qwen3.5-VL 重新定位末端执行器

## 相关概念

- [[SAM]]
- [[视觉语言模型]]
