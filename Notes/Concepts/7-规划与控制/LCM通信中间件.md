---
type: concept
aliases: [LCM, Lightweight Communications and Marshalling, LCM通信, LCM消息中间件]
---

# LCM 通信中间件

## 定义

LCM（Lightweight Communications and Marshalling）是一个面向实时系统的轻量级进程间发布/订阅通信库，支持 UDP 多播，提供自动类型序列化，常用于机器人系统的多进程低延迟通信。

## 核心要点

1. **发布/订阅模型**: 生产者无需等待消费者，消费者随时读取最新消息，天然支持异步通信
2. **非阻塞通道**: 发送方不等待接收方应答，单向缓冲最新状态
3. **自动序列化**: 通过 LCM 类型定义文件（.lcm）自动生成跨语言绑定（C++/Python/Java）
4. **低延迟**: 基于 UDP 多播，适合 kHz 级实时控制回路

## 在机器人控制中的应用

DASH 使用三个 LCM 通道连接并发进程：
- 环境进程 → VLM 服务器（发送观测）
- VLM 服务器 → 动作服务器（发送语义上下文）
- 动作服务器 → 环境进程（发送动作指令）

## 代表工作

- [[DASH]]: 使用 LCM 实现 VLM 进程（System 2）和动作头进程（System 1）的异步通信

## 相关概念

- [[Asynchronous Inference]]
- [[Decoupled Policy]]
