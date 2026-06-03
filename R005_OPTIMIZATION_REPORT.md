# R005 (Jetson Orin NX) 推理优化方案报告

> **日期**: 2026-05-12
> **设备**: R005 / Jetson Orin NX (L4T R36.4.3, CUDA 12.6, sm_87, 15.6GB 统一内存)
> **模型**: Qwen3.5-2B (Q4_K_M), RoboBrain2.5-4B (Q4_K_M)
> **推理引擎**: llama.cpp (GGML_CUDA)

---

## 摘要

从最初 CPU-only 方案（预期 3-10s）到最终 CUDA 优化方案（实测 ~9s/复杂任务），经历了 2 次重大架构修正和 3 轮优化迭代。最终在 R005 上实现 GPU 全层卸载，推理速度从 ~4 tok/s 提升到 **~35 tok/s**，端到端延迟从 21s 降到 **9s**。

---

## 方案迭代历程

### 方案 A — 原始方案（CPU only, NPU 误判）

**时间**: 2026-05-10 ~ 2026-05-11 上午

**假设**: R005 是独立 NPU 设备，需通过国讯芯微 Compass SDK 加速。

**技术路线**:
- Dockerfile FROM `arm64v8/ubuntu`（纯 ARM CPU）
- BLIS 库做 ARM CPU 矩阵加速
- llama.cpp CPU-only 编译，无 CUDA
- 等待 Compass SDK 做 NPU 推理

**问题**:
- 纯 CPU 推理极慢（预期 3-10s/短问，80s/长程）
- Compass SDK 厂商未回复
- R005 实际上是 Jetson Orin NX（NVIDIA CUDA GPU）

**结果**: ❌ 整条路线废弃

---

### 方案 B — 初版 CUDA 部署（FORCE_MMQ）

**时间**: 2026-05-11 下午

**修正**: 发现 R005 = Jetson Orin NX（CUDA 设备），与 4090 同一技术栈。

**技术路线**:
- Dockerfile FROM `nvidia/cuda:12.4.0-devel-ubuntu22.04`（多架构 arm64）
- 编译参数：`GGML_CUDA=ON` + `F16` + `FA_ALL_QUANTS` + **`FORCE_MMQ=ON`** + `NATIVE=ON`
- Docker 多阶段构建（builder + runtime）
- GPU 全层卸载（25/25, 37/37 layers）

**尝试的改动**:
1. `arm64v8/ubuntu` → `nvidia/cuda:12.4.0` 镜像，BLIS 移除
2. Docker build 加 `--network host` 解决 DNS 问题
3. GitHub clone 从 Docker build 内移到主机预下载（网络超时问题）
4. 运行时镜像 `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
5. 容器 `network_mode: "host"` 解决 bridge 握手问题

**问题**:
- **FORCE_MMQ 在 Ampere 上性能倒挂** → 实测 ~4 tok/s（500 tok 耗时 125s）
- CUDA 版本不匹配：主机 12.6 vs 容器 12.4 → 二进制不可运行
- nvcr.io L4T 镜像 401 Unauthorized

**结果**: ❌ Docker 方案废弃，改主机原生 nohup 运行

---

### 方案 C — 去 MMQ 重构（基线）

**时间**: 2026-05-11 晚 ~ 2026-05-12

**决策**: 放弃 FORCE_MMQ，重新编译 llama.cpp。

**编译参数（验证后最优）**:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DLLAMA_BUILD_SERVER=ON
```

**关键参数说明**:
- **移除 FORCE_MMQ**: Ampere 架构不适用，比默认 CUDA 内核慢 5-8x
- **指定 CMAKE_CUDA_ARCHITECTURES="87"**: 默认 arch list 不含 sm_87，不指定则运行时 JIT 编译崩溃
- **GGML_NATIVE=ON**: 针对主机 CPU 优化微内核

### 方案 D — Docker CUDA 方案修复（最终运行方案）

**时间**: 2026-05-12

**背景**: 之前方案 B 的 Docker 方案因 CUDA 版本不匹配和 cuBLAS 不兼容问题废弃。方案 C 用主机原生 nohup 运行，但无容器隔离。最终决定修好 Docker。

**核心问题**: `nvidia/cuda` 镜像内置桌面版 cuBLAS，与 Jetson L4T Tegra-optimized cuBLAS 不兼容，运行时 `cublasCreate_v2` 崩溃。

**修复方案**:
1. Dockerfile FROM `ubuntu:22.04`（裸镜像，无 CUDA 库）
2. 主机预编译 llama-server（去 MMQ + sm_87 指定），Docker 只 COPY 二进制 + `*.so*`
3. 运行时 volumes 挂载主机 CUDA 库：`/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib:ro`
4. `LD_LIBRARY_PATH=/app:/usr/local/cuda/lib`
5. `.dockerignore` 排除无用文件，构建上下文从 666MB 降到合理大小

**最终 Dockerfile 结构**:
```dockerfile
FROM ubuntu:22.04
COPY llama-server /app/
COPY *.so* /app/
RUN ldconfig /app
EXPOSE 8081
ENTRYPOINT ["/app/llama-server"]
```

**验证结果**: 容器启动成功，GPU 全层卸载（24/25, 37/37 layers），推理正常。
**docker-compose 最终形态**: 3 个 service，共用同一份 `qwen35.Dockerfile` 镜像（仅模型文件不同），`network_mode: "host"` 统一网络。

---

## 性能对比

### Qwen3.5-2B 推理速度

| 指标 | FORCE_MMQ (方案B) | 去 MMQ (方案C) | 提升倍数 |
|------|-------------------|---------------|---------|
| 短问答 tok/s | ~4 | **22.5** | **~5.6x** |
| 任务规划 tok/s | ~4 | **31.6** | **~7.9x** |
| 长文本 tok/s | ~4 | **34.7** | **~8.7x** |
| 复杂任务端到端 | 21s / 733 tok | **9s / 384 tok** | **~2.3x** |

> 注：端到端优化包含提示词精简（输入 532→200 tok，输出 733→384 tok）。

### RoboBrain2.5-4B

> 待基准测试完成补充。

---

## 优化手段汇总

### ✅ 有效

| 优化手段 | 模块 | 效果 | 说明 |
|---------|------|------|------|
| 移除 FORCE_MMQ | 编译 | **5-8x** | MMQ 内核在 Ampere 上调度开销大，适合 Ada 不适合 Jetson |
| 指定 CMAKE_CUDA_ARCHITECTURES="87" | 编译 | 稳定性 | 默认 arch list 不含 sm_87，JIT 编译会崩 |
| 裸 ubuntu:22.04 + 主机 CUDA 挂载 | Docker | **可用性** | nvidia/cuda 镜像的桌面版 cuBLAS 与 L4T 驱动不兼容，裸镜像 + host mount 绕过 |
| 指定 CMAKE_CUDA_ARCHITECTURES="87" | 编译 | **稳定性** | 默认 arch list 不含 sm_87，不指定则 JIT 编译崩溃 |
| .dockerignore | Docker | **构建速度** | 清理 .so symlink，构建上下文从 666MB→合理大小 |
| 0.8B 轻量模型栈 | 架构 | **场景覆盖** | 三级推理体系：0.8B 简单问答 / 2B 视觉理解 / 4B 长程推理 |
| 提示词模板精简 | Runtime | **输入 -60%** | SCHEDULER 从 30 行压缩到 13 行 |
| max_tokens 减半 (768→384) | Runtime | **输出 -50%** | 控制生成长度 |
| batch-size 512 (非 4096) | Runtime | 稳定性 | Jetson 显存有限，大 batch 无收益 |
| network_mode: "host" | Docker | 通信正常 | bridge 网络有握手问题 |
| --network host 构建 | Docker | DNS 正常 | 边缘设备 DNS 配置问题 |

### ❌ 无效或有害

| 尝试 | 原因 |
|------|------|
| FORCE_MMQ=ON | Ampere 架构不兼容，比默认慢 5-8x |
| BLIS CPU 加速 | CUDA 路径下不相关 |
| nvidia/cuda 镜像 (12.4/12.6) | 桌面版 cuBLAS 不兼容 Jetson L4T，cublasCreate_v2 崩溃 |
| 多阶段 Docker 构建 | builder 阶段编译的二进制含桌面版 cuBLAS 引用，无法在 L4T 运行时使用 |
| Docker 内 git clone | 网络不稳定，超时频繁 |
| nvcr.io L4T 镜像 | 需要 NGC 登录（401 Unauthorized） |
| batch-size 4096 | Jetson 显存带宽有限，512 更优 |
| no-mmap | Jetson 统一内存模型不适用 |
| threads 调优 (4 vs 6 vs 8) | cpu核数实测对性能影响很小，已锁 4 核留给系统 |
### ⏳ 待验证

| 尝试 | 状态 |
|------|------|
| TensorRT 视觉 encoder | 脚本已建，待 Jetson 上执行 |
| threads 调优 (4 vs 6 vs 8) | 已锁 4 核留给系统 |
| Compass SDK NPU | 等待厂商回复 |

---

## 提示词模板优化（Phase 0.7）

为配合 R005 推理管线，对 RoboAgent 的提示词模板做结构化优化。

### 优化目标

- **降低端到端延迟**: R005 生成速度 ~55 tok/s，模板过长直接拖慢任务执行
- **结构化任务分解**: 机器人具身任务需拆解为可执行的原子技能序列
- **HRI 场景处理**: 任务可能涉及人机交互，需动态决定社交步骤

### 优化手段

| 手段 | 原大小 | 优化后 | 压缩率 |
|------|--------|--------|--------|
| SCHEDULER | ~2500 tok | ~460 tok | **82%** |
| EXECUTION_PLANNER | ~1200 tok | ~232 tok | **81%** |
| TEMPORAL_ANALYST | ~650 tok | ~286 tok | **56%** |

### 核心设计 — HRI 优先分解

```
任务输入 → HRI判断(是/否) → 
  是: 接近→交互→操作→社交收尾(APPROACH/SPEAK/REPORT/HANDOVER/BOW)
  否: 直接操作执行
```

**技能库 5 大分类**:
- 移动: NAVIGATE/APPROACH/FOLLOW/LEAD
- 操作: GRASP/PICK/PLACE/RELEASE/PUSH/PULL/ROTATE/PRESS
- 环境: OPEN/CLOSE/OBSERVE/FIND/WAIT/POINT
- 厨房: CUT/WASH/POUR/MIX/COOK/HEAT/COOL/PACK
- 社交: HANDOVER/SPEAK/REPORT/SHAKE_HANDS/WAVE/BOW/NOD/DANCE/HUG/HIGH_FIVE/GESTURE/LISTEN/CONFIRM

### KV Cache 复用

- `chat_once` 增加 `_last_system_prompt` 追踪
- 对 `cache_prompt: True` 的支持
- 输出 `[cache HIT/MISS]` 状态指示
- 首次调用处理全部 prompt，后续复用 KV 缓存

---

## 关键教训

1. **硬件确认先行** — 不要依赖产品文档，`nvidia-smi` + `uname -a` 直接确认
2. **FORCE_MMQ 不是通用优化** — 对 Ada (4090) 好，对 Ampere (Jetson) 有害
3. **NVIDIA 镜像分桌面版和 L4T 版** — Jetson 只能用 L4T 镜像或裸 ubuntu + 主机 CUDA 挂载
4. **边缘设备的 Docker 网络** — `--network host` 解决绝大多数问题
5. **GitHub 在 Docker build 内不可靠** — 改用 COPY 预下载源码或二进制
6. **提示词工程量级不小** — 模板长度直接影响端到端延迟，输入减 60% = 延迟减半
7. **系统资源分配** — Jetson 8 核但需留核给系统控制，threads=4 是合理选择
8. **sm_87 架构指定是必需的** — 不设 `CMAKE_CUDA_ARCHITECTURES="87"` 会导致运行时 JIT 崩溃
9. **dockerignore 不小** — 不加 `.dockerignore` 会让构建上下文膨胀到 666MB+
10. **裸镜像 + host mount 是 Jetson Docker 的正确路径** — 不要试图用 nvidia/cuda 镜像跑 L4T

---

## 最终架构图

```
┌────────────────────────────────────────────────────────────────┐
│  R005 / Jetson Orin NX (L4T R36.4.3, CUDA 12.6, sm_87)       │
│                                                                │
│  ┌────────────────────┐   ┌────────────────────┐   ┌──────────────────┐│
│  │ qwen35:8081       │   │ robobrain:8082     │   │ qwen35-08b:8083  ││
│  │ Qwen3.5-2B        │   │ RoboBrain2.5-4B   │   │ Qwen3.5-0.8B    ││
│  │ batch=512 ubatch=256│   │ batch=512 ubatch=256│   │ batch=512        ││
│  │ threads=6 ctx=4096 │   │ threads=6 ctx=4096  │   │ threads=4 ctx=4096││
│  │ flash-attn on       │   │ flash-attn on       │   │ flash-attn on    ││
│  └──────────┬──────────┘   └──────────┬──────────┘   └────────┬─────────┘│
│             │                          │                       │         │
│  ┌──────────┴──────────────────────────┴───────────────────────┴────────┐│
│  │  Docker (ubuntu:22.04 + host CUDA mount)                            ││
│  │  network_mode: "host"     restart: unless-stopped                   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                │
│  CUDA libs: /usr/local/cuda-12.6/targets/aarch64-linux/lib     │
│  Models:   ~/Desktop/r005Test/models/                          │
└────────────────────────────────────────────────────────────────┘
```


