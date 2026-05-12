# Work Log — RoboAgent VLA CUDA Deployment

> 项目工作日志，按日期倒序排列。

---

## 2026-05-12

### Docker CUDA 方案最终修复 — ubuntu:22.04 + 主机 CUDA 挂载

**背景**: 之前 Docker 方案因 nvidia/cuda 镜像桌面版 cuBLAS 与 Jetson L4T 驱动不兼容而废弃（cublasCreate_v2 崩溃）。改用主机原生 nohup 方案运行。

**修复方案**:
1. Dockerfile FROM `ubuntu:22.04`（不是 nvidia/cuda）
2. 在 R005 主机预编译 llama-server（去 MMQ），Dockerfile 只 COPY 二进制 + `*.so*`
3. 运行时通过 volumes 挂载主机 CUDA 库：`/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib:ro`
4. 设置 `LD_LIBRARY_PATH=/app:/usr/local/cuda/lib`
5. 编译指定 `CMAKE_CUDA_ARCHITECTURES="87"`（默认列表不含 sm_87，不指定则运行时 JIT 崩溃）

**关键区别对比**:
| 方案 | 镜像 | CUDA 库来源 | 结果 |
|------|------|------------|------|
| nvidia/cuda | nvidia/cuda:12.4.0-devel | 镜像内置桌面版 cuBLAS | ❌ 与 L4T 驱动不兼容 |
| ubuntu + host CUDA | ubuntu:22.04 | 主机 bind mount | ✅ 完全兼容 |
| 主机原生 | N/A | 主机原生 | ✅ 兼容但无容器隔离 |

### 新增 Qwen3.5-0.8B 轻量模型服务

- 新增 `qwen35-08b` 服务（端口 8083）
- 模型文件: `Qwen3.5-0.8B-Q4_K_M.gguf`（~0.6 GB）
- 配置: ctx-size 4096, threads 4, no-mmap + mlock
- 目标场景: 简单问答、低延迟推理，与 2B/4B 模型互补

### 基准测试脚本完成

- `run_benchmark.py` — 多场景基准测试（短问答/任务规划/长文本/批量推理）
- `run_agent_test.py` — RoboAgent 端到端流程模拟
- `test_services.sh` — 服务健康检查 + 对话快速测试
- `test_interactive.py` — 交互式对话（5 种模式 + 图片输入）

### 其他

- 优化方案报告 `R005_OPTIMIZATION_REPORT.md` 完成（方案 A/B/C 迭代历程）
- 工作记忆（memory）更新 Docker CUDA 修复经验
- 目录结构新增 `Qwen3.5-0.8B-GGUF/` 和 `pic/`

---

## 2026-05-11 (续)

### Docker 方案修复 — CUDA 12.6 + 单阶段镜像

**背景**: 之前 Docker 方案因 CUDA 版本不匹配（容器 12.4 vs 主机 12.6）废弃，
改用主机原生 nohup。Phase 0 重编等待期间，决定修好 Docker 方案。

**修改内容**:
1. CUDA 12.4 → 12.6（builder 和 runtime 全对齐主机）
2. 多阶段编译 → 单阶段运行时（主机编译，Docker 只 COPY）
3. 去掉 FORCE_MMQ（Ampere 不适用）
4. COPY 单个二进制 → `COPY *.so* /app/` 抓取所有共享库
5. 加 `.dockerignore` 排除无用文件，避免 666MB 构建上下文传输卡顿
6. docker-compose 模型路径修正（qwen3_5/ 子目录、RoboBrain . 分隔符）
7. `test_interactive.py` 响应解析加固（choices 缺失时输出 API 错误信息）

**教训**: Docker 的 .so 文件很多（libggml-cuda.so 单文件 666MB），必须清理多余 symlink 和加 .dockerignore。

### Phase 0 重编完成

- R005 上去掉 FORCE_MMQ 的 llama.cpp 重编完成
- 新二进制：`ARCHS = 500,610,700,750,800,860,890`（不含 FORCE_MMQ）
- Docker 容器启动成功，GPU offload 正常（Device 0: Orin, sm_87, 15655 MiB）
- 容器进入重复重启循环（模型路径错误），已修正

---

## 2026-05-11

### 架构修正 — R005 从 NPU 切换到 CUDA 路线

**背景**: 此前误认为 R005 是独立 NPU 架构（基于之前对国讯芯微 Compass SDK 的理解），所有 Dockerfile 都基于 `arm64v8/ubuntu` 编译 CPU-only 版本。

**修正过程**:
1. 用户指出 R005 实际上是 Jetson Orin NX（ARM64 + NVIDIA GPU），与 4090 共享同一套 CUDA 技术栈
2. 重新审查 R005 硬件规格: L4T R36.4.3, CUDA 12.6, JetPack 6.x, 8×Cortex-A78AE
3. 将所有 Dockerfile 从 `arm64v8/ubuntu` 改为 `nvidia/cuda:12.4.0-devel-ubuntu22.04`（multi-arch 自动选 arm64）
4. 编译参数全面对齐 4090: `GGML_CUDA=ON`, `GGML_CUDA_F16=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `GGML_CUDA_FORCE_MMQ=ON`, `GGML_NATIVE=ON`
5. 移除 BLIS 依赖（只对 CPU 路径有效，CUDA 下不相关）

### R005 Docker 部署尝试 — 遇阻

**尝试过程**:
1. `docker build --network host` — DNS 问题通过 `--network host` 解决
2. `nvidia/cuda:12.4.0` 镜像 pull 成功（多架构镜像自动选择 arm64 variant）
3. `apt-get update` — ports.ubuntu.com DNS 失败 → 加 `--network host` 后解决
4. GitHub `git clone` 在 Docker build 中卡住（网络超时）→ 改为在主机直接克隆，编译后 COPY 到镜像

**最终障碍**:
- 主机编译使用 CUDA 12.6，Docker 运行时镜像 `nvidia/cuda:12.4.0-runtime` 只有 CUDA 12.4
- CUDA 版本不匹配导致二进制无法运行（`CUDA driver version insufficient` / `symbol not found`）
- L4T 专用镜像 `nvcr.io/nvidia/l4t-ubuntu22.04:r36.4.3` 需要 `docker login nvcr.io`，NGC 账号未配置

**决策**: 放弃 R005 Docker 方案，改为主机编译 + nohup 原生运行。

### R005 主机原生部署成功

1. 在 R005 主机直接 `git clone` + `cmake` + `make` 编译 llama.cpp（约 20-40 分钟）
2. 编译参数含 `GGML_CUDA_FORCE_MMQ=ON`
3. `nohup` 启动 llama-server:
   - Qwen3.5-2B @ 8081 — GPU 25/25 layers offloaded
   - RoboBrain2.5-4B @ 8082 — GPU 37/37 layers offloaded
4. 容器通信使用 `network_mode: "host"` 解决 bridge 网络握手问题
5. 模型文件路径修正: RoboBrain 文件名为 `RoboBrain2.5-4B.Q4_K_M.gguf`（`.` 分隔符）

### 基准测试 — 发现问题

- R005 推理速度 ~4 tok/s（远低于期望）
- 定位根因: `FORCE_MMQ=ON` 对 Ampere 架构（Jetson Orin）不利
- 启动去 MMQ 重编（不含 `GGML_CUDA_FORCE_MMQ`）

### interactive_test.py 修复

- 文件存在 SyntaxError（第 25 行中文文本在括号拼接中导致解析错误）
- 将所有 prompt 模板改写为三引号字符串
- 支持双模型切换 + 5 种专业模式（通用、操作规划、场景理解、探索指导、空间推理）

### 其他

- 创建 `robobrain_onnx_export.py` ONNX 导出管线
- 更新工作记忆（memory/project_r005_optimization.md）
- 更新 task_plan.md 为 Phase 0-3 架构
- 创建 20260511.md 部署指南
- 创建 README.md / WORK_LOG.md / FAULT_LOG.md

---

## 2026-05-10 及之前

- 项目启动，确定部署方案（4090 + R005）
- 初始 R005 架构假设为独立 NPU，通过 Compass SDK 加速
- 编写初始 Dockerfile（arm64v8/ubuntu CPU-only + BLIS）
- 编写 QwenAdapter（qwen_adapter.py）对接 llama-server HTTP API
- 4090 初始 CUDA 部署
