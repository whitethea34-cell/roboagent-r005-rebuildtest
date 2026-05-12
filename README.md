# RoboAgent VLA Inference — CUDA Deployment Kit

> **目标**: 在 RTX 4090 (x86) 和 Jetson Orin NX (ARM64) 上部署 Qwen3.5-2B / Qwen3.5-0.8B / RoboBrain2.5-4B CUDA 推理，用于 Embodied AI / VLA (Vision-Language-Action) 模型推理加速。

---

## 硬件架构

| 硬件 | 架构 | CUDA | 显存 | 角色 |
|------|------|------|------|------|
| RTX 4090 | Ada Lovelace (sm_89) | 12.4 | 24 GB | 主推理节点 |
| R005 (Jetson Orin NX) | Ampere (sm_87) | 12.6 | ~15.6 GB (统一内存) | 边缘推理节点 |

**核心发现**: 两套硬件共享同一套 NVIDIA CUDA 技术栈，编译参数和运行时参数高度统一。差异仅在算力规模和架构优化 flag 上。

**新增模型 Qwen3.5-0.8B**: 2026-05-12 新增 0.8B 轻量模型部署（端口 8083），专为简单问答和低延迟场景设计。GGUF 文件约 0.6 GB，显存占用极低，推理速度极快。

---

## 技术栈

- **推理引擎**: [llama.cpp](https://github.com/ggerganov/llama.cpp) (GGML_CUDA)
- **部署方式**: Docker (4090) / Docker (R005, ubuntu:22.04 + 主机 CUDA 挂载)
- **模型格式**: GGUF (Q4_K_M)
- **SDK 版本**: CUDA 12.4 (4090), CUDA 12.6 + JetPack 6.x (R005)
- **量化**: Q4_K_M / Q4_0 (0.8B) / Q8_0 (待测)

---

## 快速开始

### 4090 (Docker)

```bash
# 构建
docker build -t qwen35-2b:4090 -f llm-r005/qwen35.Dockerfile .
docker build -t robobrain:4090 -f llm-r005/robobrain.Dockerfile .

# 启动
docker compose -f llm-r005/docker-compose.yml up -d
```

### R005 (Docker — 推荐)

R005 使用 **ubuntu:22.04 裸镜像 + 主机 CUDA 库挂载**，避免 `nvidia/cuda` 镜像桌面版 cuBLAS 与 L4T 驱动不兼容问题。编译必须在 R005 主机上完成（sm_87 架构 + L4T cuBLAS）。

```bash
# 先在 R005 主机上编译 llama-server
# （然后 COPY 到 Docker build 上下文中）
# 构建
docker build --network host -t qwen35-2b:r005 -f qwen35.Dockerfile .
docker build --network host -t robobrain:r005 -f robobrain.Dockerfile .

# 启动全部服务
docker compose -f docker-compose.r005.yml up -d qwen35
docker compose -f docker-compose.r005.yml up -d robobrain
docker compose -f docker-compose.r005.yml up -d qwen35-08b
```

### R005 (主机原生 — 备用方案)

当 Docker 方案因特殊环境不可用时，使用主机编译 + nohup 原生运行。

```bash
# 编译 (不含 FORCE_MMQ，Ampere 不适用)
cd ~
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_F16=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DLLAMA_BUILD_SERVER=ON
make -j$(nproc) llama-server

# 运行 (nohup 后台)
nohup ~/llama.cpp/build/bin/llama-server \
    -m ~/Desktop/r005Test/models/qwen3_5/Qwen3.5-2B-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8081 \
    --threads 6 --threads-batch 6 \
    --batch-size 512 --ubatch-size 256 \
    --n-gpu-layers 99 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on \
    > ~/logs/qwen35.log 2>&1 &
```

---

## 容器配置要点

| 参数 | 4090 | R005 (2B) | R005 (0.8B) | 说明 |
|------|------|-----------|-------------|------|
| batch-size | 4096 | 512 | 512 | Prompt 处理吞吐 |
| ubatch-size | 1024 | 256 | 256 | 微批大小 |
| threads | 8 | 6 | 4 | CPU 线程 (留核给系统) |
| n-gpu-layers | 99 | 99 | 99 | 全层卸载到 GPU |
| flash-attn | on | on | on | Flash Attention |
| cache-type-k/v | q8_0 | q8_0 | q8_0 | KV 缓存量化 |
| ctx-size | 8192 | 8192 | 4096 | 上下文窗口 |
| no-mmap | yes | — | no-mmap + mlock | 4090 启用，R005 不启用 |

---

## 文件结构

```
r005_rebuildtest/
├── README.md                  ← 本文件
├── WORK_LOG.md                ← 工作日志
├── FAULT_LOG.md               ← 故障记录
├── docker-compose.r005.yml    ← R005 Docker Compose 配置
├── qwen35.Dockerfile          ← Qwen3.5-2B 镜像构建
├── robobrain.Dockerfile       ← RoboBrain2.5-4B 镜像构建
├── test_interactive.py        ← 交互式对话测试
├── test_services.sh           ← 服务健康检查
├── run_benchmark.py           ← 基准测试
├── run_agent_test.py          ← Agent 流程测试
├── R005_操作指南.md              ← 中文操作指南
├── DEPLOY_R005.md             ← 部署说明
├── Dockerfile                 ← (废弃，使用 qwen35.Dockerfile / robobrain.Dockerfile)
├── Qwen3.5-0.8B-GGUF/         ← Qwen3.5-0.8B 模型文件
└── pic/                       ← 截图与架构图
```

---

## 关键经验

1. **CUDA 版本必须匹配**: 主机编译的二进制在容器内跑需要 CUDA 版本一致（12.4 容器不能跑 12.6 编译的产物）
2. **Jetson 避免 FORCE_MMQ**: MMQ 内核针对 Ada 架构优化，在 Ampere (Jetson Orin) 上性能反而下降
3. **Jetson Docker 不能用 nvidia/cuda 镜像**: 该镜像带桌面版 cuBLAS，与 L4T 驱动不兼容 → 改用 ubuntu:22.04 + 主机 CUDA 挂载
4. **指定 sm_87 架构**: 编译必须设 `CMAKE_CUDA_ARCHITECTURES="87"`，否则默认 arch 列表不含 sm_87，运行时 JIT 会崩溃
5. **Docker network=host**: Jetson bridge 网络有容器间握手问题
6. **Docker build 加 --network host**: 边缘设备 DNS 配置问题，build 时加此参数解决 apt-get 和 git clone 问题
7. **模型文件路径注意分隔符**: RoboBrain 文件名为 `RoboBrain2.5-4B.Q4_K_M.gguf`（点分隔），非 `-`
8. **0.8B 模型用小 batch 配置**: 4 threads + 4096 ctx-size + no-mmap + mlock，追求极低延迟
9. **.dockerignore 必不可少**: llama.cpp 编译产物含大量 .so symlink，不加 .dockerignore 会导致构建上下文达 666MB

---

## 后续优化方向

1. **Phase 0**: 去 MMQ 参数调优已完成，基准测试进行中
2. **Phase 0.5**: Qwen3.5-0.8B 轻量模型部署 + 简单问答场景测试
3. **Phase 1**: 推理管线优化（token 压缩、KV Cache 复用）
4. **Phase 2**: 量化方案对比（Q8_0 vs Q4_K_M）
5. **Phase 3**: TensorRT 加速（Jetson 视觉 encoder）

详见 [task_plan.md](../task_plan.md)，优化历程详见 [R005_OPTIMIZATION_REPORT.md](./R005_OPTIMIZATION_REPORT.md)。

---

## 依赖

- Docker + nvidia-docker2 (4090)
- CUDA 12.x + NVIDIA Driver >= 540
- llama.cpp (git clone --depth 1)
- Python 3.10+ (测试脚本)
- `requests` / `openai` (Python 客户端)
