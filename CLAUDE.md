# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboAgent VLA inference deployment targeting two hardware platforms:
- **RTX 4090** (x86, CUDA 12.4) — primary training/inference node
- **R005 / Jetson Orin NX** (ARM64, CUDA 12.6, L4T R36.4.3) — edge inference

Inference engine: **llama.cpp** (GGML_CUDA) serving GGUF-quantized models via HTTP API.
Two model tracks: **Qwen3.5-2B** (lightweight VL) and **RoboBrain2.5-4B** (embodied reasoning).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  R005 / Jetson Orin NX                                      │
│  llama-server containers (ubuntu:22.04 + host CUDA mount)   │
│  network_mode: "host" (no bridge, fixes Jetson networking)  │
│                                                             │
│  qwen35:8081  — Qwen3.5-2B.Q4_K_M.gguf    25/25 GPU layers │
│  robobrain:8082 — RoboBrain2.5-4B.Q4_K_M.gguf  37/37 layers│
│  qwen35-08b:8083 — Qwen3.5-0.8B            (lightweight)   │
└─────────────────────────────────────────────────────────────┘
```

**Key architectural facts:**
- Both 4090 and R005 share NVIDIA CUDA stack — only arch flags differ (sm_89 vs sm_87)
- DO NOT use `FORCE_MMQ` on Jetson Ampere — it's ~5-8x slower than default CUDA kernels
- Docker images are built from `ubuntu:22.04` (NOT `nvidia/cuda`) — host CUDA libs are bind-mounted at runtime because desktop cuBLAS is incompatible with Jetson L4T
- llama-server binary must be pre-compiled on host then COPY'd into Docker (no git clone in Dockerfile — network unreliable on edge devices)

## File Structure

```
├── docker-compose.r005.yml       # R005 service definitions (3 model containers)
├── qwen35.Dockerfile             # Qwen3.5-2B/0.8B runtime image (shared)
├── robobrain.Dockerfile          # RoboBrain2.5-4B runtime image
├── Dockerfile                    # DEPRECATED — use qwen35.Dockerfile / robobrain.Dockerfile
├── test_interactive.py           # Menu-driven interactive chat (5 modes + vision)
├── test_services.sh              # Health check + quick dialogue test
├── run_benchmark.py              # Performance benchmark (prompt/generation throughput)
├── run_agent_test.py             # RoboAgent end-to-end pipeline simulation
├── DEPLOY_R005.md                # Step-by-step: file transfer → health check → benchmarks
├── R005_操作指南.md                # Chinese deployment operations guide
├── R005_OPTIMIZATION_REPORT.md   # Optimization history (CPU → FORCE_MMQ → final CUDA)
├── WORK_LOG.md                   # Chronological work log
├── FAULT_LOG.md                  # Fault records with RCA + fix + prevention
├── Qwen3.5-0.8B-GGUF/           # Pre-downloaded 0.8B quantized model
└── pic/                          # Test images for vision inference
```

## Common Commands

```bash
# Build Docker images (requires pre-compiled llama-server in project root)
docker build --network host -t qwen35-2b:r005 -f qwen35.Dockerfile .
docker build --network host -t robobrain:r005 -f robobrain.Dockerfile .

# Deploy containers (R005)
docker compose -f docker-compose.r005.yml up -d qwen35
docker compose -f docker-compose.r005.yml up -d robobrain

# Health check
bash test_services.sh                    # all services
bash test_services.sh qwen35             # single service

# Interactive testing
python3 test_interactive.py --port 8081  # Qwen3.5-2B chat
python3 test_interactive.py              # menu-driven (model selection built-in)

# Benchmark
python3 run_benchmark.py --all
python3 run_benchmark.py --port 8081 --model qwen35

# Agent pipeline test
python3 run_agent_test.py --port 8081
```

## Critical Lessons (from FAULT_LOG)

1. **Hardware verification first** — verify with `nvidia-smi` + `uname -a`, don't trust product docs
2. **No FORCE_MMQ on Ampere** — only benefits Ada (4090), harms Jetson Orin (5-8x slower)
3. **Docker on Jetson**: use `ubuntu:22.04` base + mount host CUDA libs; `nvidia/cuda` images are incompatible with L4T
4. **Edge device networking**: always `--network host` for both build and runtime
5. **Avoid git clone in Dockerfile** on edge devices — pre-download and COPY instead
6. **Model file naming**: RoboBrain uses `.` separator (`RoboBrain2.5-4B.Q4_K_M.gguf`), not `-`; always `find` before filling compose paths
7. **HTTP API client hardening**: llama-server may return non-standard responses; guard `data["choices"]` access
