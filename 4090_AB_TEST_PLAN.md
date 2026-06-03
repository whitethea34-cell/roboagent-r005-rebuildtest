# 4090 WSL A/B Test 工作计划 — InternVL3-1B

> **日期**: 2026-05-13
> **目标**: 在 4090 WSL 环境下完成 InternVL3-1B 的 llama.cpp vs TensorRT 对照实验，验证管线可行性，拿到性能对比数据，为 R005 迁移提供依据。
> **模型**: InternVL3-1B（标准 Qwen2 架构，TensorRT 零适配）
> **硬件**: RTX 4090 (Ada, SM89, 24GB VRAM, CUDA 12.4, WSL Ubuntu)

---

## 0. 前置说明：4090 vs R005 的差异与注意事项

| 参数 | 4090 (WSL) | R005 (Jetson Orin NX) | 影响 |
|------|:---------:|:---------------------:|------|
| 架构 | Ada (SM89) | Ampere (SM87) | CUDA kernel 二进制不通用 |
| 内存带宽 | 1008 GB/s | 102 GB/s | 带宽相近，但 4090 更充裕 |
| VRAM | 24 GB (独立) | 16 GB (统一内存) | 4090 更宽松 |
| FP8 | ✅ 支持 | ❌ 不支持 | **不能用 FP8**，否则 R005 无法复现 |
| CUDA | 12.4 (桌面版) | 12.6 (L4T) | API 版本差异 |
| TensorRT | 桌面版 TensorRT 10.x | TensorRT Edge-LLM | **不同产品** |

**跨平台一致性的核心原则**：
- 只用 FP16 和 INT4 AWQ —— 两种精度在 4090 (SM89) 和 R005 (SM87) 上都支持
- 4090 上用 TensorRT Edge-LLM 的 x86 debug 模式验证**功能正确性**（非性能）
- 真实性能对比在 R005 上进行，4090 的数据作为快速迭代参考

---

## 1. 阶段总览

```
Phase 1: llama.cpp GGUF 基线  (0.5 天)
  ├── 1.1 下载模型 + mmproj
  ├── 1.2 启动 llama-server (视觉模式)
  ├── 1.3 基准测试 (图文推理 + 纯文本)
  └── 1.4 记录基线数据

Phase 2: TensorRT 管线验证  (2-3 天)
  ├── 2.1 Python 环境 + Edge-LLM 安装 (x86)
  ├── 2.2 INT4 AWQ 量化 + ONNX 导出
  ├── 2.3 C++ Runtime 编译 (x86 debug)
  ├── 2.4 Engine 构建
  └── 2.5 推理测试 + 数据记录

Phase 3: 对比分析 + R005 迁移准备  (0.5 天)
  ├── 3.1 llama.cpp vs TensorRT 数据对比
  ├── 3.2 精度验证 (同一输入对比输出)
  └── 3.3 R005 迁移清单
```

---

## 2. Phase 1 — llama.cpp GGUF 基线

### 2.1 模型获取

使用 llama.cpp 官方预量化版本（含 mmproj）：

```bash
# 方案 A: 直接 -hf 自动下载 (最简单)
# llama-server 会自动拉取模型 + mmproj

# 方案 B: 手动下载到本地
cd /mnt/e/Claude_roboagent/llm-r005/models/
mkdir -p InternVL3-1B-GGUF
cd InternVL3-1B-GGUF

# 下载文本模型 GGUF (Q4_K_M, ~680 MB)
wget https://huggingface.co/ggml-org/InternVL3-1B-Instruct-GGUF/resolve/main/InternVL3-1B-Instruct-Q4_K_M.gguf

# 下载 mmproj (视觉投影器, ~620 MB)
wget https://huggingface.co/ggml-org/InternVL3-1B-Instruct-GGUF/resolve/main/mmproj-InternVL3-1B-Instruct-f16.gguf
```

### 2.2 启动 llama-server（视觉模式）

```bash
# 在 4090 Docker 容器中启动
# 复用现有 docker-compose.4090.yaml 框架

# 方式 A: 新增 docker-compose 条目
# 在 4090_test/docker-compose.4090.yaml 中添加 internvl 服务

# 方式 B: 直接 docker run
docker run -d --rm --runtime nvidia --gpus "device=0" \
  --name r005-internvl-llama \
  --network host \
  -v /mnt/e/Claude_roboagent/llm-r005/models/InternVL3-1B-GGUF:/models:ro \
  -v /mnt/e/Claude_roboagent/r005_rebuildtest/pic:/pic:ro \
  roboagent:4090-cuda \
  -m /models/InternVL3-1B-Instruct-Q4_K_M.gguf \
  --mmproj /models/mmproj-InternVL3-1B-Instruct-f16.gguf \
  --host 0.0.0.0 --port 8084 \
  --ctx-size 8192 --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --verbose
```

### 2.3 基准测试脚本

创建 `4090_test/bench_internvl_llama.py`：

```python
"""
InternVL3-1B llama.cpp 基线测试
测试项:
  1. 纯文本推理 (decode tok/s)
  2. 图文推理 (decode tok/s + 视觉编码时间)
  3. 长文本推理 (2K context)
  4. 批量基准 (连续 10 次请求)
"""
import requests, time, json, base64
from pathlib import Path

SERVER = "http://localhost:8084/v1/chat/completions"

# 测试用图片
TEST_IMAGE = Path("/mnt/e/Claude_roboagent/r005_rebuildtest/pic/test.jpg")

# 测试用例
TEST_CASES = {
    "text_short": [
        {"role": "user", "content": "What is a robot? Answer in one sentence."}
    ],
    "text_long": [
        {"role": "user", "content": "Explain in detail the differences between autonomous and teleoperated robots, including their sensing, planning, and actuation pipelines."}
    ],
    "vision_describe": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(TEST_IMAGE.read_bytes()).decode()}"}},
            {"type": "text", "text": "Describe this image in detail."}
        ]}
    ],
    "vision_spatial": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(TEST_IMAGE.read_bytes()).decode()}"}},
            {"type": "text", "text": "Where is the red object relative to the blue object? Be specific about spatial relationships."}
        ]}
    ],
}

def bench_once(payload, max_tokens=256):
    t0 = time.time()
    resp = requests.post(SERVER, json={
        "messages": payload,
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    elapsed = time.time() - t0
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "elapsed_s": round(elapsed, 2),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tok_s": round(usage.get("completion_tokens", 0) / elapsed, 1) if elapsed > 0 else 0,
        "text": data["choices"][0]["message"]["content"][:100],
    }

def main():
    print("=" * 60)
    print("InternVL3-1B llama.cpp 基线测试 (4090)")
    print("=" * 60)

    results = {}
    for name, msgs in TEST_CASES.items():
        print(f"\n--- {name} ---")
        # 预热
        bench_once(msgs, max_tokens=64)
        time.sleep(1)
        # 正式测试 (3 次取平均)
        runs = []
        for i in range(3):
            r = bench_once(msgs, max_tokens=256)
            runs.append(r)
            print(f"  Run {i+1}: {r['tok_s']} tok/s, {r['elapsed_s']}s, "
                  f"prompt={r['prompt_tokens']}, gen={r['completion_tokens']}")
            time.sleep(0.5)
        avg_tok_s = sum(r["tok_s"] for r in runs) / len(runs)
        results[name] = {"avg_tok_s": round(avg_tok_s, 1), "runs": runs}

    print("\n" + "=" * 60)
    print("Summary:")
    for name, r in results.items():
        print(f"  {name}: {r['avg_tok_s']} tok/s (avg over 3 runs)")

    # 保存结果
    with open("baseline_llama_cpp.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to baseline_llama_cpp.json")

if __name__ == "__main__":
    main()
```

### 2.4 预期结果

| 指标 | 预期值 | 说明 |
|------|:-----:|------|
| 纯文本 decode | **100-200 tok/s** | 1B 模型在 4090 上极快 |
| 视觉编码 | **0.3-0.8s** | mmproj 在 GPU 上处理 448×448 |
| 图文 decode | **80-150 tok/s** | image tokens 增加 context |
| 显存占用 | **~3-4 GB** | 模型 ~0.7GB + KV Cache + CUDA 开销 |

> 这些数据仅作 4090 上的对照基线。R005 上会显著更低。

---

## 3. Phase 2 — TensorRT 管线验证

### 3.1 关键约束

> **TensorRT Edge-LLM 在 x86/4090 上的定位是开发调试（非性能优化）**。
> 此阶段的目标是验证管线功能的正确性：量化→导出→构建→推理 全流程跑通。
> 真实的性能数据在 R005 上获取。

### 3.2 Python 导出环境 (WSL)

```bash
# --- 在 WSL Ubuntu 中执行 ---

# 1. 安装系统依赖
sudo apt update
sudo apt install -y python3-pip python3-venv cmake build-essential git

# 2. 确认 CUDA 环境
nvcc --version    # 应显示 CUDA 12.4
nvidia-smi        # 确认 4090 可用

# 3. 确认 TensorRT 已安装 (NVIDIA 官方镜像通常预装)
dpkg -l | grep tensorrt
# 如未安装: pip install tensorrt

# 4. 克隆 TensorRT Edge-LLM
cd ~
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git submodule update --init --recursive

# 5. 创建 Python 虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 6. 安装 Edge-LLM Python 依赖
pip install -r requirements.txt
pip install -r experimental/llm_loader/requirements.txt
pip install --no-deps .

# 7. 验证安装
tensorrt-edgellm-quantize-llm --help
tensorrt-edgellm-export-llm --help
tensorrt-edgellm-export-visual --help
```

### 3.3 INT4 AWQ 量化 + ONNX 导出

```bash
cd ~/TensorRT-Edge-LLM
source venv/bin/activate

export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
mkdir -p $WORKSPACE_DIR

# ====== 路径 A: 从官方 AWQ 模型直接导出 (推荐) ======
# 先下载官方 AWQ 量化版本
huggingface-cli download OpenGVLab/InternVL3-1B-AWQ \
  --local-dir $WORKSPACE_DIR/InternVL3-1B-AWQ

# 从 AWQ checkpoint 导出 ONNX
export PYTHONPATH=$(pwd):$(pwd)/experimental:$PYTHONPATH
python -m llm_loader.export_all_cli \
    $WORKSPACE_DIR/InternVL3-1B-AWQ \
    $WORKSPACE_DIR/InternVL3-1B/onnx-int4


# ====== 路径 B: 自行量化 (完整流程) ======
# Step 1: 下载原始 HF 模型
huggingface-cli download OpenGVLab/InternVL3-1B-hf \
  --local-dir $WORKSPACE_DIR/InternVL3-1B-hf

# Step 2: INT4 AWQ 量化
tensorrt-edgellm-quantize-llm \
    --model_dir $WORKSPACE_DIR/InternVL3-1B-hf \
    --output_dir $WORKSPACE_DIR/InternVL3-1B/quantized-int4 \
    --quantization int4_awq \
    --dtype fp16

# Step 3: 导出 LLM ONNX
tensorrt-edgellm-export-llm \
    --model_dir $WORKSPACE_DIR/InternVL3-1B/quantized-int4 \
    --output_dir $WORKSPACE_DIR/InternVL3-1B/onnx-int4/llm

# Step 4: 导出视觉 encoder ONNX (用原始 FP16 模型)
tensorrt-edgellm-export-visual \
    --model_dir $WORKSPACE_DIR/InternVL3-1B-hf \
    --output_dir $WORKSPACE_DIR/InternVL3-1B/onnx-int4/visual


# ====== 路径 C: FP16 (不量化, 仅验证功能) ======
tensorrt-edgellm-export-llm \
    --model_dir $WORKSPACE_DIR/InternVL3-1B-hf \
    --output_dir $WORKSPACE_DIR/InternVL3-1B/onnx-fp16/llm

tensorrt-edgellm-export-visual \
    --model_dir $WORKSPACE_DIR/InternVL3-1B-hf \
    --output_dir $WORKSPACE_DIR/InternVL3-1B/onnx-fp16/visual
```

> **建议先走路径 C (FP16)** 验证功能，确认管线没问题后再走路径 A (AWQ) 做性能对比。

### 3.4 C++ Runtime 编译 (x86 debug)

```bash
cd ~/TensorRT-Edge-LLM
source venv/bin/activate

# 确认 TensorRT 路径
TRT_DIR=$(python3 -c "import tensorrt; print(tensorrt.__path__[0])" 2>/dev/null || echo "/usr/local/tensorrt")
echo "TensorRT at: $TRT_DIR"

# x86 桌面 GPU 编译 (非 Jetson, 非 DRIVE)
rm -rf build && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=$TRT_DIR \
    -DCUDA_CTK_VERSION=12.4

make -j$(nproc)
# 编译约 5-15 分钟

# 验证产物
ls -la build/examples/llm/llm_build
ls -la build/examples/multimodal/mm_inference
ls -la build/examples/multimodal/visual_build
```

### 3.5 构建 TensorRT Engine

```bash
cd ~/TensorRT-Edge-LLM

# LLM engine
./build/examples/llm/llm_build \
    --onnxDir $WORKSPACE_DIR/InternVL3-1B/onnx-int4/llm \
    --engineDir $WORKSPACE_DIR/InternVL3-1B/engine/llm \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096

# Visual encoder engine
./build/examples/multimodal/visual_build \
    --onnxDir $WORKSPACE_DIR/InternVL3-1B/onnx-int4/visual \
    --engineDir $WORKSPACE_DIR/InternVL3-1B/engine
```

### 3.6 推理测试

```bash
# 准备输入 JSON
cat > $WORKSPACE_DIR/input_test.json << 'EOF'
{
    "batch_size": 1,
    "temperature": 0,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/mnt/e/Claude_roboagent/r005_rebuildtest/pic/test.jpg"},
                        {"type": "text", "text": "Describe this image in detail."}
                    ]
                }
            ]
        }
    ]
}
EOF

# 运行推理
./build/examples/multimodal/mm_inference \
    --engineDir $WORKSPACE_DIR/InternVL3-1B/engine \
    --inputFile $WORKSPACE_DIR/input_test.json \
    --outputFile $WORKSPACE_DIR/output_test.json \
    --dumpOutput

# 查看结果
cat $WORKSPACE_DIR/output_test.json
```

### 3.7 性能基准脚本

创建 `4090_test/bench_internvl_tensorrt.py`（Python 封装，调用 Edge-LLM 的 HLAPI 或 OpenAI-compatible server）：

```python
"""
InternVL3-1B TensorRT 基准测试
复用 Phase 1 的测试用例结构，替换 API endpoint
"""
# ... (结构同 Phase 1 bench 脚本，endpoint 改为 TensorRT 端口)

# 如果使用 Edge-LLM 的 OpenAI-compatible server:
# SERVER = "http://localhost:9080/v1/chat/completions"

# 如果是直接 C++ runtime:
# 使用 subprocess 调用 mm_inference 并解析输出
```

---

## 4. Phase 3 — 对比分析

### 4.1 数据对比模板

| 测试项 | llama.cpp (Q4_K_M) | TensorRT (INT4 AWQ) | 增益 |
|--------|:---:|:---:|:---:|
| 纯文本 decode tok/s | | | |
| 视觉编码时间 | | | |
| 图文 decode tok/s | | | |
| 端到端延迟 (图+文) | | | |
| 显存占用 | | | |
| 模型加载时间 | | | |
| **输出内容一致性** | — | — | ✅/⚠️/❌ |

### 4.2 精度验证

```bash
# 同一张图片 + 同一个 prompt，对比两个框架的输出
# 检查关键事实是否一致（对象识别、空间关系、颜色等）

# 图片: pic/test.jpg
# Prompt: "Describe this image in detail."
# 对比维度:
#   - 对象识别准确率 (列举了多少个正确对象)
#   - 空间关系正确性 (left/right/above/below)
#   - 颜色描述准确率
#   - 幻觉内容数量
```

### 4.3 通过标准

| 检查项 | 通过标准 |
|--------|---------|
| TensorRT 管线跑通 | llm_build + visual_build 成功生成 .engine |
| 推理可用 | mm_inference 返回正常文本 |
| 图文正确 | 输出包含照片中的关键对象（不是随机的） |
| 精度不退化 | INT4 AWQ 输出与 FP16 基线语义一致 (>90%) |
| 速度有增益 | TensorRT decode tok/s >= llama.cpp (至少不更慢) |

### 4.4 R005 迁移清单

当 4090 上验证通过后，R005 迁移只需改以下内容：

| 步骤 | 4090 | R005 | 差异 |
|------|------|------|------|
| ONNX 导出 | x86 WSL | **同一个 ONNX** | 直接 scp 过去 |
| C++ 编译 | x86 cmake (no toolchain) | aarch64 cmake + Jetson toolchain | 不同 cmake 参数 |
| Engine 构建 | SM89 | SM87 | **必须在 R005 上重新构建** |
| CUDA 版本 | 12.4 | 12.6 | Engine 构建时自动适配 |
| 推理运行 | x86 debug 模式 | aarch64 生产模式 | 同一个代码库 |
| 量化精度 | INT4 AWQ | INT4 AWQ | 完全一致 |

**R005 编译命令** (仅差异部分)：
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin
```

---

## 5. 时间线与产出

```
Day 1 (2-3h):
  ☐ 下载 InternVL3-1B GGUF + mmproj
  ☐ 启动 llama-server 视觉模式
  ☐ 运行 bench_internvl_llama.py
  ☐ 记录基线数据 → baseline_llama_cpp.json

Day 1-2 (4-6h):
  ☐ 安装 TensorRT Edge-LLM Python 环境
  ☐ 下载 InternVL3-1B-HF / AWQ 模型
  ☐ INT4 AWQ 量化 + ONNX 导出
  ☐ 编译 C++ Runtime (x86)
  ☐ 构建 Engine
  ☐ 运行推理测试

Day 2-3 (2-3h):
  ☐ 调试 TensorRT 管线问题
  ☐ 编写 bench_internvl_tensorrt.py
  ☐ 运行性能基准
  ☐ 精度对比验证

Day 3 (1-2h):
  ☐ 填写对比数据表
  ☐ 判定通过/失败
  ☐ 输出 R005 迁移方案
  ☐ 更新 WORK_LOG.md
```

### 每阶段完成后的产出

| 阶段 | 产出物 |
|------|--------|
| Phase 1 | `baseline_llama_cpp.json` — llama.cpp 性能数据 |
| Phase 2 | `baseline_tensorrt.json` — TensorRT 性能数据, `InternVL3-1B/engine/` — 可复用的 Engine 文件 |
| Phase 3 | `ab_test_report.md` — 对比分析报告, R005 迁移清单 |

---

## 6. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|:---:|------|------|
| Edge-LLM x86 debug 编译失败 | 中 | 无法走 TensorRT 路径 | 准备备选: 用 mainline TensorRT-LLM 验证 (仅 4090) |
| mmproj 版本不匹配 | 低 | llama.cpp 视觉模式无法启动 | 用官方 ggml-org GGUF (text + mmproj 配套) |
| INT4 AWQ 构建 OOM | 低 | 24GB VRAM 对 1B 模型绰绰有余 | `maxInputLen` 和 `maxKVCacheCapacity` 调小 |
| ONNX 导出 dtype 漂移 | 中 | Engine 构建失败 | 路径 C (FP16) 先验证，排除量化引入的问题 |
| Edge-LLM x86 debug 性能不反映真实效果 | 高 | 4090 上 TensorRT 可能不比 llama.cpp 快 | **此测试主要验证功能**，性能结论以 R005 为准 |

---

## 7. 关键决策点

```
Phase 1 完成后:
  llama.cpp 基线数据到手
  → 如果基线 tok/s 异常低 (< 50)：排查 4090 环境问题
  → 如果正常: 继续 Phase 2

Phase 2 完成后:
  TensorRT 管线功能验证结果:
  → 如果跑通: 进入 Phase 3 对比
  → 如果失败: 分析根因 → 是否切换到 vLLM 备选方案？

Phase 3 完成后:
  对比数据到手:
  → 4090 上有正向增益 → 启动 R005 迁移
  → 4090 上无增益 → 不代表 R005 上也无增益
     (Edge-LLM x86 debug 模式本身性能不优化)
     → 仍建议在 R005 上验证一次 (R005 才是目标平台)
```

---

## 参考

- [ggml-org/InternVL3-1B-Instruct-GGUF](https://huggingface.co/ggml-org/InternVL3-1B-Instruct-GGUF) — 官方 GGUF (含 mmproj)
- [OpenGVLab/InternVL3-1B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-1B-AWQ) — 官方 AWQ INT4
- [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) — HF Transformers 格式
- [TensorRT Edge-LLM Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
- [TensorRT Edge-LLM x86 Build Discussion](https://forums.developer.nvidia.com/t/can-tensorrt-edge-llm-compile-and-run-device-part-build-and-inference-on-x86-host-machine-for-debug-purpose/356970)
- [[方案对比]] — 分析背景
- [[模型选型与验证路线]] — 选型决策
