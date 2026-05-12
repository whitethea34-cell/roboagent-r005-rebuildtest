# Fault Log — 故障记录与解决方案

> 按时间倒序排列。每个故障包含: 现象 → 根因 → 解决方案 → 预防措施。

---

## F010 — nvidia/cuda 镜像 cuBLAS 与 L4T 不兼容

- **日期**: 2026-05-12
- **严重度**: 高 (容器无法运行推理)

**现象**: 基于 `nvidia/cuda:12.4.0-devel-ubuntu22.04` 的容器启动成功、GPU 识别正常，但推理时崩溃。`docker logs` 显示 `cublasCreate_v2` 相关错误。

**根因**: `nvidia/cuda` 官方镜像包含桌面级 cuBLAS（Desktop Linux x86_64 版），与 Jetson L4T 嵌入式环境的 cuBLAS 库不兼容。L4T 使用 Tegra-optimized cuBLAS 实现，API 签名虽兼容但底层实现不同。

**解决方案**: 
1. Dockerfile FROM 改为 `ubuntu:22.04` 裸镜像
2. 运行时通过 volumes 挂载主机 CUDA 库: `/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib:ro`
3. 容器内设 `LD_LIBRARY_PATH=/app:/usr/local/cuda/lib`

**预防**: Jetson L4T 上不要用 `nvidia/cuda` 系列镜像，需用裸 ubuntu + 主机 CUDA 挂载方案。

---

## F009 — sm_87 架构未指定导致 JIT 编译崩溃

- **日期**: 2026-05-12
- **严重度**: 高 (容器启动后秒崩)

**现象**: 容器启动后 GPU 初始化阶段崩溃，日志含 `CUDA error: no kernel image is available for execution on the device` 或 JIT 编译超时。

**根因**: llama.cpp CUDA 编译默认的 `CUDA_ARCHITECTURES` 列表不包含 sm_87（Jetson Orin 架构）。运行时 CUDA driver 试图 JIT 编译 PTX 到 sm_87，在 Jetson 环境下 JIT 编译非常耗时或直接崩溃。

**解决方案**: 编译时显式指定 `-DCMAKE_CUDA_ARCHITECTURES="87"`，跳过对其他架构的编译，也避免运行时 JIT。

**预防**: Jetson Orin (sm_87) 必须设置此 cmake 参数。`nvidia-smi` 输出中可确认 compute capability。

---

## F008 — test_interactive.py 'choices' KeyError

- **日期**: 2026-05-11
- **严重度**: 中 (脚本运行崩溃)

**现象**: 选择功能后报 `KeyError: 'choices'`，脚本崩溃。日志显示 `调度耗时: 0.00s (输入0 → 输出0 tok)`。

**根因**: llama-server API 返回了非标准响应（可能模型未加载完成、路径错误或内部错误），响应 JSON 不包含 `choices` 字段。脚本直接访问 `data["choices"]` 未做安全检查。

**解决方案**: 3 处响应解析全部加 `if "choices" not in data` 守卫，输出 `data.get("error")` 辅助排查。

**预防**: HTTP API 客户端应对所有可能的响应格式做防御性处理。

---

## F007 — Docker 镜像缺少共享库

- **日期**: 2026-05-11
- **严重度**: 高 (容器启动失败)

**现象**: 容器秒退出，log 显示 `error while loading shared libraries: libllama-common.so.0`，修复后又报 `libmtmd.so.0`。

**根因**: Dockerfile 只 COPY 了 `llama-server` 二进制，没带依赖的共享库。llama.cpp CUDA 编译产物包含多个 `.so`，运行时需 ld 加载。

**解决方案**:
1. 第一次: 追加 `COPY libllama-common.so.0 /app/`
2. 第二次: 改为 `COPY *.so* /app/` 一次性抓取所有共享库
3. 构建前清理多余 symlink（`.so`, `.so.0.0.1`），减少 666MB 重复传输

**预防**: COPY 二进制时务必连带所有 `.so`；先用 `ldd llama-server` 查看依赖清单。

---

## F006 — Docker CUDA 版本不匹配

- **日期**: 2026-05-11
- **严重度**: 高 (容器无法启动)

**现象**: 编译好的 llama-server 在容器内无法运行，报 CUDA driver 相关错误。

**根因**: 
- 主机编译使用 CUDA 12.6（JetPack 预装）
- Docker 运行时镜像 `nvidia/cuda:12.4.0-runtime` 只包含 CUDA 12.4 运行时库
- CUDA 主版本不匹配（12.6 编译的 PTX 不能被 12.4 的 driver 完整执行）

**解决方案**: 放弃 Docker 方案，改为主机原生运行 (`nohup`)。

**预防**: 使用 Docker 时需确保编译镜像和运行时镜像的 CUDA 版本一致。跨版本运行需升级运行时镜像 CUDA 版本。

---

## F005 — L4T 镜像拉取 401 未授权

- **日期**: 2026-05-11
- **严重度**: 中 (备用方案受阻)

**现象**: `docker pull nvcr.io/nvidia/l4t-ubuntu22.04:r36.4.3` 返回 401 Unauthorized。

**根因**: NGC (NVIDIA GPU Cloud) 需要 `docker login nvcr.io` 鉴权，且 L4T 镜像可能需要特定授权。

**解决方案**: 
1. 临时方案 — 改用主机原生运行
2. 长期方案 — 申请 NGC 账号，配置 `docker login`

**预防**: 需要在镜像拉取前配置 NGC 认证。

---

## F004 — Docker build 中 GitHub clone 卡死

- **日期**: 2026-05-11
- **严重度**: 中 (构建超时)

**现象**: Docker build 过程中 `git clone https://github.com/ggerganov/llama.cpp.git` 卡住不动。

**根因**: Docker 网络隔离导致 GitHub 连接超时（已配置 `--network host` 但 git 操作仍不稳定）。

**解决方案**: 
1. 在 R005 主机直接 `git clone`
2. 主机编译 llama.cpp
3. Dockerfile 改为 `COPY` 编译产物而非重新克隆编译

**预防**: 网络受限环境中避免在 Dockerfile 内执行 git clone，改用主机下载 + COPY 方式。

---

## F003 — apt-get DNS 解析失败

- **日期**: 2026-05-11
- **严重度**: 中 (Docker build 中 apt 失败)

**现象**: Docker build 执行到 `apt-get update` 时报 DNS 解析错误。

**根因**: Docker 默认 bridge 网络下 DNS 配置未继承主机的 DNS 设置（Jetson 网络环境特定）。

**解决方案**: `docker build --network host ...` 使用主机网络栈。

**预防**: 在 Jetson/边缘设备上 `docker build` 默认加 `--network host`。

---

## F002 — interactive_test.py SyntaxError

- **日期**: 2026-05-11
- **严重度**: 高 (脚本无法运行)

**现象**: `python3 test_interactive.py` 报 `SyntaxError: invalid syntax` line 25。

**根因**: Python 字符串隐式拼接中混入了未转义的中文特殊字符，导致解析器无法正确识别字符串边界。

```python
# 错误写法示例
prompt = "你好"  "世界"  # 中文在括号拼接中导致解析歧义
```

**解决方案**: 将所有 prompt 模板从括号拼接改为一对 `"""..."""` 三引号字符串。

**预防**: 三引号字符串对中文内容更安全，避免隐式字符串拼接中的编码歧义。

---

## F001 — 最初 R005 架构误判

- **日期**: 2026-05-11 (修正)
- **严重度**: 高 (整条技术路线错误)

**现象**: 所有 Dockerfile 基于 `arm64v8/ubuntu` CPU-only 编译，依赖 BLIS 做 ARM CPU 加速，未使用 CUDA。

**根因**: 对 R005 硬件架构理解错误 — 误认为 R005 是独立 NPU 设备（国讯芯微 Compass SDK），实际是 Jetson Orin NX（NVIDIA CUDA GPU）。

**解决方案**:
1. 删除 BLIS 依赖（CUDA 路径不需要）
2. Dockerfile FROM 改为 `nvidia/cuda:12.4.0-devel-ubuntu22.04`（multi-arch）
3. cmake 启用 GGML_CUDA，与 4090 编译参数对齐
4. 更新所有文档

**预防**: 硬件架构确认阶段应直接登录设备检查 (`nvidia-smi`, `uname -a`, `dpkg -l | grep cuda`)，不要依赖产品文档或第三方描述。
