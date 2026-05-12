# ============================================================
# RoboBrain2.5-4B 运行时镜像 (R005 / Jetson Orin NX, CUDA 12.6)
#
# 构建方式（主机编译 + COPY，跳过 Docker 内 git clone）:
#
#   1. 确保主机已有编译好的 llama-server:
#       ~/llama.cpp/build/bin/llama-server
#
#   2. 拷贝构建产物到当前目录:
#       cp ~/llama.cpp/build/bin/llama-server .
#       cp ~/llama.cpp/build/bin/*.so* .
#
#   3. 构建镜像（必须 --network host，修 DNS 解析）:
#       docker build --network host -t robobrain:r005 -f robobrain.Dockerfile .
#
#   4. docker-compose 启动:
#       docker compose -f docker-compose.r005.yml up -d robobrain
#
# 模型: RoboBrain2.5-4B
# GGUF: RoboBrain2.5-4B.Q4_K_M.gguf
# 后端: llama.cpp + CUDA 全层 GPU 卸载
# ============================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY llama-server /app/llama-server
COPY *.so* /app/

ENV LD_LIBRARY_PATH=/app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/llama-server"]
