#!/bin/bash
# ============================================================
# 4090 WSL 推理服务测试
#
# 复用 r005_rebuildtest/ 目录下的已有测试脚本:
#   test_interactive.py    — 菜单驱动交互测试
#   run_benchmark.py       — 性能基准测试
#   run_agent_test.py      — RoboAgent 管线仿真
#
# 用法:
#   bash test_4090.sh                 # 健康检查全部服务
#   bash test_4090.sh check           # 仅健康检查
#   bash test_4090.sh dialogue        # 全部模型快速对话
#   bash test_4090.sh benchmark       # 跑性能基准测试
#   bash test_4090.sh agent           # 跑 Agent 管线测试
#   bash test_4090.sh interactive     # 进入交互菜单
#   bash test_4090.sh qwen35          # 单模型测试
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$BASE_DIR"

# ---- 模型端口映射 ----
declare -A PORTS=(
    ["qwen35"]=8081
    ["robobrain"]=8082
    ["qwen35-08b"]=8083
)
declare -A NAMES=(
    ["qwen35"]="Qwen3.5-2B"
    ["robobrain"]="RoboBrain2.5-4B"
    ["qwen35-08b"]="Qwen3.5-0.8B"
)
ALL_MODELS=("qwen35" "robobrain" "qwen35-08b")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "  ${YELLOW}[INFO]${NC} $1"; }

# ---- 健康检查 ----
cmd_check() {
    echo ""
    echo "============================================"
    echo "  4090 推理服务健康检查"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    # 检查 Docker 环境
    info "检查 Docker..."
    docker info > /dev/null 2>&1 || { fail "Docker 未运行"; return 1; }
    ok "Docker 运行中"

    # 检查 GPU
    nvidia-smi > /dev/null 2>&1 || { fail "GPU 不可用"; return 1; }
    ok "GPU 可用"

    # 检查容器
    local targets=("$@")
    [ ${#targets[@]} -eq 0 ] && targets=("${ALL_MODELS[@]}")

    local all_ok=0
    for key in "${targets[@]}"; do
        local port=${PORTS[$key]}
        local cname
        case "$key" in
            qwen35)     cname="qwen35-4090" ;;
            robobrain)  cname="robobrain-4090" ;;
            qwen35-08b) cname="qwen35-08b-4090" ;;
        esac

        # 容器状态
        local status
        status=$(docker inspect --format='{{.State.Status}}' "$cname" 2>/dev/null || echo "not_found")
        if [ "$status" != "running" ]; then
            fail "${NAMES[$key]} 容器状态: $status"
            all_ok=1
            continue
        fi

        # HTTP 健康检查
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            ok "${NAMES[$key]} (:$port) 在线"
        else
            fail "${NAMES[$key]} (:$port) 未响应"
            all_ok=1
        fi
    done

    echo ""
    [ $all_ok -eq 0 ] && ok "全部服务正常" || fail "部分服务异常"
    echo "============================================"
    echo ""
    return $all_ok
}

# ---- 快速对话测试（复用 run_benchmark.py 的简短问答） ----
cmd_dialogue() {
    local targets=("$@")
    [ ${#targets[@]} -eq 0 ] && targets=("${ALL_MODELS[@]}")

    for key in "${targets[@]}"; do
        local port=${PORTS[$key]}
        echo "--------------------------------------------"
        echo "  ${NAMES[$key]} (:${port}) 对话测试"
        echo "--------------------------------------------"

        python3 "$SCRIPTS_DIR/run_benchmark.py" --port "$port" --model "$key" || {
            fail "${NAMES[$key]} 对话失败"
        }
        echo ""
    done
}

# ---- 性能基准测试（复用 run_benchmark.py） ----
cmd_benchmark() {
    local targets=("$@")
    [ ${#targets[@]} -eq 0 ] && targets=("${ALL_MODELS[@]}")

    for key in "${targets[@]}"; do
        local port=${PORTS[$key]}
        echo ""
        echo "============================================"
        echo "  性能基准测试: ${NAMES[$key]} (:$port)"
        echo "============================================"
        python3 "$SCRIPTS_DIR/run_benchmark.py" --port "$port" --model "$key"
    done
}

# ---- Agent 管线测试（复用 run_agent_test.py） ----
cmd_agent() {
    local targets=("$@")
    [ ${#targets[@]} -eq 0 ] && targets=("${ALL_MODELS[@]}")

    for key in "${targets[@]}"; do
        local port=${PORTS[$key]}
        echo ""
        echo "============================================"
        echo "  Agent 测试: ${NAMES[$key]} (:$port)"
        echo "============================================"
        python3 "$SCRIPTS_DIR/run_agent_test.py" --port "$port"
    done
}

# ---- 交互菜单（复用 test_interactive.py） ----
cmd_interactive() {
    local key="${1:-qwen35}"
    local port=${PORTS[$key]}
    echo ""
    echo "启动交互菜单: ${NAMES[$key]} (:$port)"
    echo "提示: 菜单内可按 [6] 切换模型"
    echo ""
    python3 "$SCRIPTS_DIR/test_interactive.py" --port "$port"
}

# ---- 主入口 ----
main() {
    local cmd="${1:-check}"
    shift 2>/dev/null || true

    case "$cmd" in
        check|health)
            cmd_check "$@"
            ;;
        dialogue|chat)
            cmd_dialogue "$@"
            ;;
        benchmark|perf)
            cmd_benchmark "$@"
            ;;
        agent)
            cmd_agent "$@"
            ;;
        interactive|menu)
            cmd_interactive "$1"
            ;;
        # 单模型模式: bash test_4090.sh qwen35 → 测这一整组
        qwen35|robobrain|qwen35-08b)
            info "单模型模式: ${NAMES[$cmd]}"
            cmd_check "$cmd"
            cmd_dialogue "$cmd"
            ;;
        *)
            echo "用法: bash test_4090.sh <命令>"
            echo ""
            echo "命令:"
            echo "  check         健康检查（默认）"
            echo "  dialogue      全部模型快速对话"
            echo "  benchmark     性能基准测试"
            echo "  agent         Agent 管线测试"
            echo "  interactive   交互菜单测试"
            echo "  qwen35        仅测试 Qwen3.5-2B"
            echo "  robobrain     仅测试 RoboBrain"
            echo "  qwen35-08b    仅测试 Qwen3.5-0.8B"
            exit 1
            ;;
    esac
}

main "$@"
