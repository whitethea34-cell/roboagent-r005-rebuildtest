#!/bin/bash
# ============================================================
# R005 服务健康检查 + 快速测试脚本
# 用途: 验证推理服务是否正常运行
#
# 用法:
#   bash test_services.sh              # 测试所有服务
#   bash test_services.sh qwen35       # 只测 Qwen3.5-2B
#   bash test_services.sh robobrain    # 只测 RoboBrain
# ============================================================

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

SERVICES=(
    "qwen35:8081:Qwen3.5-2B"
    "robobrain:8082:RoboBrain2.5-4B"
    "qwen35-08b:8083:Qwen3.5-0.8B"
)

test_service() {
    local name="$1" port="$2" desc="$3"
    echo ""
    info "测试: $desc (port $port)"

    # 1. Health check
    HEALTH=$(curl -sf http://localhost:$port/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        pass "Health check OK ($HEALTH)"
    else
        fail "Health check 失败 — 服务未运行或端口错误"
        return 1
    fi

    # 2. 简单对话测试
    local result
    result=$(curl -sf http://localhost:$port/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "'"$name"'",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "用一句话介绍你自己"}
            ],
            "temperature": 0.7,
            "max_tokens": 128
        }' 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$result" ]; then
        local reply
        reply=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
        if [ -n "$reply" ]; then
            pass "对话测试通过"
            echo "    回复: ${reply:0:80}..."
        else
            fail "对话返回为空"
        fi
    else
        fail "对话测试失败 (HTTP 请求异常)"
    fi
}

# ==================== 主流程 ====================
echo "=========================================="
echo "  R005 推理服务测试"
echo "  时间: $(date)"
echo "=========================================="

if [ $# -ge 1 ]; then
    # 测试指定服务
    service_key="$1"
    for s in "${SERVICES[@]}"; do
        IFS=':' read -r name port desc <<< "$s"
        if [ "$name" = "$service_key" ]; then
            test_service "$name" "$port" "$desc"
            break
        fi
    done
else
    # 测试全部
    for s in "${SERVICES[@]}"; do
        IFS=':' read -r name port desc <<< "$s"
        test_service "$name" "$port" "$desc"
    done
fi

echo ""
echo "=========================================="
echo "  测试完成"
echo "=========================================="
