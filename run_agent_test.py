#!/usr/bin/env python3
"""
RoboAgent 端到端集成测试
模拟 RoboAgent 的核心流程: 调度员→专家→动作

用法:
    # Qwen3.5-2B
    python3 run_agent_test.py --port 8081
    # Qwen3.5-0.8B
    python3 run_agent_test.py --port 8083
    # RoboBrain2.5-4B
    python3 run_agent_test.py --port 8082
"""

import argparse
import json
import os
import time
import requests

API_URL = "http://localhost:{}/v1/chat/completions"

# ==================== Prompt 模板（简化版） ====================

SYSTEM_PROMPT = """You are RoboAgent, an embodied AI assistant for household tasks.
You have the following abilities:
1. exploration_guidance(object) - suggest where to find an object
2. object_grounding(object) - locate object in current view
3. manipulation_planner(subtask) - plan atomic actions for a subtask
4. experience_summarization(subtask) - summarize task experience

Output format:
Think: <your reasoning>
Query: AbilityName(args)"""

TASK_INSTRUCTION = "put a clean apple in the fridge"
SCENE = "You are in a kitchen. CounterTop 1, Fridge 1, Apple 1, Sink 1, Plate 1"


def query_llm(port: int, messages: list, max_tokens: int = 512) -> str:
    """调用 llama-server API"""
    payload = {
        "model": "default",
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(API_URL.format(port), json=payload, timeout=120)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"


def test_step(port: int, step_name: str, prompt: str) -> str:
    """执行单步测试并打印结果"""
    print(f"\n{'='*50}")
    print(f"  [{step_name}]")
    print(f"{'='*50}")
    print(f"  Prompt: {prompt[:100]}...")

    t0 = time.time()
    reply = query_llm(port, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    elapsed = time.time() - t0

    print(f"  回复: {reply}")
    print(f"  耗时: {elapsed:.1f}s")
    return reply


def main():
    parser = argparse.ArgumentParser(description="RoboAgent 端到端集成测试")
    parser.add_argument("--port", type=int, default=8080, help="llama-server 端口")
    args = parser.parse_args()

    print("=" * 50)
    print("  RoboAgent 端到端测试")
    print(f"  API: localhost:{args.port}")
    print(f"  任务: {TASK_INSTRUCTION}")
    print("=" * 50)

    # Step 1: 调度员 — 任务分解
    result1 = test_step(args.port, "Step 1: 调度员 - 任务分解",
        f"Task: {TASK_INSTRUCTION}\nScene: {SCENE}\nAnalyze the task and generate ability queries.")

    # Step 2: 导航规划
    result2 = test_step(args.port, "Step 2: 导航规划",
        "You need to find and access a fridge. The kitchen has CounterTop 1, Fridge 1.\n"
        "Generate a navigation plan to reach the fridge.")

    # Step 3: 操作规划
    result3 = test_step(args.port, "Step 3: 操作规划",
        f"You are at Fridge 1. You need to: {TASK_INSTRUCTION}\n"
        f"Current inventory: nothing\n"
        f"Generate atomic actions.")

    # Step 4: 经验总结
    result4 = test_step(args.port, "Step 4: 经验总结",
        "Previous actions: [go to Sink 1, take Apple 1 from CounterTop 1, "
        "clean Apple 1 with Sink 1, go to Fridge 1, put Apple 1 to Fridge 1]\n"
        "Summarize the task execution experience.")

    # 汇总
    print("\n\n")
    print("=" * 50)
    print("  测试汇总")
    print("=" * 50)
    steps = [
        ("调度员 - 任务分解", result1),
        ("导航规划", result2),
        ("操作规划", result3),
        ("经验总结", result4),
    ]
    for name, result in steps:
        status = "OK" if result and "[ERROR]" not in result else "FAIL"
        print(f"  [{status}] {name}")


if __name__ == "__main__":
    main()
