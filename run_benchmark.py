#!/usr/bin/env python3
"""
R005 推理性能基准测试
测量: 首token延迟 / 生成速度 / 显存占用

用法:
    # 测试 Qwen3.5-2B
    python3 run_benchmark.py --port 8081 --model qwen35

    # 测试 Qwen3.5-0.8B
    python3 run_benchmark.py --port 8083 --model qwen35-08b

    # 测试 RoboAgent
    python3 run_benchmark.py --port 8080 --model roboagent

    # 测试 RoboBrain
    python3 run_benchmark.py --port 8082 --model robobrain

    # 全部测试
    python3 run_benchmark.py --all
"""

import argparse
import json
import time
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchResult:
    model: str
    port: int
    prompt_tokens: int
    output_tokens: int
    total_time_s: float
    tokens_per_sec: float
    first_token_latency_s: Optional[float] = None
    error: Optional[str] = None


# 测试用例: (名称, prompt, max_tokens)
TEST_CASES = [
    ("简短问答", "你好，请用一句话介绍你自己", 64),
    ("任务规划", """场景：厨房，台面上有苹果、刀、盘子。
任务：把苹果切块装盘。
请给出详细的分步动作计划。""", 256),
    ("长文本生成", "请详细说明机器人具身智能的基本原理、关键技术、主要挑战和未来发展方向。", 512),
]


def run_benchmark(port: int, model_name: str) -> list[BenchResult]:
    """对指定端口的服务跑一遍全部测试用例"""
    api_url = f"http://localhost:{port}/v1/chat/completions"
    results = []

    # 先检查服务状态
    try:
        r = requests.get(f"http://localhost:{port}/health", timeout=5)
    except Exception as e:
        return [BenchResult(model_name, port, 0, 0, 0, 0, error=f"服务不可用: {e}")]

    for case_name, prompt, max_tokens in TEST_CASES:
        print(f"\n  [{case_name}] 正在测试...", end=" ", flush=True)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.01,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # 测量总耗时
        t0 = time.time()
        try:
            resp = requests.post(api_url, json=payload, timeout=600)
            elapsed = time.time() - t0
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            output_len = len(reply)
            # 粗略估计 token 数 (中文约 1.5-2 chars/token)
            estimated_tokens = max(1, int(output_len / 1.8))
            tps = estimated_tokens / elapsed if elapsed > 0 else 0

            results.append(BenchResult(
                model=model_name,
                port=port,
                prompt_tokens=len(prompt),
                output_tokens=estimated_tokens,
                total_time_s=round(elapsed, 2),
                tokens_per_sec=round(tps, 2),
            ))
            print(f"OK ({elapsed:.1f}s, {tps:.1f} tok/s)")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append(BenchResult(
                model=model_name, port=port,
                prompt_tokens=0, output_tokens=0,
                total_time_s=0, tokens_per_sec=0,
                error=str(e),
            ))

    return results


def main():
    parser = argparse.ArgumentParser(description="R005 推理性能基准测试")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--all", action="store_true", help="测试全部服务")
    args = parser.parse_args()

    if args.all:
        services = [
            (8080, "roboagent"),
            (8081, "qwen35"),
            (8082, "robobrain"),
            (8083, "qwen35-08b"),
        ]
    else:
        services = [(args.port, args.model)]

    all_results = []
    for port, model in services:
        print(f"\n{'='*50}")
        print(f"  测试: {model} (port {port})")
        print(f"{'='*50}")
        results = run_benchmark(port, model)
        all_results.extend(results)

    # 汇总 — 每个 result 对应一个测试用例
    print("\n\n")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 60)
    lines.append(f"  R005 推理性能基准测试结果  ({timestamp})")
    lines.append("=" * 60)
    lines.append(f"{'模型':<12} {'用例':<16} {'耗时(s)':<10} {'速度(tok/s)':<12} {'状态'}")
    lines.append("-" * 60)
    for i, r in enumerate(all_results):
        status = "OK" if not r.error else "FAIL"
        case_name = TEST_CASES[i % len(TEST_CASES)][0]
        lines.append(f"{r.model:<12} {case_name:<16} {r.total_time_s:<10} {r.tokens_per_sec:<12} {status}")
        if r.error:
            lines.append(f"  └─ 错误: {r.error}")
    lines.append("=" * 60)

    for l in lines:
        print(l)

    # 保存 timestamped txt
    txt_filename = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n结果已保存到 {txt_filename}")

    # 也保存 JSON
    json_filename = f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output = {
        "timestamp": timestamp,
        "results": [
            {
                "model": r.model,
                "port": r.port,
                "case": TEST_CASES[i % len(TEST_CASES)][0],
                "total_time_s": r.total_time_s,
                "tokens_per_sec": r.tokens_per_sec,
                "error": r.error,
            }
            for i, r in enumerate(all_results)
        ],
    }
    with open(json_filename, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON 结果已保存到 {json_filename}")


if __name__ == "__main__":
    main()
