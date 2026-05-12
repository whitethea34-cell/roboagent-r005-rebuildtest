#!/usr/bin/env python3
"""
R005 双模型交互测试脚本 — 菜单驱动版

适配 R005 (Jetson Orin NX) HTTP API (llama-server)，非直接 PyTorch 推理。
提供与 qwen_vl35_0_8B_test.py 类似的固定 prompt 模板 + 菜单 + 计时。

模型:
  qwen35      → http://localhost:8081 (Qwen3.5-2B)
  robobrain   → http://localhost:8082 (RoboBrain2.5-4B)
  qwen35-08b  → http://localhost:8083 (Qwen3.5-0.8B)

用法:
    python3 test_interactive.py
"""

import argparse
import base64
import json
import os
import sys
import time
import requests
from pathlib import Path

# ============================================================
# Prompt 模板 (移植自 qwen_vl35_0_8B_test.py)
# ============================================================
class PromptTemplates:
    SCHEDULER = (
        "你是一个机器人任务调度专家。将复杂自然语言任务拆解为原子步骤。\n"
        "严格按以下格式输出，不要多余内容。\n\n"
        "【思考阶段】\n"
        "1. 任务目标：\n"
        "2. 地点清单：\n"
        "3. 关键物体：\n"
        "4. 物理依赖：\n"
        "5. 时序逻辑：\n"
        "【执行计划】\n"
        "1. [动作] [物体/地点英文(中文)]\n"
        "2. ...\n"
        "【异常处理】\n"
        "1. [失败场景] → [替代策略]\n"
        "2. ...\n\n"
        "规则：首次进入的地点先加NAVIGATE；首次交互的物体先加FIND；每步单一动作；中英对照。"
    )

    VISUAL_EXPERT = (
        "你是一个视觉理解专家。请严格基于图片内容回答，不要猜测、不要编造。\n"
        "输出要求：\n"
        "1. 先用一句话说明这张图的「主体是什么/在做什么/场景是什么」。\n"
        "2. 列出 5 个最显著的物体/元素（如果不足 5 个就如实写）。\n"
        "3. 对每个物体给出：颜色、相对位置（左/右/中/上/下）、状态（例如开/关/亮/暗/是否在运动）。\n"
        "4. 如果图片是夜景/室内/文字截图/动漫插画/建筑等，请明确指出类型。\n"
        "5. 如果看不清或不确定，请输出「不确定」，不要用常识补全。"
    )

    TEMPORAL_ANALYST = (
        "你是一个时序分析专家。请根据任务描述分析动作的先后顺序和时间逻辑，输出清晰的因果链与步骤序列。"
    )

    EXECUTION_PLANNER = (
        "你是一个机器人动作规划专家。你的任务是将高级指令转化为精确的原子动作序列，且必须使用中文。\n"
        "可用动作库（示例）：\n"
        "- NAVIGATE to [地点]\n"
        "- FIND [物体]\n"
        "- PICK [物体]\n"
        "- PLACE [物体] to [地点/容器]\n"
        "- OPEN/CLOSE [物体]\n"
        "- USE [工具] to [动作] [物体]\n"
        "- WAIT [时间/事件]\n"
        "请严格按步骤输出，每一步需简要说明动作的物理目的，且严禁重复动作。"
    )


# ============================================================
# 模型注册表
# ============================================================
MODELS = {
    "qwen35": {
        "port": 8081,
        "name": "Qwen3.5-2B",
        "desc": "轻量对话/规划",
    },
    "robobrain": {
        "port": 8082,
        "name": "RoboBrain2.5-4B",
        "desc": "视觉+推理",
    },
    "qwen35-08b": {
        "port": 8083,
        "name": "Qwen3.5-0.8B",
        "desc": "超轻量快速推理",
    },
}


class R005Chat:
    """HTTP API 客户端（适配 llama-server）"""

    def __init__(self, default_model="qwen35"):
        self.models = MODELS
        self.current = default_model
        self.history = []
        self.last_time = 0
        self._check_all_services()

    # ---------- 服务状态 ----------

    def _health(self, port):
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=3)
            if r.status_code == 200:
                return True, "在线"
            return True, f"状态码 {r.status_code}"
        except Exception:
            return False, "未连接"

    def _check_all_services(self):
        print("=" * 50)
        print("  R005 推理服务状态")
        print("=" * 50)
        for key, cfg in self.models.items():
            status, detail = self._health(cfg["port"])
            icon = "OK" if status else "!!"
            print(f"  [{icon}] {cfg['name']:20s} :{cfg['port']}  {detail}")
        print(f"  {'─' * 40}")
        print(f"  当前模型: {self.model_name}")
        print("=" * 50)

    def show_status(self):
        print(f"\n{'='*50}")
        print(f"  R005 推理服务状态")
        print(f"{'='*50}")
        for key, cfg in self.models.items():
            ok, detail = self._health(cfg["port"])
            icon = "OK" if ok else "!!"
            marker = " ← 当前" if key == self.current else ""
            print(f"  [{icon}] /{key:12s} {cfg['name']:20s} :{cfg['port']}  {detail}{marker}")
        print(f"{'='*50}")

    @property
    def port(self):
        return self.models[self.current]["port"]

    @property
    def model_name(self):
        return self.models[self.current]["name"]

    # ---------- 模型切换 ----------

    def switch(self, model_key):
        if model_key not in self.models:
            available = ", ".join(self.models.keys())
            print(f"  [错误] 未知模型: {model_key}，可选: {available}")
            return False
        ok, detail = self._health(self.models[model_key]["port"])
        if not ok:
            print(f"  [错误] {self.models[model_key]['name']} 不在线 ({detail})")
            return False
        self.current = model_key
        self.history = []
        self.last_time = 0
        print(f"  [切换] → {self.model_name} (port {self.port})，上下文已清空")
        return True

    # ---------- 核心推理 ----------

    def chat(self, text, temperature=0.7, max_tokens=512):
        """纯文本对话"""
        self.history.append({"role": "user", "content": text})
        payload = {
            "model": self.current,
            "messages": list(self.history),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            data = resp.json()
            if "choices" not in data:
                err_msg = data.get("error", {}).get("message", str(data))
                return f"[API ERROR] {err_msg}", 0, {}
            reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            elapsed = time.time() - t0
            self.history.append({"role": "assistant", "content": reply})
            if len(self.history) > 20:
                self.history = self.history[-20:]
            return reply, elapsed, usage
        except Exception as e:
            return f"[ERROR] {e}", 0, {}

    def chat_once(self, system_prompt, user_text, temperature=0.7, max_tokens=512):
        """单轮对话（不使用 history），返回 (reply, elapsed, usage)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.current,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            data = resp.json()
            if "choices" not in data:
                err_msg = data.get("error", {}).get("message", str(data))
                return f"[API ERROR] {err_msg}", 0, {}
            reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            elapsed = time.time() - t0
            return reply, elapsed, usage
        except Exception as e:
            return f"[ERROR] {e}", 0, {}

    def vision(self, image_path, question, temperature=0.7, max_tokens=512):
        """图片理解（多模态）"""
        if not os.path.exists(image_path):
            return f"[错误] 图片不存在: {image_path}", 0, {}

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.current,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"http://localhost:{self.port}/v1/chat/completions",
                json=payload,
                timeout=300,
            )
            data = resp.json()
            if "choices" not in data:
                err_msg = data.get("error", {}).get("message", str(data))
                return f"[API ERROR] {err_msg}", 0, {}
            reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            elapsed = time.time() - t0
            return reply, elapsed, usage
        except Exception as e:
            return f"[ERROR] {e}", 0, {}

    def clear(self):
        self.history = []
        print("  [上下文已清空]")


# ============================================================
# 菜单 & 主循环
# ============================================================

MENU = """
===================== R005 双模型交互测试 =====================
  当前模型: {model_name:16s}   |  端口: {port}

  [1] 任务调度    — 复杂任务拆解 (SCHEDULER)
  [2] 视觉理解    — 图片分析 (VISION)
  [3] 时序分析    — 动作时序因果链 (TEMPORAL)
  [4] 执行规划    — 原子动作序列 (EXECUTION)
  [5] 自由对话    — 纯文本聊天
  [6] 切换模型    — 切换 qwen35 / robobrain
  [7] 设置图片    — 指定推理用图片路径
  [8] 状态检查    — 显示所有服务在线状态

  [0] / clear    — 清空上下文
  [q] 退出
===============================================================
选择功能: """


def print_banner():
    print("=" * 55)
    print("  R005 双模型交互测试 (Jetson Orin NX)")
    print("  模板移植自 qwen_vl35_0_8B_test.py")
    print("=" * 55)


def print_timing(label, elapsed, usage):
    """统一格式化时间输出"""
    prompt_tok = usage.get("prompt_tokens", 0)
    completion_tok = usage.get("completion_tokens", 0)
    tok_s = completion_tok / elapsed if elapsed > 0 and completion_tok else 0
    print(f"  ⏱ {label}: {elapsed:.2f}s  (输入{prompt_tok} → 输出{completion_tok} tok, {tok_s:.1f} tok/s)")


def main():
    parser = argparse.ArgumentParser(description="R005 双模型交互测试")
    parser.add_argument("--port", type=int, default=None, help="默认端口")
    args = parser.parse_args()

    default_model = "qwen35"
    if args.port:
        for key, cfg in MODELS.items():
            if cfg["port"] == args.port:
                default_model = key
                break

    client = R005Chat(default_model)
    print_banner()

    current_img = None

    while True:
        try:
            choice = input(MENU.format(
                model_name=client.model_name,
                port=client.port,
            )).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not choice:
            continue

        # ---- 退出 ----
        if choice in ("q", "quit", "exit"):
            break

        # ---- 清空上下文 ----
        if choice in ("0", "clear"):
            client.clear()
            continue

        # ---- 6. 切换模型 ----
        if choice == "6":
            target = input("  目标模型 (qwen35 / qwen35-08b / robobrain): ").strip().lower()
            client.switch(target)
            continue

        # ---- 7. 设置图片 ----
        if choice == "7":
            path = input("  图片路径: ").strip().strip("\"'")
            if not path:
                current_img = None
                print("  图片已清空")
            elif os.path.exists(path):
                current_img = path
                print(f"  图片已设置: {current_img}")
            else:
                print(f"  [警告] 文件不存在: {path}")
                current_img = path  # 仍然设置，vision 时会再次检查
            continue

        # ---- 8. 状态检查 ----
        if choice == "8":
            client.show_status()
            continue

        # ---- 1. 任务调度 (SCHEDULER) ----
        if choice == "1":
            user_input = input("  请输入复杂任务指令: ").strip()
            if not user_input:
                print("  [跳过] 空指令")
                continue
            print(f"  [{client.model_name}] 调度规划中...", end="", flush=True)
            reply, elapsed, usage = client.chat_once(
                PromptTemplates.SCHEDULER, user_input, max_tokens=384
            )
            print()
            print_timing("调度耗时", elapsed, usage)
            print(f"\n{reply}\n")
            continue

        # ---- 2. 视觉理解 (VISION) ----
        if choice == "2":
            if current_img is None:
                img_input = input("  请输入图片路径: ").strip().strip("\"'")
                if not img_input or not os.path.exists(img_input):
                    print("  [跳过] 图片不存在或未指定")
                    continue
                current_img = img_input
            question = input("  问题 (回车=默认): ").strip()
            if not question:
                question = "请按要求进行视觉理解。"
            system_prompt = PromptTemplates.VISUAL_EXPERT
            full_text = f"系统指令:\n{system_prompt}\n\n当前任务: {question}"

            print(f"  [{client.model_name}] 视觉理解中...", end="", flush=True)
            reply, elapsed, usage = client.vision(current_img, full_text, max_tokens=512)
            print()
            print_timing("视觉耗时", elapsed, usage)
            print(f"\n{reply}\n")
            continue

        # ---- 3. 时序分析 (TEMPORAL) ----
        if choice == "3":
            user_input = input("  请输入时序任务描述: ").strip()
            if not user_input:
                print("  [跳过] 空描述")
                continue
            print(f"  [{client.model_name}] 时序分析中...", end="", flush=True)
            reply, elapsed, usage = client.chat_once(
                PromptTemplates.TEMPORAL_ANALYST, user_input, max_tokens=512
            )
            print()
            print_timing("时序分析耗时", elapsed, usage)
            print(f"\n{reply}\n")
            continue

        # ---- 4. 执行规划 (EXECUTION) ----
        if choice == "4":
            user_input = input("  请输入执行指令: ").strip()
            if not user_input:
                print("  [跳过] 空指令")
                continue
            print(f"  [{client.model_name}] 执行规划中...", end="", flush=True)
            reply, elapsed, usage = client.chat_once(
                PromptTemplates.EXECUTION_PLANNER, user_input, max_tokens=256
            )
            print()
            print_timing("规划耗时", elapsed, usage)
            print(f"\n{reply}\n")
            continue

        # ---- 5. 自由对话 ----
        if choice == "5":
            user_input = input("  你说: ").strip()
            if not user_input:
                print("  [跳过] 空消息")
                continue
            print(f"  [{client.model_name}] ", end="", flush=True)
            reply, elapsed, usage = client.chat(user_input)
            print(f"({elapsed:.1f}s)" if elapsed > 0 else "(失败)")
            if usage:
                print(f"  (输入{usage.get('prompt_tokens',0)} → 输出{usage.get('completion_tokens',0)} tok)")
            print(f"  {reply}\n")
            continue

        # ---- 无效 ----
        print("  无效选择，请重新输入")


if __name__ == "__main__":
    main()
