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
# Prompt 模板 (v2 — 优化分阶段调度 + 用户交互 + 烹饪技能)
#
# 主要改进:
#   1. 技能库增加 CUT/COOK/WASH/POUR/PACK 等厨房技能
#   2. 调度器引入"分阶段"概念，复杂任务拆为多个阶段
#   3. 格式简化，减少特殊符号，小模型更容易跟随
#   4. 每个模板加入 few-shot 示例
#   5. 明确用户交互位置（何时问/何时答/何时汇报）
# ============================================================
class PromptTemplates:
    # ============================================================
    # 技能库（分类）
    # ============================================================
    SKILL_MOVE = "NAVIGATE(自主导航) APPROACH(靠近目标) FOLLOW(跟随人) LEAD(引导人)"
    SKILL_HAND = "REACH(伸手) GRASP(抓取) PICK(拾取) HOLD(持握) PLACE(放置) RELEASE(放下) PUSH(推动) PULL(拉动) ROTATE(旋转) PRESS(按压)"
    SKILL_ENV  = "OPEN(打开) CLOSE(关闭) OBSERVE(观察) FIND(寻找) WAIT(等待) POINT(指向)"
    SKILL_KITCHEN = "CUT(切割) WASH(清洗) POUR(倾倒) MIX(搅拌) COOK(烹饪) HEAT(加热) COOL(冷却) PACK(包装)"
    SKILL_SOCIAL = (
        "HANDOVER(递交物品给人) SPEAK(说话/对话) REPORT(汇报状态) "
        "SHAKE_HANDS(握手) WAVE(挥手打招呼) BOW(鞠躬) NOD(点头) "
        "DANCE(跳舞) HUG(拥抱) HIGH_FIVE(击掌) GESTURE(做手势示意) "
        "LISTEN(聆听/等待指令) CONFIRM(向用户确认)"
    )
    SKILL_LIB = (
        "【移动】" + SKILL_MOVE + "\n"
        "【操作】" + SKILL_HAND + "\n"
        "【环境】" + SKILL_ENV + "\n"
        "【厨房】" + SKILL_KITCHEN + "\n"
        "【社交】" + SKILL_SOCIAL
    )

    SCHEDULER = (
        "你拆解指令为分阶段动作序列。\n\n"
        "先判断是否涉及人：提到\"我/你/客人/递给/握手/陪伴\" → HRI=是。\n"
        "HRI=是：最终阶段必须含NAVIGATE(靠近人)+社交技能。\n"
        "HRI=否：纯物体操作，不用社交动作。\n\n"
        "【技能库】\n"
        + SKILL_LIB + "\n\n"
        "输出：HRI判断 + 阶段划分 + 每步：动作|目标|目的\n\n"
        "【示例1：纯操作】\n"
        "指令：把苹果放冰箱\n"
        "HRI：否 | 阶段：拿→开→放\n"
        "1. NAVIGATE|台面|到苹果处 2. GRASP|苹果|拿起\n"
        "3. NAVIGATE|冰箱|走过去 4. OPEN|冰箱门|打开\n"
        "5. PLACE|苹果入冰箱|放进去 6. CLOSE|冰箱门|关上\n\n"
        "【示例2：递送】\n"
        "指令：做个汉堡端给我\n"
        "HRI：是|需靠近用户+递交+告知\n"
        "阶段：备料→烹饪→组装→递送\n"
        "1. NAVIGATE|冰箱|取鸡肉 2. OPEN|冰箱门|打开\n"
        "3. GRASP|鸡肉|取出 4. WASH|鸡肉|清洗\n"
        "5. CUT|鸡肉|切块 6. NAVIGATE|灶台|去灶台\n"
        "7. HEAT|油锅|热油 8. COOK|鸡排|炸金黄\n"
        "9. NAVIGATE|台面|去台面 10. GRASP|面包|取\n"
        "11. PLACE|鸡排放面包|组装 12. NAVIGATE|用户位置|靠近\n"
        "13. HANDOVER|汉堡|递上 14. SPEAK|告知|请享用\n"
        "收尾：BOW|鞠躬|致谢"
    )

    VISUAL_EXPERT = (
        "你是视觉理解专家。严格基于图片回答，不确定就说\"不确定\"。\n\n"
        "【输出格式】\n"
        "场景描述：[一句话概括画面]\n"
        "检测到人：[是/否] 如果是：有几个人、他们在做什么、表情如何、是否面向机器人\n"
        "物体清单（最多5个）：\n"
        "- [物体名称] | 颜色 | 位置 | 状态\n"
        "场景类型：[室内/室外/夜景/文字/动漫/建筑/其他]\n"
        "社交回应建议：[如果画面中有人，建议机器人做什么社交动作]\n"
        "可执行动作建议：[基于场景推荐1-2个机器人可做的动作]\n"
        "\n"
        "【示例1：有人的场景】\n"
        "用户图片：客人站在门口\n"
        "场景描述：一个人站在门口，面带微笑看向机器人\n"
        "检测到人：是，1人，面向机器人微笑\n"
        "物体清单：\n"
        "- 门 | 棕色 | 左侧 | 半开\n"
        "- 手提包 | 黑色 | 客人手中 | 提着\n"
        "场景类型：室内\n"
        "社交回应建议：客人到来，应WAVE挥手打招呼，SPEAK问候\n"
        "可执行动作建议：WAVE(挥手) → SPEAK问候 → GESTURE(引导入座)\n"
        "\n"
        "【示例2：物体场景】\n"
        "用户图片：厨房台面\n"
        "场景描述：厨房台面上有一个苹果和一把刀\n"
        "检测到人：否\n"
        "物体清单：\n"
        "- 苹果 | 红色 | 台面中央 | 完整\n"
        "- 刀 | 银色 | 台面右侧 | 干净\n"
        "场景类型：室内\n"
        "社交回应建议：无需社交交互\n"
        "可执行动作建议：GRASP(苹果) → WASH(清洗) → CUT(切块) → PLACE(装盘)"
    )

    TEMPORAL_ANALYST = (
        "你分析任务步骤的依赖关系和执行顺序。\n\n"
        "输出：因果链(标注前提) + 必须串行/可以并行 + 风险点\n\n"
        "【示例：做饭】\n"
        "任务：取食材→洗→切→炒→装盘\n"
        "因果链：OPEN冰箱(前提:先走到)→GRASP食材→CLOSE→WASH→CUT(前提:洗过)→"
        "HEAT锅→COOK(前提:切好+锅热)→PLACE装盘\n"
        "串行：洗→切→炒（不能颠倒）\n"
        "并行：热锅和切菜可同时\n"
        "风险：冰箱无食材则切换采购任务\n\n"
        "【示例：社交】\n"
        "任务：迎接→引导→倒茶→送别\n"
        "因果链：OBSERVE门口(前提:门铃)→NAVIGATE→WAVE→SPEAK问候→"
        "GESTURE引导→NAVIGATE带路→SPEAK询问→HANDOVER递茶→"
        "NAVIGATE送客→BOW→WAVE道别\n"
        "串行：问候→引导→入座→招待\n"
        "并行：倒茶同时可聊天\n"
        "风险：客人不喝则跳过倒茶"
    )

    EXECUTION_PLANNER = (
        "你为指定阶段生成原子动作序列。\n\n"
        "先判断阶段类型：操作用物体技能，社交/递交用社交技能。\n"
        "递交阶段必须：NAVIGATE(靠近人)+HANDOVER(递交)+SPEAK(告知)。\n\n"
        "【技能库】\n"
        + SKILL_LIB + "\n\n"
        "输出：阶段类型 + 每步：动作|目标|目的\n\n"
        "【示例1：操作】\n"
        "阶段：烹饪 | 类型：操作\n"
        "1. NAVIGATE|灶台|过去 2. HEAT|油锅|热油 3. COOK|鸡排|炸金黄\n\n"
        "【示例2：递交】\n"
        "阶段：递送 | 类型：递交\n"
        "1. NAVIGATE|用户位置|靠近 2. SPEAK|告知|说您的咖啡\n"
        "3. HANDOVER|咖啡杯|递上 4. BOW|鞠躬|致意"
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
        self._last_system_prompt = None  # 用于 KV cache 跟踪
        self._cache_hits = 0
        self._cache_misses = 0
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
        """单轮对话，利用 llama-server KV cache"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})

        # 检测 cache 是否可复用（同一 system prompt 第二次调用）
        cache_hit = (system_prompt is not None
                     and system_prompt == self._last_system_prompt)
        self._last_system_prompt = system_prompt
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        payload = {
            "model": self.current,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "cache_prompt": True,  # 启用 KV cache（最新版 llama-server 默认开启）
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
            usage["_cache"] = "HIT" if cache_hit else "MISS"
            usage["_cache_hits"] = self._cache_hits
            usage["_cache_misses"] = self._cache_misses
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
    """格式化时间输出，含 KV cache 状态"""
    prompt_tok = usage.get("prompt_tokens", 0)
    completion_tok = usage.get("completion_tokens", 0)
    tok_s = completion_tok / elapsed if elapsed > 0 and completion_tok else 0
    cache = usage.get("_cache", "?")
    cache_str = f" [cache {cache}]" if cache in ("HIT", "MISS") else ""
    print(f"  [{label}] {elapsed:.2f}s  (输入{prompt_tok} -> 输出{completion_tok} tok, {tok_s:.1f} tok/s){cache_str}")


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
