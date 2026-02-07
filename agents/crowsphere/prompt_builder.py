from __future__ import annotations

import re
from typing import Any

from agents.crowsphere.tool_definitions import ToolResult


def _parse_mario_step_summary(obs_text: str) -> str:
    """
    从预处理后的 Mario obs_text 中抽取“本步摘要”，把模型必须读懂的关键信息压缩成几行：
    - Urgency / max_jump_level
    - 哪些 skill 在 Risk Report 中是 OK
    - OK 且 jump_level>0 的 skill（用于 CRITICAL 禁止慢走的硬约束）

    这不是替模型做决策，只是降低“看漏 Risk Report / Candidates”的概率。
    """
    text = str(obs_text or "")
    if ("[Decision Support]" not in text) or ("[Risk Report]" not in text):
        return ""

    # Urgency
    urgency = "UNKNOWN"
    m = re.search(r"Urgency:\s*([A-Z]+)", text, re.IGNORECASE)
    if m:
        urgency = str(m.group(1)).upper()

    # Nearest threat (name + distance)
    nearest_name: str | None = None
    nearest_dist: int | None = None
    m = re.search(r"Nearest:\s*([A-Za-z_]+)\s+at\s*\(", text, re.IGNORECASE)
    if m:
        nearest_name = str(m.group(1)).strip().upper()
    m = re.search(r"Distance:\s*(\d+)\s*px", text, re.IGNORECASE)
    if m:
        try:
            nearest_dist = int(m.group(1))
        except Exception:
            nearest_dist = None

    # max_jump_level
    max_jump_level: int | None = None
    m = re.search(r"max_jump_level\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        try:
            max_jump_level = int(m.group(1))
        except Exception:
            max_jump_level = None

    # Candidates: skill -> jump_level
    candidates: dict[str, int] = {}
    in_candidates = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == "Candidates:":
            in_candidates = True
            continue
        if in_candidates:
            # Candidates 结束通常是空行或进入下一个 section
            if (not line) or line.startswith("["):
                break
            cm = re.match(r"-\s*([A-Z_]+)\s*:\s*jump_level\s*=\s*(\d+)\b", line, re.IGNORECASE)
            if not cm:
                continue
            skill = str(cm.group(1)).strip().upper()
            try:
                jl = int(cm.group(2))
            except Exception:
                continue
            candidates[skill] = jl

    # Risk Report: skill -> flags(list); OK => []
    risk_block = text.split("[Risk Report]", 1)[1]
    risk_block = re.split(r"\n\[[^\]]+\]\n", risk_block, maxsplit=1)[0]
    risks: dict[str, list[str]] = {}
    for raw in risk_block.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        rm = re.match(r"-\s*([A-Z_]+)\s*:\s*.*=>\s*(.*)$", line)
        if not rm:
            continue
        skill = str(rm.group(1)).strip().upper()
        flags_str = str(rm.group(2)).strip()
        if flags_str.upper() == "OK":
            risks[skill] = []
        else:
            flags = [f.strip() for f in flags_str.split(",") if f.strip()]
            # NEAR_PIT_EDGE 是“靠近坑边缘”的提示，通常不是硬致命；
            # 在 stairs+pit 窗口里它经常是唯一可行解，摘要里把它视为 OK，避免模型被迫选 SLOW_WALK 卡死。
            if flags and all(str(f).strip().upper().startswith("NEAR_PIT_EDGE@") for f in flags):
                risks[skill] = []
            else:
                risks[skill] = flags

    ok_skills = [s for s, flags in risks.items() if isinstance(flags, list) and len(flags) == 0]
    ok_skills.sort()
    ok_nonzero = [s for s in ok_skills if int(candidates.get(s, 0)) > 0]
    ok_nonzero.sort()

    ok_skills_str = "无" if not ok_skills else ", ".join(ok_skills)
    ok_nonzero_str = "无" if not ok_nonzero else ", ".join(ok_nonzero)
    max_str = "未知" if max_jump_level is None else str(max_jump_level)

    # 给出一个“可执行的推荐”，用于把以前 adapter 的强制兜底迁移到 prompt 层：
    # - 只要存在 OK，就优先在 OK 内按固定优先级选 1 个
    # - Urgency=CRITICAL 且 OK 内存在 jump_level>0 时，禁止推荐 jump_level=0
    # - Urgency=HIGH/CRITICAL 且最近威胁是敌人且距离<=75px，且 OK 内存在 jump_level>0 时，禁止推荐 jump_level=0
    # 竞赛约束：Super Mario 不允许宏动作/确定性规划。
    priority = ["STAIRS_JUMP", "DENSE_SAFE", "SAFE", "FAST", "STOMP", "SLOW_WALK"]
    ok_selectable = [s for s in ok_skills if int(candidates.get(s, 0)) <= int(max_jump_level or 6)]
    pool = list(ok_selectable)
    ok_nonzero_selectable = [s for s in ok_selectable if int(candidates.get(s, 0)) > 0]
    forbid_zero = False
    if urgency == "CRITICAL" and ok_nonzero_selectable:
        forbid_zero = True
    if (
        urgency in {"HIGH", "CRITICAL"}
        and ok_nonzero_selectable
        and nearest_name not in {None, "PIPE", "STAIRS"}
        and isinstance(nearest_dist, int)
        and nearest_dist <= 75
    ):
        forbid_zero = True
    if forbid_zero:
        pool = ok_nonzero_selectable
    recommended_ok: str | None = None
    for s in priority:
        if s in pool:
            recommended_ok = s
            break
    recommend_line = ""
    if recommended_ok is not None:
        if len(pool) == 1:
            recommend_line = f"- 本步必须选: {recommended_ok}\n"
        else:
            recommend_line = f"- 推荐(OK 内最高优先级): {recommended_ok}\n"

    return (
        "[本步摘要]\n"
        f"- Urgency: {urgency}\n"
        f"- max_jump_level: {max_str}\n"
        f"- Risk Report => OK: {ok_skills_str}\n"
        f"- OK 且 jump_level>0: {ok_nonzero_str}\n"
        f"{recommend_line}"
        "（你仍需按下方硬规则自己选择，并输出对应 jump_level；摘要只是帮助你不要漏看关键信息。）\n"
    )


def build_prompt(
    *,
    game: str,
    obs_text: str,
    game_info: dict[str, Any],
    memory_hint: str,
    tools_enabled: bool = False,
    tool_results: list[ToolResult] | None = None,
) -> str:
    if game == "twenty_fourty_eight":
        # 基础 prompt
        base_prompt = (
            "只输出 JSON 对象，不要解释。\n"
            "游戏：2048。\n"
            "目标：尽量让合并得分更高，并避免连续多次棋盘不变化。\n"
            "重要：连续 5 次动作让棋盘不变化会直接结束。\n"
            "请在输出前在脑中自检：你选择的方向必须让棋盘变化（Move Analysis 里 change=yes），否则会很快死亡。\n"
        )

        # 规则更新：2048 只允许 calculator 工具（如需做纯算术计算）。
        # 注意：常规情况下 obs_text 已包含 Move Analysis 的数值，模型可直接据此选方向；
        # 这里只保留 calculator 说明用于兼容与调试。
        tool_section = ""
        if tools_enabled:
            tool_section = (
                "\n[可用工具]\n"
                "- calculator: 计算数学表达式（四则运算/括号/abs/min/max）\n"
                "如需调用工具，输出：\n"
                '  {"type":"tool_call","tool_name":"calculator","arguments":{"expression":"(1+2)*3"}}\n'
            )
        if tool_results:
            tool_section += "\n[工具执行结果]\n"
            for tr in tool_results:
                if tr.success:
                    tool_section += f"{tr.tool_name}: {tr.result}\n"
                else:
                    tool_section += f"{tr.tool_name} 失败: {tr.error}\n"

        rules_once = (
            "\n[输出要求]\n"
            "只输出 1 个方向（up/down/left/right），且只输出 1 个 JSON 对象（不要输出额外文本）。\n"
            "重要：如果 obs_text 里存在 Recommended Dir 且不为 NONE，优先直接输出它（它已按 expected_score + tie-break 计算）。\n"
            "否则按下列硬规则选择方向：\n"
            "1) 只从 Move Analysis 中 change=yes 的方向里选择（change=no 视为无效动作）\n"
            "2) 在这些方向里，选择 expected_score 最大的方向\n"
            "3) 如果 expected_score 并列，按 tie-break：left > down > right > up\n"
            "4) 如果 Move Analysis 缺失，则从 up/down/left/right 中选一个最可能让棋盘变化的方向（避免连续无变化）\n"
            "输出格式（必须 chunk_size=1）：\n"
            '  {"type":"action_chunk","chunk_size":1,"actions":[{"dir":"left"}]}\n'
        )
        # 重复 2 遍：参考 prompt repetition（让模型更不容易忽略硬规则）。
        rules_twice = rules_once + "\n" + rules_once
        return (
            base_prompt
            + tool_section
            + "以下硬规则重复 2 遍（为了提高遵守率）：\n"
            + rules_twice
            + f"{memory_hint}"
            + f"obs_text:\n{obs_text}\n"
        )

    if game == "super_mario":
        step_summary = _parse_mario_step_summary(obs_text)
        rules_once = (
            "[选择规则]\n"
            "1) OK 优先：若 [Risk Report] 存在 “=> OK” 或仅含 “NEAR_PIT_EDGE@...” 的候选，你必须只从这些候选里选 1 个。\n"
            "2) 无 OK：在剩余候选里尽量避开硬致命 LAND_IN_PIT / HIT_* / TAKEOFF_COLLISION / PIT_TOO_CLOSE_TO_JUMP / PIT_EDGE_TOO_CLOSE；若存在不含 LANDING_DANGER 的候选，尽量不要选带 LANDING_DANGER 的候选。\n"
            "3) 约束：该 skill 对应的 jump_level 必须 <= max_jump_level（按 Candidates 映射）。\n"
            "4) 禁止慢走窗口：\n"
            "   - Urgency=CRITICAL 且 OK 中存在 jump_level>0：禁止选择 jump_level=0。\n"
            "   - Urgency=HIGH/CRITICAL 且 [Threat Analysis] 的 Nearest 是敌人（不是 Pipe/Stairs）且 Distance<=75px，且 [Risk Report] 中存在 jump_level>0 的 OK 候选：禁止选择 jump_level=0。\n"
            "5) 贴脸必死窗口：如果你选择的候选在 [Risk Report] 里包含 PANIC_AHEAD@<n>px 且 n<=8，必须改选不含该标志且非 hard-fatal 的其他候选（如果存在）。\n"
            "6) 省步数（Urgency 低）：若 Urgency=LOW/NONE 且 FAST 在 [Risk Report] 为 OK，则优先选 FAST（不要选 jump_level=0）。\n"
            "7) 并列优先级：STAIRS_JUMP > DENSE_SAFE > SAFE > FAST > STOMP > SLOW_WALK。\n"
            "8) 禁止使用宏动作/确定性规划：每步只能直接输出 jump_level（不要输出多步序列/自定义技能名）。\n"
            "9) 你必须输出 jump_level（0..6）。建议同时输出 skill 便于校验；若输出 skill，则 jump_level 必须与 Candidates 中该 skill 的 jump_level 一致。\n"
        )
        # 重复 2 遍：参考 prompt repetition（让模型在第二遍更容易“回看”并遵守规则）。
        rules_twice = rules_once + "\n" + rules_once
        return (
            "只输出 JSON 对象，不要解释。\n"
            "游戏：Super Mario。\n"
            "目标：向右到达旗帜，避免碰敌人/掉坑。\n"
            "注意：jump_level=0 会继续向右慢走，不是原地等待。\n"
            f"{step_summary}"
            "\n"
            "以下硬规则重复 2 遍（为了提高遵守率）：\n"
            f"{rules_twice}"
            "\n"
            "输出格式（必须 chunk_size=1）：\n"
            "  {\"type\":\"action_chunk\",\"chunk_size\":1,\"actions\":[{\"jump_level\":1,\"skill\":\"SAFE\"}]}\n"
            f"{memory_hint}"
            f"obs_text:\n{obs_text}\n"
        )

    if game == "pokemon_red":
        base = (
            "只输出 JSON 对象，不要解释。\n"
            "游戏：Pokémon Red。\n"
            "目标：推进任务进度（离开房子、遇到博士、领取初始宝可梦等），不要卡在标题界面。\n"
            "提示：你可以一次性输出一个 ActionPack（action_chunk 的 actions 数组包含多个 action），引擎会把它缓存起来，后续步骤直接依次执行，从而减少模型调用次数。\n"
            "提示：你也可以在单步 action 中使用 env_tool（环境内置工具）。env_tool 会被转换成 use_tool(...) 并在环境内部执行一段宏动作，从而显著减少 step 消耗。\n"
            "提示：obs_text 的 Map Info 里会给出 Local Map（以 P 为你当前位置，. 为可走，# 为墙，W 为传送点/楼梯，T 为可对话/告示牌）。\n"
            "最高优先级提示：如果出现 Edge WarpPoint Hint（例如 'try pressing: down'），立即按它建议的方向（如 down）触发换图/离开房子，不要再走到其他 WarpPoint。\n"
            "提示：如果没有 Edge WarpPoint Hint，但有 Exit WarpPoint / Next Keys to Exit WarpPoint，则按 Next Keys 走到出口。\n"
            "提示：如果只看到 Nearest WarpPoint，则优先按 Next Keys to Nearest WarpPoint 走到最近的楼梯/传送点。\n"
            "提示：如果出现 Standing on WarpPoint，且你发现自己在 1f/2f 之间来回切换，说明你在楼梯 WarpPoint 旁边左右横跳；此时应远离楼梯，去找 Exit WarpPoints。\n"
            "决策规则（按优先级）：\n"
            "1) 如果 State: Title：直接输出 keys=[\"a\"] 进入游戏。光标通常默认在 NEW GAME 上，按 a 即可开始。\n"
            "2) 如果 State: Dialog：\n"
            "   - 如果对话内容包含 'SNES' 或 'playing the'（说明在看电视），按 b 退出，例如 keys=[\"b\"]。\n"
            "   - 否则优先用 env_tool continue_dialog() 推进对话（一次 step 内宏推进，最省步）。\n"
            "   - 如果出现 YES/NO 选择框：通常默认 YES 已选中，直接按 a 选择 YES 推进剧情。\n"
            "     例外：如果对话里出现 'nickname'（是否起昵称），优先按 down+a 选择 NO 以避免手动输入名字。\n"
            "3) 如果 State: Field：优先走向能推进里程碑的位置，不要无脑按 a。\n"
            "   - 优先参考决策支持返回的 candidates（尤其是 env_tool move_to / warp_with_warp_point / overworld_map_transition）。\n"
            "   - 如果提供 candidates：你必须从 candidates 里选一个 action；优先选 priority 最大的候选；若同优先级则优先 env_tool。\n"
            "   - 避免反复进出同一栋楼（例如刚离开房子又立刻走回门口按 up 重新进屋）。\n"
            "   - 只有当你确实要换图/进目标建筑时才踩 W（WarpPoint）；否则把 W 视为风险点，先 move_to 到更安全的落脚点再继续。\n"
            "4) 如果你判断“现在不需要输入/需要等待动画结束”，可以输出空数组 keys=[]（等同本步 pass）。\n"
            "5) 如果看到选择框（Selection Box Text 非 N/A）：用 up/down 选择，再按 a 确认。\n"
            "输出格式（示例）：\n"
            "  低级按键：\n"
            "  {\"type\":\"action_chunk\",\"chunk_size\":1,\"actions\":[{\"keys\":[\"start\"]}]}\n"
            "  env_tool（推荐用于长距离移动/对话/战斗）：\n"
            "  {\"type\":\"action_chunk\",\"chunk_size\":1,\"actions\":[{\"env_tool\":{\"name\":\"continue_dialog\",\"args\":{}}}]}\n"
            "  {\"type\":\"action_chunk\",\"chunk_size\":1,\"actions\":[{\"env_tool\":{\"name\":\"overworld_map_transition\",\"args\":{\"direction\":\"north\"}}}]}\n"
            "约束：keys 是按键序列数组，元素只能是 up/down/left/right/a/b/start/select/none/quit。\n"
            "约束：env_tool.name 必须是以下之一：move_to / warp_with_warp_point / overworld_map_transition / interact_with_object / continue_dialog / select_move_in_battle / switch_pkmn_in_battle / run_away / use_item_in_battle。\n"
            "重要：除非你确认需要退出本局，否则不要输出 quit。\n"
        )

        if not tools_enabled:
            return base + f"{memory_hint}" + f"obs_text:\n{obs_text}\n"

        # 决策支持模式：引擎会提供候选动作与进度信息；模型只需在候选中选择一个最终动作。
        if tool_results:
            tool_section = "\n[决策支持结果]\n"
            for tr in tool_results:
                if tr.success:
                    tool_section += f"{tr.result}\n"
                else:
                    tool_section += f"工具执行失败: {tr.error}\n"
            tool_section += (
                "\n【基于候选输出最终动作】\n"
                "决策支持已返回 candidates 列表，每个候选包含：\n"
                "- priority: 优先级（整数，越高越推荐；没有固定上限）\n"
                "- env_tool: 环境宏工具（一次完成复杂操作，如对话推进/换图/战斗，显著节省步数）\n"
                "- keys: 低级按键序列（适合简单移动）\n"
                "- why: 推荐理由\n"
                "\n"
                "【决策目标】：\n"
                "1. 以 progress.next_milestone 为第一目标：选择最可能推进里程碑的一步。\n"
                "2. 默认选择 priority 最高的候选；只有当 why/evidence 明确显示会走回头路、反复进出同一栋楼、或无法推进时，才选择次高候选。\n"
                "3. candidates 中若出现“直接完成 next_milestone”的动作（例如在 ViridianMart 与店员交互、在 OaksLab 与 Oak 交互），应优先选择。\n"
                "4. 若 anti_stuck.note 提示 oscillating/unchanged，避免重复 Last action，优先选能改变地图或位置的候选。\n"
                "5. 需要长距离移动/换图/对话/战斗时，env_tool 通常比 keys 更稳、更省步。\n"
                "\n"
                "【输出格式】：\n"
                "env_tool 示例：\n"
                '  {"type":"action_chunk","chunk_size":1,"actions":[{"env_tool":{"name":"continue_dialog","args":{}}}]}\n'
                "keys 示例：\n"
                '  {"type":"action_chunk","chunk_size":1,"actions":[{"keys":["down"]}]}\n'
                "\n"
                "【重要】：\n"
                "- 只输出一个 action（chunk_size=1）\n"
                "- 要么输出 env_tool，要么输出 keys，不要同时输出两者\n"
                "- 你的输出必须匹配 candidates 中某一项（env_tool.name+args 或 keys 完全一致），不要发明新动作\n"
            )
            return base + tool_section + f"{memory_hint}obs_text:\n{obs_text}\n"

        # tools_enabled=True 但没有 tool_results 的情况：降级为基础提示（不引入任何工具调用协议）。
        return base + f"{memory_hint}" + f"obs_text:\n{obs_text}\n"

    if game == "star_craft":
        from agents.crowsphere.prompt_builder_sc2 import build_sc2_prompt

        return build_sc2_prompt(obs_text=obs_text, game_info=game_info)

    raise ValueError(f"未知游戏: {game}")
