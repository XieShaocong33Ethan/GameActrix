from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

from agents.crowsphere.sc2_enemy_classifier import aggregate_enemy_units


def _parse_sc2_state(obs_str: str) -> dict[str, Any]:
    """从obs_str解析SC2关键状态，供prompt展示（历史遗留，当前不用于候选生成）。"""
    state: dict[str, Any] = {}

    # StarCraftObs.to_text() 的关键字段基本都是类似：
    # "- Mineral: 123", "- Nexus count: 1"
    def _parse_int(pattern: str, default: int = 0) -> int:
        m = re.search(pattern, obs_str, re.IGNORECASE | re.MULTILINE)
        if not m:
            return default
        try:
            return int(m.group(1))
        except Exception:
            return default

    # 资源与人口
    state["mineral"] = _parse_int(r"(?:^|\n)\s*-\s*Mineral:\s*(\d+)")
    state["gas"] = _parse_int(r"(?:^|\n)\s*-\s*Gas:\s*(\d+)")
    state["supply_left"] = _parse_int(r"(?:^|\n)\s*-\s*Supply left:\s*(-?\d+)")
    state["supply_cap"] = _parse_int(r"(?:^|\n)\s*-\s*Supply cap:\s*(\d+)")
    state["supply_used"] = _parse_int(r"(?:^|\n)\s*-\s*Supply used:\s*(\d+)")

    # 我方建筑/单位（只取做宏观决策最关键的几项）
    state["nexus"] = _parse_int(r"(?:^|\n)\s*-\s*Nexus count:\s*(\d+)")
    state["pylon"] = _parse_int(r"(?:^|\n)\s*-\s*Pylon count:\s*(\d+)")
    state["gateway"] = _parse_int(r"(?:^|\n)\s*-\s*Gateway count:\s*(\d+)")
    state["assimilator"] = _parse_int(r"(?:^|\n)\s*-\s*Gas buildings count:\s*(\d+)")
    state["assimilator_constructing"] = _parse_int(
        r"(?:^|\n)\s*-\s*Constructing gas buildings count:\s*(\d+)"
    )
    state["cybernetics_core"] = _parse_int(r"(?:^|\n)\s*-\s*Cybernetics\s*core count:\s*(\d+)")
    state["robotics_facility"] = _parse_int(r"(?:^|\n)\s*-\s*Robotics\s*facility count:\s*(\d+)")

    state["probe"] = _parse_int(r"(?:^|\n)\s*-\s*Probe count:\s*(\d+)")
    # 某些版本 env 同时给 Worker supply / Probe count；用 probe 为主。
    state["worker_supply"] = _parse_int(r"(?:^|\n)\s*-\s*Worker supply:\s*(\d+)")
    state["army"] = _parse_int(r"(?:^|\n)\s*-\s*Army supply:\s*(\d+)")
    state["zealot"] = _parse_int(r"(?:^|\n)\s*-\s*Zealot count:\s*(\d+)")
    state["stalker"] = _parse_int(r"(?:^|\n)\s*-\s*Stalker count:\s*(\d+)")
    state["immortal"] = _parse_int(r"(?:^|\n)\s*-\s*Immortal count:\s*(\d+)")

    # 建造/生产队列（用于减少“无意义重复点建造/训练”）
    state["pylon_constructing"] = _parse_int(r"(?:^|\n)\s*-\s*Constructing pylon count:\s*(\d+)")
    state["gateway_constructing"] = _parse_int(r"(?:^|\n)\s*-\s*Constructing gateway count:\s*(\d+)")
    state["nexus_constructing"] = _parse_int(r"(?:^|\n)\s*-\s*Constructing nexus count:\s*(\d+)")
    state["cybernetics_core_constructing"] = _parse_int(
        r"(?:^|\n)\s*-\s*Constructing cybernetics\s*core count:\s*(\d+)"
    )
    state["robotics_facility_constructing"] = _parse_int(
        r"(?:^|\n)\s*-\s*Constructing robotics\s*facility count:\s*(\d+)"
    )
    state["probe_producing"] = _parse_int(r"(?:^|\n)\s*-\s*Producing probe count:\s*(\d+)")
    state["zealot_producing"] = _parse_int(r"(?:^|\n)\s*-\s*Producing zealot count:\s*(\d+)")
    state["stalker_producing"] = _parse_int(r"(?:^|\n)\s*-\s*Producing stalker count:\s*(\d+)")
    state["immortal_producing"] = _parse_int(r"(?:^|\n)\s*-\s*Producing immortal count:\s*(\d+)")

    # 解析敌方单位（通用：支持 bot_race != Zerg）
    enemy_units: dict[str, int] = {}
    for m in re.finditer(r"Enemy unittypeid\.([a-z0-9_]+)\s*:\s*(\d+)", obs_str, re.IGNORECASE):
        unit_name = m.group(1).lower()
        count = int(m.group(2))
        if count <= 0:
            continue
        enemy_units[unit_name] = count
    agg = aggregate_enemy_units(enemy_units)
    state["enemy_visible"] = agg["total_combat"]
    state["enemy_ground_ranged"] = agg["ground_ranged"]
    state["enemy_ground_melee"] = agg["ground_melee"]
    state["enemy_air"] = agg["air"]
    state["enemy_units"] = enemy_units

    # 解析游戏时间
    time_match = re.search(r"(?:At\s+|Game time:\s*)(\d+:\d+)", obs_str, re.IGNORECASE)
    state["time"] = time_match.group(1) if time_match else "00:00"

    return state


def _detect_sc2_events(state: dict[str, Any], context: dict[str, Any]) -> list[str]:
    """检测当前发生的事件（历史遗留，当前未直接用于决策）。"""
    events = []

    # 检测人口卡住
    if state.get("supply_left", 0) <= 0:
        events.append("SUPPLY_BLOCKED")

    # 检测生产设施不足
    if state.get("gateway", 0) < 3:
        events.append("LOW_PRODUCTION")

    # 检测军队准备好
    army = state.get("army", 0)
    enemy_visible = state.get("enemy_visible", 0)
    if army >= 30 and enemy_visible == 0:
        events.append("ARMY_READY")

    # 检测正在交战
    if enemy_visible > 0 and context.get("last_attack", False):
        events.append("UNDER_ATTACK")

    # 检测军队损失（相比peak下降超过30%）
    peak = context.get("peak_army", 0)
    if peak > 10 and army < peak * 0.7:
        events.append("ARMY_LOSS")

    # 默认：发展中
    if not events:
        events.append("DEVELOPING")

    return events


def _build_sc2_context_hint(context: dict[str, Any]) -> str:
    """构建决策历史提示（历史遗留）。"""
    history = context.get("history", [])
    if not history:
        return ""

    lines = ["决策历史："]
    for h in history[-5:]:
        step = h.get("step", 0)
        decision = h.get("decision", "?")
        army = h.get("army", 0)
        lines.append(f"- Step {step}: {decision}, Army={army}")

    return "\n".join(lines) + "\n"


def build_sc2_prompt(*, obs_text: str, game_info: dict[str, Any]) -> str:
    """构建 StarCraft II 的提示词。

    目标：
    - 提示词：用“通用硬规则”明确 enemy_total_visible==0 不能作为进攻依据
    - 候选动作：门控更严格，不满足门槛时不提供 attack_* / retreat_*（更稳）
    """
    num_actions = int(game_info.get("num_actions", 0))
    num_actions = max(1, num_actions)
    player_race = game_info.get("player_race", "Protoss")
    enemy_race = game_info.get("enemy_race", "Zerg")
    def _resolve_map_name(info: dict[str, Any]) -> str:
        # Official game_info varies across runners; try a few common locations.
        #
        # NOTE: In some REMOTE runs, `map_name/map_idx` is missing from game_info.
        # However, official evaluation rotates 3 maps in a fixed order across 3 episodes
        # (base_map_idx=0): [Ancient Cistern LE, LastFantasyAIE, Flat64].
        # We can safely infer the map name from episode_index as a last-resort fallback.
        default_pool = ("Ancient Cistern LE", "LastFantasyAIE", "Flat64")

        raw = info.get("map_name")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        raw = info.get("map")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        nested = info.get("info") if isinstance(info.get("info"), dict) else {}
        for key in ("map_name", "map"):
            v = nested.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Fallback: map_idx (preferred over episode_index).
        raw_idx = info.get("map_idx")
        if raw_idx is None:
            raw_idx = nested.get("map_idx")
        try:
            map_idx = int(raw_idx) if raw_idx is not None else None
        except Exception:
            map_idx = None
        if isinstance(map_idx, int) and default_pool:
            return str(default_pool[int(map_idx) % len(default_pool)])

        # Final fallback: infer from episode_index (official rotation order).
        raw_ep = info.get("episode_index")
        if raw_ep is None:
            raw_ep = nested.get("episode_index")
        try:
            ep = int(raw_ep) if raw_ep is not None else None
        except Exception:
            ep = None
        if isinstance(ep, int) and default_pool:
            return str(default_pool[int(ep) % len(default_pool)])

        return ""

    map_name = _resolve_map_name(game_info)
    if not map_name:
        # Some REMOTE runners omit map metadata in game_info. If the engine has inferred
        # the current map via cross-step memory, use it here so we can apply map-specific
        # gating (Flat64 rush / LastFantasy thresholds).
        ctx = game_info.get("sc2_context") if isinstance(game_info.get("sc2_context"), dict) else {}
        ctx_map = ctx.get("_map_name")
        if isinstance(ctx_map, str) and ctx_map.strip():
            map_name = ctx_map.strip()
    map_name_norm = map_name.strip().lower()
    is_flat64 = map_name_norm == "flat64"
    is_lastfantasy = map_name_norm == "lastfantasyaie"

    obs_str = str(game_info.get("obs_str") or obs_text or "")

    # action_dict 可能是 dict 或字符串；解析出允许动作集合（用于候选生成的防呆）。
    allowed_actions: set[str] | None = None
    raw_action_dict = game_info.get("action_dict")
    parsed_action_dict: dict[str, Any] | None = None
    if isinstance(raw_action_dict, dict):
        parsed_action_dict = raw_action_dict
    elif isinstance(raw_action_dict, str):
        text = raw_action_dict.strip()
        if text:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                parsed_action_dict = parsed
    if parsed_action_dict:
        allowed_actions = {str(k).upper() for k in parsed_action_dict.keys()}

    from agents.crowsphere.sc2_macro_policy import postprocess_actions as _sc2_postprocess_actions
    from agents.crowsphere.sc2_strategy_toolbox import parse_game_state as _parse_sc2_policy_state

    state = _parse_sc2_policy_state(obs_str)

    time_str = str(state.get("game_time") or "00:00")
    time_seconds = int(state.get("game_time_seconds") or 0)
    mineral = int(state.get("mineral") or 0)
    gas = int(state.get("gas") or 0)
    supply_used = int(state.get("supply_used") or 0)
    supply_cap = int(state.get("supply_cap") or 0)
    supply_left = int(state.get("supply_left") or 0)

    nexus_total = int(state.get("nexus_count") or 0) + int(state.get("nexus_constructing") or 0)
    pylon_total = int(state.get("pylon_count") or 0) + int(state.get("pylon_constructing") or 0)
    gateway_total = int(state.get("gateway_count") or 0) + int(state.get("gateway_constructing") or 0)
    assimilator_total = int(state.get("gas_buildings_count") or 0) + int(state.get("gas_buildings_constructing") or 0)
    core_total = int(state.get("cybernetics_core_count") or 0) + int(state.get("cybernetics_core_constructing") or 0)
    robo_total = int(state.get("robotics_facility_count") or 0) + int(state.get("robotics_facility_constructing") or 0)

    probe = int(state.get("probe_count") or 0) or int(state.get("worker_supply") or 0)
    probe_producing = int(state.get("probe_producing") or 0)
    army_supply = int(state.get("army_supply") or 0)
    zealot = int(state.get("zealot_count") or 0)
    stalker = int(state.get("stalker_count") or 0)
    immortal = int(state.get("immortal_count") or 0)

    enemy_total = int(state.get("enemy_total_visible") or 0)
    enemy_ranged = int(state.get("enemy_ground_ranged_threat", state.get("enemy_ranged_threat", 0)) or 0)
    enemy_melee = int(state.get("enemy_ground_melee_threat") or 0)
    enemy_air = int(state.get("enemy_air_threat") or 0)

    state_summary = (
        f"时间={time_str} (time_seconds={time_seconds}) | 矿={mineral} 气={gas} | 人口={supply_used}/{supply_cap} (剩余{supply_left})\n"
        f"我方：Nexus={nexus_total} Pylon={pylon_total} Gateway={gateway_total} Assimilator={assimilator_total}\n"
        f"科技：Core={core_total} Robo={robo_total}\n"
        f"军队：ArmySupply={army_supply} (Zealot={zealot} Stalker={stalker} Immortal={immortal})\n"
        f"敌方可见：Total={enemy_total} (Melee={enemy_melee} Ranged={enemy_ranged} Air={enemy_air})"
    )

    sc2_context = game_info.get("sc2_context") if isinstance(game_info.get("sc2_context"), dict) else {}
    ctx_step = int(sc2_context.get("step") or 0)
    peak_army_ctx = int(sc2_context.get("peak_army") or 0)
    last_attack_step = int(sc2_context.get("last_attack_step") or -999999)
    last_retreat_step = int(sc2_context.get("last_retreat_step") or -999999)
    steps_since_attack = ctx_step - last_attack_step if last_attack_step > 0 else None
    steps_since_retreat = ctx_step - last_retreat_step if last_retreat_step > 0 else None

    # ----------------------------
    # 候选动作门控（更稳）
    # ----------------------------
    # 关键原则（通用）：enemy_total_visible==0 只代表“当前视野没有看到”，不能作为“可以进攻”的依据。
    # 因此 attack_* 是否提供，只看“兵力/时间阈值”等硬条件，而不把 enemy_total_visible==0 当作放行条件。
    lastfantasy_attack_min_time_seconds = 360
    # LastFantasyAIE：地图更大、且对手更可能“藏建筑拖时间”。过早小规模出门会导致 sweep 分组=1，
    # 收尾效率很差，容易在 400 step（~12 分钟）上限下超时。
    # 因此把最小出门兵力抬到 60，确保环境侧 bot 会启用 3 组 sweep（supply_army>=60）。
    lastfantasy_attack_min_army_supply = 60
    # 标准图：如果等到 09:00 才出门，留给“找隐藏建筑/清尾巴”的时间太少，容易超时。
    # 将时间门槛提前到 07:00（420s），但仍要求 army_supply>=60。
    standard_attack_min_time_seconds = 420
    # 标准图：线上复现表明 army_supply 很难稳定到 60（常见峰值 36~44），导致 attack_* 永远不出现。
    # 将阈值下调到 36：足够形成“可打得动”的一波，同时留出时间收尾，避免 400 step 超时。
    standard_attack_min_army_supply = 36
    rush_attack_min_time_seconds = 240
    rush_attack_min_army_supply = 40

    if is_lastfantasy:
        allow_attack = (time_seconds >= lastfantasy_attack_min_time_seconds and army_supply >= lastfantasy_attack_min_army_supply)
    elif is_flat64:
        # Rush 图：不因为 “看不到敌人” 就出门；必须兵力成型。
        allow_attack = (time_seconds >= rush_attack_min_time_seconds and army_supply >= rush_attack_min_army_supply)
    else:
        allow_attack = (time_seconds >= standard_attack_min_time_seconds and army_supply >= standard_attack_min_army_supply)

    # retreat_* 的提供也做门控：避免“没出门先撤退 / 连续撤退”这类低收益动作。
    army_loss = bool(peak_army_ctx >= 20 and army_supply < int(peak_army_ctx * 0.6) and enemy_total > 0)
    retreat_off_cooldown = (steps_since_retreat is None) or (steps_since_retreat >= 30)
    recently_attacked = steps_since_attack is not None and steps_since_attack <= 60
    allow_retreat = bool(retreat_off_cooldown and (army_loss or recently_attacked))

    base_actions = ["EMPTY ACTION"] * num_actions
    policy_base = {
        "macro_policy": "two_base",
        "disable_static_defense": True,
        "enable_retreat": True,
        "rush_map_expand_min_army_supply": 4,
        # 注意：这里的 attack_* 配置仅供候选生成器携带上下文（真正是否出门由模型选 attack_*/retreat_* 决定）。
        # 宏策略本身不在这里自动出兵，避免“代码替模型决定动作”。
        "attack_min_time_seconds": standard_attack_min_time_seconds,
        "attack_army_threshold": standard_attack_min_army_supply,
        "force_attack_army_supply": 120,
        # LastFantasyAIE 在 400 step 上限（约 12 分钟）下更容易拖到超时：少采一点矿、早一点铺开产能更稳。
        "probe_target_cap": 44 if is_lastfantasy else 48,
        "zealot_queue_depth": 2 if is_lastfantasy else 1,
        "air_cleanup_min_time_seconds": 360 if is_lastfantasy else 420,
        "enable_warpgate_research": bool(is_lastfantasy),
    }

    def _inject(actions: list[str], token: str) -> list[str]:
        out = list(actions)
        if not out:
            return out
        try:
            i = out.index("EMPTY ACTION")
        except ValueError:
            i = len(out) - 1
        out[i] = token
        return out

    candidates: list[tuple[str, str, list[str]]] = []
    tech_plans = ("immortal",) if is_lastfantasy else ("immortal", "zealot_only")
    for tech_plan in tech_plans:
        pol = dict(policy_base, tech_plan=tech_plan)
        develop = _sc2_postprocess_actions(
            state=state,
            actions=base_actions,
            train=True,
            build_defense=False,
            military="defend",  # 不让程序自动出兵；由模型选 attack/retreat 候选来决定。
            allowed_actions=allowed_actions,
            policy_config=pol,
            context=sc2_context,
        )
        dev_tuple = (f"develop_{tech_plan}", f"发展/运营（{tech_plan}）", develop)
        atk_tuple = (
            f"attack_{tech_plan}",
            f"进攻（{tech_plan}）：插入 MULTI-ATTACK",
            _inject(develop, "MULTI-ATTACK"),
        )
        ret_tuple = (
            f"retreat_{tech_plan}",
            f"撤退回防（{tech_plan}）：插入 MULTI-RETREAT",
            _inject(develop, "MULTI-RETREAT"),
        )

        plan_allows_attack = allow_attack
        if "BUILD NEXUS" in develop:
            plan_allows_attack = False
        if is_flat64 and tech_plan == "immortal" and robo_total <= 0 and immortal <= 0:
            plan_allows_attack = False

        if is_lastfantasy:
            # 大图：先攒到关键兵力再出门，避免小规模送兵导致 12 分钟内打不死。
            if allow_attack:
                candidates.append(atk_tuple)
            candidates.append(dev_tuple)
            if allow_retreat:
                candidates.append(ret_tuple)
        else:
            candidates.append(dev_tuple)
            if plan_allows_attack:
                candidates.append(atk_tuple)
            if allow_retreat:
                candidates.append(ret_tuple)

    cand_lines: list[str] = []
    for key, desc, actions in candidates:
        cand_lines.append(f"- {key}: {desc}\n  actions={json.dumps(actions, ensure_ascii=False)}")
    candidates_block = "\n".join(cand_lines)

    # ----------------------------
    # 局面提示（只提示，不改动作）
    # ----------------------------
    danger_rush_small_army = bool(is_flat64 and enemy_total > 0 and army_supply < rush_attack_min_army_supply)
    hint_lines: list[str] = []
    if danger_rush_small_army:
        hint_lines.append(
            f"- 你在 rush 图（Flat64）且敌军可见>0 且 army_supply={army_supply}<{rush_attack_min_army_supply}：优先 develop_* 防守与补兵，不要贸然出门。"
        )
    if army_loss and retreat_off_cooldown:
        hint_lines.append(
            f"- 你军队明显损失：peak_army={peak_army_ctx} 当前 army_supply={army_supply}，且敌军仍可见：若候选中存在 retreat_*，建议选一次回防。"
        )
    elif army_loss and (not retreat_off_cooldown):
        hint_lines.append(
            f"- 你军队仍在损失但刚撤退过（steps_since_retreat={steps_since_retreat}）：不要再连续撤退；优先 develop_* 补兵。"
        )
    computed_hints = ("\n【当前局面提示】\n" + "\n".join(hint_lines) + "\n") if hint_lines else ""

    prompt = (
        "只输出 JSON 对象，不要解释。\n"
        f"游戏：StarCraft II（{player_race} vs {enemy_race}）\n"
        f"地图：{map_name or 'Unknown'}\n"
        "\n"
        "目标：赢（最终 Victory）。\n"
        "\n"
        f"当前关键状态（已从 obs_text 提取）：\n{state_summary}\n"
        "关键变量（用于你选择候选）：\n"
        f"- time_seconds={time_seconds} mineral={mineral} gas={gas} supply_left={supply_left}\n"
        f"- army_supply={army_supply} enemy_total_visible={enemy_total} enemy_ranged={enemy_ranged} enemy_air={enemy_air}\n"
        "跨步记忆（用于判断是否需要撤退/是否一直在进攻）：\n"
        f"- step={ctx_step} peak_army={peak_army_ctx} steps_since_attack={steps_since_attack} steps_since_retreat={steps_since_retreat}\n"
        "\n"
        "候选动作（从下列候选中选 1 个；不要发明新动作；不要修改候选数组内容）：\n"
        f"{candidates_block}\n"
        "\n"
        f"{computed_hints}"
        "硬规则（所有地图都适用）：\n"
        "1) enemy_total_visible==0 只表示你当前视野没有看到敌人，不代表安全；不能因为 enemy_total_visible==0 选择 attack_*。\n"
        "2) 如果候选里没有 attack_*，你必须在 develop_* 中选 1 个。\n"
        "3) retreat_* 只在“你最近进攻过（steps_since_attack 很小）”或“军队明显损失且敌军仍可见”时才选；不要连续撤退。\n"
        "\n"
        "门槛说明（用于理解为何候选缺失）：\n"
        f"- LastFantasyAIE：time_seconds>={lastfantasy_attack_min_time_seconds} 且 army_supply>={lastfantasy_attack_min_army_supply} 才会提供 attack_*。\n"
        f"- Rush 图（Flat64）：time_seconds>={rush_attack_min_time_seconds} 且 army_supply>={rush_attack_min_army_supply} 才会提供 attack_*（不会因为 enemy_total_visible==0 放行）。\n"
        f"- 其他标准图：time_seconds>={standard_attack_min_time_seconds} 且 army_supply>={standard_attack_min_army_supply} 才会提供 attack_*。\n"
        "- 不要一直只 develop：当 attack_* 出现在候选里时，通常应在合适时机选 attack_* 推进终局。\n"
        "\n"
        "输出要求：\n"
        "- 你必须输出 1 个 action_chunk，chunk_size=1。\n"
        f"- actions[0].actions 必须是长度严格等于 {num_actions} 的字符串数组。\n"
        "- 每个字符串必须来自 action_dict 的 key（候选里已经是合法动作名，直接复制即可）。\n"
        "\n"
        "输出格式：\n"
        "  {\"type\":\"action_chunk\",\"chunk_size\":1,\"actions\":[{\"actions\": <复制你选中的候选 actions 数组>}]}\n"
    )

    # 对小模型：重复同一份提示词有时能提升“遵循格式/规则”的概率。
    # 可用环境变量覆盖：CROWSPHERE_SC2_PROMPT_REPEAT=1 表示不重复。
    repeat = 2
    env_repeat = (os.getenv("CROWSPHERE_SC2_PROMPT_REPEAT") or "").strip()
    if env_repeat:
        try:
            repeat = int(env_repeat)
        except ValueError:
            repeat = 2
    repeat = max(1, min(3, repeat))
    return ("\n\n---\n\n".join([prompt] * repeat)) if repeat > 1 else prompt
