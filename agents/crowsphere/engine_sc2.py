"""SC2 context maintenance for CrowSphereEngine.

This module handles:
- Enemy memory tracking (enemy_seen, enemy_first_seen_time_seconds, etc.)
- last_issued_step for cross-step deduplication
- peak_army tracking
- Decision history recording
"""
from __future__ import annotations

import re
from typing import Any

from agents.crowsphere.sc2_enemy_classifier import classify_enemy_unit


def update_sc2_context(
    *,
    memory: dict[str, dict[str, Any]],
    obs_text: str,
    action_str: str,
) -> None:
    """Update SC2 context after an action is executed.
    
    Args:
        memory: The engine's memory dictionary (will be mutated in place).
        obs_text: The preprocessed observation text.
        action_str: The action string that was executed.
    """
    game = "star_craft"
    ctx = memory.get(game, {
        "history": [],
        "peak_army": 0,
        "step": 0,
        "last_attack": False,
    })
    ctx["step"] = ctx.get("step", 0) + 1

    # Parse current army supply
    army_match = re.search(r"Army supply:\s*(\d+)", obs_text, re.IGNORECASE)
    army = int(army_match.group(1)) if army_match else 0

    # Parse game time (for enemy memory)
    game_time_seconds = _parse_game_time_seconds(obs_text)

    # Update enemy memory
    _update_enemy_memory(ctx, obs_text, game_time_seconds)

    # Update last_issued_step for cross-step deduplication
    _update_last_issued_step(ctx, action_str)

    # Update peak_army
    if army > ctx.get("peak_army", 0):
        ctx["peak_army"] = army

    # Detect attack/retreat/scout from action_str
    is_attack = "MULTI-ATTACK" in action_str
    if is_attack:
        ctx["last_attack_step"] = ctx["step"]
    is_retreat = "MULTI-RETREAT" in action_str
    if is_retreat:
        ctx["last_retreat_step"] = ctx["step"]
    if "SCOUTING " in action_str:
        ctx["last_scout_step"] = ctx["step"]
        # Track how many scouting commands we have issued. This helps decide whether
        # "no enemy info" means "we didn't scout" vs "the env does not expose it".
        ctx["scout_count"] = int(ctx.get("scout_count", 0) or 0) + 1
    decision = "ATTACK" if is_attack else "DEVELOP"

    # Record decision history (only on change or key moments)
    history = ctx.get("history", [])
    should_record = (
        not history or  # First record
        history[-1].get("decision") != decision or  # Decision changed
        ctx["step"] % 20 == 0  # Every 20 steps
    )
    if should_record:
        history.append({
            "step": ctx["step"],
            "decision": decision,
            "army": army,
        })
        # Keep only last 5 records
        ctx["history"] = history[-5:]

    ctx["last_attack"] = is_attack
    memory[game] = ctx


def maybe_reset_sc2_episode(
    *,
    memory: dict[str, dict[str, Any]],
    game_info: dict[str, Any],
    obs_text: str | None = None,
) -> None:
    """在新一局开始时重置 SC2 跨步记忆，避免跨地图污染策略。

    官方评测会在同一个 agent 实例上连续跑多局（3 张地图）。如果我们不做 reset：
    - 上一局记录的 enemy_seen/step/last_issued_step 会影响下一局（例如误判早期有远程威胁 → 乱补 Gateway）
    - 会导致扩张/进攻节奏在不同 map/seed 下不稳定

    现实约束：
    - 部分 REMOTE runner 的 game_info 不提供 episode_index / map_name / map_idx。
    - 因此这里支持两种 reset 信号：
      1) 优先使用 game_info.episode_index（如果存在）
      2) 否则从 obs_text 的 Game time 回退（例如 12:00 -> 00:00）推断新一局
    """
    if not isinstance(game_info, dict):
        return
    raw_episode = game_info.get("episode_index")
    try:
        cur_episode = int(raw_episode) if raw_episode is not None else None
    except Exception:
        cur_episode = None

    def _resolve_map_name(info: dict[str, Any], *, episode_index: int | None) -> str:
        """Best-effort SC2 map name resolver.

        REMOTE runners sometimes omit `map_name/map_idx` in game_info. Official evaluation
        rotates 3 maps across 3 episodes with base_map_idx=0:
        - 0: Ancient Cistern LE
        - 1: LastFantasyAIE
        - 2: Flat64
        """
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
        raw_idx = info.get("map_idx")
        if raw_idx is None:
            raw_idx = nested.get("map_idx")
        try:
            map_idx = int(raw_idx) if raw_idx is not None else None
        except Exception:
            map_idx = None
        if isinstance(map_idx, int) and default_pool:
            return str(default_pool[int(map_idx) % len(default_pool)])
        if isinstance(episode_index, int) and default_pool:
            return str(default_pool[int(episode_index) % len(default_pool)])
        return ""

    # When episode_index is missing, we maintain our own 0-based episode counter.
    # This is used to infer the map name under official 3-map rotation.
    map_pool = ("Ancient Cistern LE", "LastFantasyAIE", "Flat64")

    if cur_episode is None:
        ctx = memory.get("star_craft")
        if not isinstance(ctx, dict):
            ctx = {}
        last_time = ctx.get("_last_game_time_seconds")
        try:
            last_time_i = int(last_time) if last_time is not None else None
        except Exception:
            last_time_i = None
        cur_time_i: int | None = None
        if obs_text:
            try:
                cur_time_i = int(_parse_game_time_seconds(str(obs_text or "")))
            except Exception:
                cur_time_i = None
        episode_counter = ctx.get("_episode_index")
        try:
            episode_counter_i = int(episode_counter) if episode_counter is not None else 0
        except Exception:
            episode_counter_i = 0

        is_new_episode = False
        if cur_time_i is not None and last_time_i is not None:
            # A big backward jump in game time almost certainly means a new episode.
            if int(cur_time_i) < int(last_time_i) - 60:
                is_new_episode = True
            # A very small time after a long run is also a strong reset signal.
            if int(cur_time_i) <= 20 and int(last_time_i) >= 120:
                is_new_episode = True

        if is_new_episode:
            episode_counter_i += 1
            memory["star_craft"] = {
                "history": [],
                "peak_army": 0,
                "step": 0,
                "last_attack": False,
                "_episode_index": int(episode_counter_i),
                "_map_name": str(map_pool[int(episode_counter_i) % len(map_pool)]) if map_pool else "",
                "_last_game_time_seconds": int(cur_time_i or 0),
            }
            return

        # Initialize baseline fields so the prompt builder can always read `_map_name`.
        if "_episode_index" not in ctx:
            ctx["_episode_index"] = int(episode_counter_i)
        if "_map_name" not in ctx and map_pool:
            ctx["_map_name"] = str(map_pool[int(episode_counter_i) % len(map_pool)])
        if cur_time_i is not None:
            ctx["_last_game_time_seconds"] = int(cur_time_i)
        memory["star_craft"] = ctx
        return

    map_name = _resolve_map_name(game_info, episode_index=cur_episode)

    ctx = memory.get("star_craft")
    if not isinstance(ctx, dict):
        ctx = {}
    last = ctx.get("_episode_index")
    try:
        last_episode = int(last) if last is not None else None
    except Exception:
        last_episode = None

    if last_episode is None:
        ctx["_episode_index"] = cur_episode
        ctx["_map_name"] = map_name
        memory["star_craft"] = ctx
        return

    if cur_episode != last_episode:
        memory["star_craft"] = {
            "history": [],
            "peak_army": 0,
            "step": 0,
            "last_attack": False,
            "_episode_index": cur_episode,
            "_map_name": map_name,
        }


def _parse_game_time_seconds(obs_text: str) -> int:
    """Parse game time from observation text."""
    time_match = re.search(r"Game time:\s*(\d+:\d+)", obs_text, re.IGNORECASE)
    if not time_match:
        return 0
    mmss = time_match.group(1)
    m = re.match(r"(\d+):(\d+)", mmss)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return 0


def _update_enemy_memory(
    ctx: dict[str, Any],
    obs_text: str,
    game_time_seconds: int,
) -> None:
    """Update enemy memory based on current observation.
    
    Enemy memory persists - once we've seen an enemy unit, we remember it.
    This prevents "attack because we don't see any enemies" mistakes.
    """
    enemy_seen = ctx.get("enemy_seen") if isinstance(ctx.get("enemy_seen"), dict) else {}
    enemy_last_seen_step = (
        ctx.get("enemy_last_seen_step")
        if isinstance(ctx.get("enemy_last_seen_step"), dict)
        else {}
    )
    enemy_last_seen_time = (
        ctx.get("enemy_last_seen_time_seconds")
        if isinstance(ctx.get("enemy_last_seen_time_seconds"), dict)
        else {}
    )
    enemy_first_seen_step = (
        ctx.get("enemy_first_seen_step")
        if isinstance(ctx.get("enemy_first_seen_step"), dict)
        else {}
    )
    enemy_first_seen_time = (
        ctx.get("enemy_first_seen_time_seconds")
        if isinstance(ctx.get("enemy_first_seen_time_seconds"), dict)
        else {}
    )

    found_any = False
    found_combat = False
    for match in re.finditer(
        r"Enemy unittypeid\.([a-z0-9_]+)\s*:\s*(\d+)",
        obs_text,
        re.IGNORECASE,
    ):
        unit_name = match.group(1).lower()
        count = int(match.group(2))
        if count <= 0:
            continue
        found_any = True
        # Only treat *combat* sightings as "enemy info supported" for attack-safety logic.
        # Non-combat sightings (e.g. overlord) are not enough to conclude the enemy is "safe".
        if classify_enemy_unit(unit_name) not in {"noncombat", "building"}:
            found_combat = True
            ctx["last_enemy_combat_seen_step"] = ctx["step"]
            if game_time_seconds:
                ctx["last_enemy_combat_seen_time_seconds"] = game_time_seconds
        prev = int(enemy_seen.get(unit_name, 0) or 0)
        enemy_seen[unit_name] = max(prev, count)
        enemy_last_seen_step[unit_name] = ctx["step"]
        if game_time_seconds:
            enemy_last_seen_time[unit_name] = game_time_seconds
            if unit_name not in enemy_first_seen_time:
                enemy_first_seen_time[unit_name] = game_time_seconds
        if unit_name not in enemy_first_seen_step:
            enemy_first_seen_step[unit_name] = ctx["step"]

    # If we ever see *combat* `Enemy unittypeid.*` lines, remember that the
    # observation stream can expose enemy combat info. Seeing only non-combat
    # units (e.g. overlord) is not enough for "attack is safe" decisions.
    if found_combat:
        ctx["enemy_info_supported"] = True

    ctx["enemy_seen"] = enemy_seen
    ctx["enemy_last_seen_step"] = enemy_last_seen_step
    ctx["enemy_last_seen_time_seconds"] = enemy_last_seen_time
    ctx["enemy_first_seen_step"] = enemy_first_seen_step
    ctx["enemy_first_seen_time_seconds"] = enemy_first_seen_time


def _update_last_issued_step(ctx: dict[str, Any], action_str: str) -> None:
    """Update last_issued_step for cross-step deduplication.
    
    This prevents repeatedly issuing the same build/train command
    when the observation hasn't updated yet.
    """
    last_issued_step = (
        ctx.get("last_issued_step") if isinstance(ctx.get("last_issued_step"), dict) else {}
    )
    for line in str(action_str or "").splitlines():
        m = re.match(r"^\s*\d+\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        token = m.group(1).strip().upper()
        if not token or token == "EMPTY ACTION":
            continue
        last_issued_step[token] = ctx["step"]
    ctx["last_issued_step"] = last_issued_step


def get_sc2_context(memory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Get the current SC2 context from memory.
    
    Returns a copy to prevent accidental mutation.
    """
    return dict(memory.get("star_craft", {}))
