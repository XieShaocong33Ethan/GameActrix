from __future__ import annotations

import re
from typing import Any

from agents.crowsphere.logging_jsonl import JsonlLogger


_NEXT_THREATS_RE = re.compile(r"^Next threats:\s*(.+)$", re.MULTILINE)
_GOOMBA_RE = re.compile(r"^Goomba:\s*(.*)$", re.MULTILINE)
_KOOPA_RE = re.compile(r"^Koopa:\s*(.*)$", re.MULTILINE)


def mario_use_model(config: dict[str, Any]) -> bool:
    """Whether to use the VLM for Mario (default: False for robustness)."""
    mario_cfg = config.get("super_mario") if isinstance(config.get("super_mario"), dict) else {}
    return bool(mario_cfg.get("use_model", False))


def _count_visible_enemies(obs_str: str) -> int:
    """A cheap proxy for "enemy cluster": count listed Goomba/Koopa positions in [Objects]."""
    text = str(obs_str or "")
    count = 0
    m = _GOOMBA_RE.search(text)
    if m:
        count += str(m.group(1) or "").count("(")
    m = _KOOPA_RE.search(text)
    if m:
        count += str(m.group(1) or "").count("(")
    return count


def _has_next_threats_line(obs_str: str) -> bool:
    return bool(_NEXT_THREATS_RE.search(str(obs_str or "")))


def _pick_skill_from_candidates(
    *,
    obs_str: str,
    candidates: dict[str, int],
    risks: dict[str, list[str]],
    max_level: int,
    urgency: str | None,
    nearest_name: str | None,
    nearest_dist: int | None,
) -> tuple[str, int, str]:
    """
    Deterministically pick a Mario jump skill from Candidates + Risk Report.

    Design:
    - Prefer progress (FAST) when safe.
    - When enemies are close (HIGH/CRITICAL), avoid staying on the ground too long.
    - When multiple enemies are present, avoid STOMP even if it's "OK" in the risk table;
      jumping over a cluster is usually safer than landing on one enemy.
    """
    from agents.crowsphere.adapters import super_mario as mario_adapter

    def _sentinel_true(key: str) -> bool:
        return bool(re.search(rf"\\b{re.escape(key)}\\s*:\\s*True\\b", obs_str, re.IGNORECASE))

    on_platform = _sentinel_true("on_platform")
    enemy_below_ahead = _sentinel_true("enemy_below_ahead")

    enemy_near = (
        nearest_name is not None
        and nearest_name not in {"PIPE", "STAIRS"}
        and urgency in {"HIGH", "CRITICAL"}
        and isinstance(nearest_dist, int)
        and nearest_dist <= 85
    )
    cluster = enemy_near and (_has_next_threats_line(obs_str) or _count_visible_enemies(obs_str) >= 2)

    def _flags_for(skill: str) -> list[str]:
        fl = risks.get(skill)
        return list(fl) if isinstance(fl, list) else []

    def _is_hard_fatal(skill: str) -> bool:
        return mario_adapter._has_hard_fatal_flag(_flags_for(skill))

    def _soft_count(skill: str) -> int:
        # Keep it simple: treat every non-hard flag as one unit of risk.
        return len(_flags_for(skill))

    def _is_ok(skill: str) -> bool:
        return _soft_count(skill) == 0

    # Keep only usable candidates (respect max_jump_level and drop unknown entries).
    usable: dict[str, int] = {}
    for name, jl in candidates.items():
        if not isinstance(name, str):
            continue
        if not isinstance(jl, int):
            continue
        jl_i = int(jl)
        if jl_i < 0 or jl_i > int(max_level):
            continue
        usable[name.upper()] = jl_i

    if not usable:
        return ("SLOW_WALK", 0, "no_candidates_fallback")

    # When we're on an elevated platform and there's an enemy on the ground ahead,
    # long forward jumps are prone to mid-air collisions. Prefer stomping from above
    # when it is explicitly safe; otherwise stay conservative.
    if on_platform and enemy_below_ahead and (not cluster):
        if "STOMP" in usable and (not _is_hard_fatal("STOMP")) and _is_ok("STOMP"):
            return ("STOMP", usable["STOMP"], "platform_stomp_ok")
        if "SLOW_WALK" in usable and (not _is_hard_fatal("SLOW_WALK")) and _is_ok("SLOW_WALK"):
            return ("SLOW_WALK", usable["SLOW_WALK"], "platform_hold_position")

    # Special-case: stairs pit is rare but lethal; always prioritize it if available.
    if "STAIRS_JUMP" in usable and (not _is_hard_fatal("STAIRS_JUMP")):
        return ("STAIRS_JUMP", usable["STAIRS_JUMP"], "stairs_jump_priority")

    # Default preference orders (by scenario).
    if cluster:
        pref = ["DENSE_SAFE", "SAFE", "FAST", "SLOW_WALK", "STOMP"]
    elif enemy_near:
        pref = ["SAFE", "FAST", "DENSE_SAFE", "STOMP", "SLOW_WALK"]
    else:
        # Default: keep moving with jump_level=0 unless we have a strong reason to jump.
        pref = ["SLOW_WALK", "FAST", "SAFE", "DENSE_SAFE", "STOMP"]

    # 1) If an OK option exists, take it (but for clusters, avoid STOMP unless it's the only choice).
    ok = [s for s in pref if s in usable and _is_ok(s)]
    if ok:
        if cluster and "STOMP" in ok and len(ok) > 1:
            ok = [s for s in ok if s != "STOMP"]
        if ok:
            s = ok[0]
            return (s, usable[s], "risk_ok")

    # 2) For close enemies, avoid choosing jump_level=0 when we have any non-hard-fatal nonzero option.
    if enemy_near:
        nonzero_non_hard = [
            s for s, jl in usable.items() if int(jl) > 0 and (not _is_hard_fatal(s)) and s != "STOMP"
        ]
        if cluster and nonzero_non_hard:
            # Pick the least risky "jump over the cluster" option, even if STOMP is OK.
            nonzero_non_hard.sort(key=lambda s: (_soft_count(s), -int(usable[s]), pref.index(s) if s in pref else 999))
            s = nonzero_non_hard[0]
            return (s, usable[s], "cluster_avoid_stomp")
        if nonzero_non_hard:
            nonzero_non_hard.sort(key=lambda s: (_soft_count(s), pref.index(s) if s in pref else 999))
            s = nonzero_non_hard[0]
            return (s, usable[s], "enemy_near_force_jump")

    # 3) General fallback: prefer non-hard-fatal with the fewest flags.
    non_hard = [s for s in pref if s in usable and (not _is_hard_fatal(s))]
    if non_hard:
        non_hard.sort(key=lambda s: (_soft_count(s), pref.index(s) if s in pref else 999))
        s = non_hard[0]
        return (s, usable[s], "min_soft_flags")

    # 4) Last resort: take the first available candidate.
    s = next(iter(usable.keys()))
    return (s, usable[s], "last_resort")


def act_mario_deterministic(
    *,
    preproc: Any,
    game_info: dict[str, Any],
    image_meta: dict[str, Any],
    config: dict[str, Any],
    logger: JsonlLogger,
    step_id: int,
    action_queue: dict[str, list[str]],
    update_memory_fn: Any,
) -> str:
    """Deterministic Mario act(): pick a jump_level from Candidates + Risk Report without model calls."""
    from agents.crowsphere.adapters import super_mario as mario_adapter

    obs_str = str(getattr(preproc, "text", "") or "")
    max_level, urgency = mario_adapter._parse_mario_constraints_from_obs_str(obs_str)
    candidates = mario_adapter._parse_mario_candidate_jump_levels(obs_str)
    risks = mario_adapter._parse_risk_report_flags(obs_str)
    nearest_name, nearest_dist = mario_adapter._parse_nearest_threat_from_obs_str(obs_str)

    skill, jump_level, reason = _pick_skill_from_candidates(
        obs_str=obs_str,
        candidates=candidates,
        risks=risks,
        max_level=max_level,
        urgency=urgency,
        nearest_name=nearest_name,
        nearest_dist=nearest_dist,
    )
    jump_level = max(0, min(int(max_level), int(jump_level)))
    final_action = f"Jump Level: {jump_level}"

    action_queue["super_mario"] = []
    update_memory_fn(game="super_mario", obs_text=obs_str, action_str=final_action)

    logger.write(
        {
            "event": "act",
            "game": "super_mario",
            "step_id": step_id,
            "status": "deterministic",
            "policy": "candidates_risk_minimize",
            "decision": {
                "skill": skill,
                "jump_level": jump_level,
                "reason": reason,
                "constraints": {"max_jump_level": max_level, "urgency": urgency},
                "nearest": {"name": nearest_name, "dist_px": nearest_dist},
            },
            "obs_text": obs_str,
            "image": image_meta,
            "candidate_table": dict(candidates),
            "risk_table": dict(risks),
            "final_action_str": final_action,
            "game_info": game_info,
        }
    )
    return final_action
