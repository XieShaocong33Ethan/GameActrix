from __future__ import annotations

import re
from typing import Any

from agents.crowsphere.errors import RetryableActionPlanError

SKILL_ENUM = [
    "SAFE",
    "FAST",
    "STOMP",
    "DENSE_SAFE",
    "STAIRS_JUMP",
    "SLOW_WALK",
]

# 竞赛约束：Super Mario 不允许使用宏动作/确定性规划技能。
# 当前 schema/prompt 已不提供相关 skill；这里保留空集合仅作为兜底。
DISALLOWED_SKILLS: set[str] = set()


def _parse_mario_constraints_from_obs_str(obs_str: str) -> tuple[int, str | None]:
    """
    从（预处理后的）obs_str 中提取 Mario 的约束：
    - max_jump_level（默认 6）
    - urgency（可为空）
    """
    max_level = 6
    urgency: str | None = None

    max_match = re.search(r"max_jump_level\s*=\s*(\d+)", obs_str, re.IGNORECASE)
    if max_match:
        try:
            max_level = int(max_match.group(1))
        except Exception:
            max_level = 6
    max_level = max(0, min(6, int(max_level)))

    urgency_match = re.search(r"Urgency:\s*([A-Z]+)", obs_str, re.IGNORECASE)
    if urgency_match:
        urgency = str(urgency_match.group(1)).upper()

    return max_level, urgency


_NEAREST_THREAT_RE = re.compile(r"Nearest:\s*([A-Za-z_]+)\s+at\s*\(", re.IGNORECASE)
_NEAREST_DIST_RE = re.compile(r"Distance:\s*(\d+)\s*px", re.IGNORECASE)


def _parse_nearest_threat_from_obs_str(obs_str: str) -> tuple[str | None, int | None]:
    """
    从（预处理后的）obs_str 里解析最近威胁（名字 + 距离）。

    预处理文本格式示例：
      Nearest: Pipe at (182, 79)
        Distance: 60px
    """
    text = str(obs_str or "")
    m_name = _NEAREST_THREAT_RE.search(text)
    m_dist = _NEAREST_DIST_RE.search(text)
    name = str(m_name.group(1)).strip() if m_name else None
    dist: int | None = None
    if m_dist:
        try:
            dist = int(m_dist.group(1))
        except Exception:
            dist = None
    return (name.upper() if name else None), dist


def _parse_mario_candidate_jump_levels(obs_str: str) -> dict[str, int]:
    """从 [Decision Support] Candidates 中提取 jump_level 映射。"""
    out: dict[str, int] = {}
    if "[Decision Support]" not in obs_str:
        return out
    for name in SKILL_ENUM:
        match = re.search(rf"-\s*{name}\s*:\s*jump_level\s*=\s*(\d+)", obs_str, re.IGNORECASE)
        if not match:
            continue
        try:
            jump_level = int(match.group(1))
        except Exception:
            continue
        out[name.upper()] = max(0, min(6, jump_level))
    return out


def _parse_risk_report_flags(obs_str: str) -> dict[str, list[str]]:
    """
    Parse [Risk Report] lines:
      - SAFE: jl=4 dx≈91px => HIT_Pipe@99px, LANDING_DANGER(...), ...
    into {"SAFE": ["HIT_Pipe@99px", "LANDING_DANGER(...)"], ...}
    """
    if "[Risk Report]" not in obs_str:
        return {}
    block = obs_str.split("[Risk Report]", 1)[1]
    block = re.split(r"\n\[[^\]]+\]\n", block, maxsplit=1)[0]
    out: dict[str, list[str]] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        match = re.match(r"-\s*([A-Z_]+)\s*:\s*.*=>\s*(.*)$", line)
        if not match:
            continue
        name = match.group(1).strip().upper()
        flags_str = match.group(2).strip()
        if flags_str.upper() == "OK":
            out[name] = []
        else:
            out[name] = [flag.strip() for flag in flags_str.split(",") if flag.strip()]
    return out


PANIC_FATAL_DIST_PX = 8
_PANIC_AHEAD_DIST_RE = re.compile(r"PANIC_AHEAD@(\d+)\s*PX", re.IGNORECASE)


def _panic_ahead_distance_px(flag: str) -> int | None:
    m = _PANIC_AHEAD_DIST_RE.search(flag or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _has_hard_fatal_flag(flags: list[str]) -> bool:
    """硬致命风险：无论是否存在其他候选，一律拒绝并要求模型改选。"""
    for flag in flags:
        flag_upper = flag.upper()
        if "LAND_IN_PIT" in flag_upper:
            return True
        if "PIT_TOO_CLOSE_TO_JUMP" in flag_upper or "PIT_EDGE_TOO_CLOSE" in flag_upper:
            return True
        if flag_upper.startswith("HIT_"):
            return True
    return False


def _has_soft_fatal_flag(flags: list[str]) -> bool:
    """
    软致命风险：只有当存在 Risk Report => OK 的其他候选时，才拒绝并要求模型改选。
    用途：避免模型“明知危险仍选择”导致早死，同时避免“所有候选都被拒绝”卡死。
    """
    for flag in flags:
        flag_upper = flag.upper()
        if flag_upper.startswith("LANDING_DANGER"):
            return True
        # TAKEOFF_COLLISION is a heuristic risk (not a deterministic collision like HIT_Pipe),
        # so only treat it as fatal when there is an OK alternative.
        if "TAKEOFF_COLLISION" in flag_upper:
            return True
        if "PANIC_AHEAD@" in flag_upper:
            dist = _panic_ahead_distance_px(flag)
            if dist is None:
                return True
            if 0 <= int(dist) <= int(PANIC_FATAL_DIST_PX):
                return True
    return False


def _normalize_skill_name(raw: str) -> str:
    # 容忍模型把 DENSE_SAFE 写成 "dense safe"/"dense-safe" 之类的格式差异。
    token = str(raw).strip().upper()
    token = re.sub(r"[\s-]+", "_", token)
    return token


def _parse_vlm_skill(action: dict[str, Any]) -> str | None:
    raw = action.get("skill")
    if not isinstance(raw, str):
        return None
    skill = _normalize_skill_name(raw)
    return skill if skill else None


def _parse_vlm_jump_level(action: dict[str, Any]) -> int | None:
    if "jump_level" not in action:
        return None
    raw = action.get("jump_level")
    if not isinstance(raw, int):
        return None
    return max(0, min(6, int(raw)))


def mario_plan_to_action_strs(
    plan: dict[str, Any],
    game_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    actions = plan["actions"]
    if len(actions) != 1:
        raise RetryableActionPlanError(
            "mario_actions_length_mismatch",
            "Super Mario 每次只能输出 1 个动作（actions 长度必须为 1）",
        )

    obs_str = ""
    if isinstance(game_info, dict):
        obs_str = str(game_info.get("obs_str") or "")

    max_level, urgency = _parse_mario_constraints_from_obs_str(obs_str)
    candidates = _parse_mario_candidate_jump_levels(obs_str)
    risks = _parse_risk_report_flags(obs_str)

    action = actions[0]
    if not isinstance(action, dict):
        raise RetryableActionPlanError("invalid_action_item_for_mario", "actions[0] 必须是对象")

    vlm_jump_level = _parse_vlm_jump_level(action)
    if vlm_jump_level is None:
        raise RetryableActionPlanError(
            "mario_missing_jump_level",
            "必须输出 jump_level（整数，0..6）。",
        )
    if int(vlm_jump_level) > int(max_level):
        raise RetryableActionPlanError(
            "mario_jump_level_exceeds_max",
            f"jump_level={vlm_jump_level} 超过 max_jump_level={max_level}",
        )
    chosen_jump_level = int(vlm_jump_level)

    vlm_skill = _parse_vlm_skill(action)
    final_skill: str | None = vlm_skill
    if vlm_skill is not None:
        # 竞赛约束：Super Mario 不允许宏动作/确定性规划技能；只允许模型直接选择 jump_level。
        if vlm_skill in DISALLOWED_SKILLS:
            raise RetryableActionPlanError(
                "mario_disallowed_skill",
                f"skill={vlm_skill} 属于宏动作/确定性规划，不允许使用。请改选其他候选（或省略 skill 字段）。",
            )
        if candidates and (vlm_skill not in candidates):
            raise RetryableActionPlanError(
                "mario_skill_not_in_candidates",
                f"skill={vlm_skill} 不在 Candidates 中（若不想输出 skill，可以省略该字段）",
            )
        if candidates:
            expected = int(candidates.get(vlm_skill, chosen_jump_level))
            if int(expected) != int(chosen_jump_level):
                raise RetryableActionPlanError(
                    "mario_jump_level_mismatch",
                    f"jump_level={chosen_jump_level} 与 Candidates 中 {vlm_skill} 的 jump_level={expected} 不一致",
                )

    # Urgency=CRITICAL 且前方是 Pipe/Stairs 时，禁止 jump_level=0（避免贴脸无起跳窗口）。
    nearest_name, nearest_dist = _parse_nearest_threat_from_obs_str(obs_str)
    has_nonzero_non_hard_fatal = False
    if candidates and risks:
        has_nonzero_non_hard_fatal = any(
            (name not in DISALLOWED_SKILLS)
            and int(jl) > 0
            and int(jl) <= int(max_level)
            and (not _has_hard_fatal_flag(list(risks.get(name) or [])))
            for name, jl in candidates.items()
        )

    if (
        chosen_jump_level == 0
        and urgency == "CRITICAL"
        and nearest_name in {"PIPE", "STAIRS"}
        and isinstance(nearest_dist, int)
        and nearest_dist <= 30
        and has_nonzero_non_hard_fatal
    ):
        raise RetryableActionPlanError(
            "mario_forbidden_wait_on_critical_solid",
            f"最近威胁是 {nearest_name} 距离={nearest_dist}px 且 Urgency=CRITICAL：禁止选择 jump_level=0（必须立即跳）。",
        )

    # HIGH/CRITICAL 的近距离敌人：若存在 Risk=OK 的非 0 候选，则禁止 jump_level=0。
    if (
        chosen_jump_level == 0
        and urgency in {"HIGH", "CRITICAL"}
        and nearest_name not in {None, "PIPE", "STAIRS"}
        and isinstance(nearest_dist, int)
        and nearest_dist <= 75
        and candidates
        and risks
    ):
        has_ok_nonzero = any(
            (name not in DISALLOWED_SKILLS)
            and int(jl) > 0
            and int(jl) <= int(max_level)
            and isinstance(risks.get(name), list)
            and (len(risks.get(name) or []) == 0)
            for name, jl in candidates.items()
        )
        if has_ok_nonzero:
            raise RetryableActionPlanError(
                "mario_forbidden_wait_on_high_enemy",
                f"最近威胁是 {nearest_name} 距离={nearest_dist}px 且 Urgency={urgency}：存在 Risk Report=>OK 的非 0 候选，禁止选择 jump_level=0。",
            )

    # 根据 Risk Report 对本次 jump_level 做安全校验（优先用模型 skill 定位）。
    flags: list[str] = []
    eval_skill: str | None = None
    if vlm_skill is not None and isinstance(risks.get(vlm_skill), list):
        flags = list(risks.get(vlm_skill) or [])
        eval_skill = vlm_skill
    elif candidates and risks:
        same_level = [
            name
            for name, jl in candidates.items()
            if (name not in DISALLOWED_SKILLS)
            and int(jl) == int(chosen_jump_level)
            and int(jl) <= int(max_level)
        ]
        ok_same_level = [name for name in same_level if isinstance(risks.get(name), list) and len(risks.get(name) or []) == 0]
        picked = ok_same_level[0] if ok_same_level else (same_level[0] if same_level else None)
        if picked is not None:
            flags = list(risks.get(picked) or [])
            eval_skill = str(picked)

    if final_skill is None:
        final_skill = eval_skill

    def _has_severe_panic(fs: list[str]) -> bool:
        for f in fs:
            dist = _panic_ahead_distance_px(f)
            if dist is not None and 0 <= int(dist) <= int(PANIC_FATAL_DIST_PX):
                return True
        return False

    def _has_landing_danger(fs: list[str]) -> bool:
        for f in fs:
            if str(f).strip().upper().startswith("LANDING_DANGER"):
                return True
        return False

    def _suggest_non_severe_candidates() -> list[str]:
        if not candidates or not risks:
            return []
        pref = ["SLOW_WALK", "DENSE_SAFE", "SAFE", "FAST", "STOMP"]
        out: list[str] = []
        for name in pref:
            if name == eval_skill:
                continue
            jl = candidates.get(name)
            fl = risks.get(name)
            if jl is None or int(jl) > int(max_level) or not isinstance(fl, list):
                continue
            if _has_hard_fatal_flag(list(fl)) or _has_severe_panic(list(fl)):
                continue
            out.append(name)
        return out

    # 合规：不直接改写模型动作；用约束 + 反馈让模型在候选中重新选择。
    if eval_skill is not None and candidates and risks and _has_severe_panic(flags):
        suggestions = _suggest_non_severe_candidates()
        if suggestions:
            raise RetryableActionPlanError(
                "mario_severe_panic",
                "你选择的候选触发 PANIC_AHEAD@<=8px（贴脸必死窗口）。"
                f"请从 candidates 中改选一个不含该标志且非 hard-fatal 的候选（建议：{', '.join(suggestions)}）。",
            )

    def _suggest_non_landing_danger_candidates() -> list[str]:
        if not candidates or not risks:
            return []
        pref = ["SLOW_WALK", "SAFE", "FAST", "DENSE_SAFE", "STOMP"]
        out: list[str] = []
        for name in pref:
            if name == eval_skill:
                continue
            jl = candidates.get(name)
            fl = risks.get(name)
            if jl is None or int(jl) > int(max_level) or not isinstance(fl, list):
                continue
            if _has_hard_fatal_flag(list(fl)) or _has_severe_panic(list(fl)) or _has_landing_danger(list(fl)):
                continue
            out.append(name)
        return out

    # 当 Urgency 很高且落点风险明显时，如果存在“没有 LANDING_DANGER”的替代候选，则要求模型改选。
    if urgency in {"HIGH", "CRITICAL"} and eval_skill is not None and candidates and risks and _has_landing_danger(flags):
        suggestions = _suggest_non_landing_danger_candidates()
        if suggestions:
            raise RetryableActionPlanError(
                "mario_landing_danger",
                "你选择的候选在 [Risk Report] 中包含 LANDING_DANGER，且当前 Urgency 很高。"
                f"请从 candidates 中改选一个不含 LANDING_DANGER 且非 hard-fatal 的候选（建议：{', '.join(suggestions)}）。",
            )

    # 合规：不强制改为 FAST；当 FAST=>OK 且 Urgency 低时，禁止输出 jump_level=0，要求模型改选 FAST。
    if candidates and risks and chosen_jump_level == 0 and urgency in {None, "NONE", "LOW"}:
        fast_jl = candidates.get("FAST")
        fast_flags = risks.get("FAST")
        if isinstance(fast_jl, int) and 0 < int(fast_jl) <= int(max_level) and fast_flags == []:
            raise RetryableActionPlanError(
                "mario_prefer_fast_when_ok",
                f"Urgency={urgency or 'NONE'} 且 FAST 在 Risk Report 中为 OK：请输出 FAST（建议 skill=FAST, jump_level={fast_jl}），不要选择 jump_level=0。",
            )

    has_ok_alternative = False
    if candidates and risks:
        for name, jl in candidates.items():
            if name in DISALLOWED_SKILLS:
                continue
            if name == eval_skill:
                continue
            if int(jl) > int(max_level):
                continue
            fl = risks.get(name, [])
            if isinstance(fl, list) and (len(fl) == 0):
                has_ok_alternative = True
                break

    hard_fatal = _has_hard_fatal_flag(flags)
    soft_fatal = has_ok_alternative and _has_soft_fatal_flag(flags)

    # 避免“无解状态”死锁：当所有可选候选都带 hard-fatal 标志时，继续要求模型改选没有意义。
    has_non_hard_fatal_alternative = False
    if candidates and risks:
        for name, jl in candidates.items():
            if name in DISALLOWED_SKILLS:
                continue
            if name == eval_skill:
                continue
            if int(jl) > int(max_level):
                continue
            alt_flags = risks.get(name, [])
            if not _has_hard_fatal_flag(alt_flags):
                has_non_hard_fatal_alternative = True
                break

    should_reject_for_hard_fatal = bool(hard_fatal and has_non_hard_fatal_alternative)
    if should_reject_for_hard_fatal or soft_fatal:
        note = (
            "存在非 hard-fatal 的其他候选，必须改选"
            if should_reject_for_hard_fatal
            else "存在 Risk Report => OK 的候选，必须改选到 OK"
        )
        detail = ", ".join(flags) if flags else "风险评估缺失"
        raise RetryableActionPlanError(
            "mario_fatal_risk",
            f"对 jump_level={chosen_jump_level} 的评估结果：{detail}（{note}）",
        )

    normalized_plan = dict(plan)
    normalized_plan["actions"] = [{"jump_level": chosen_jump_level}]
    normalized_plan["chunk_size"] = 1

    # Observability for analysis/debugging (adapter does not override model choice).
    normalized_plan["vlm_skill"] = vlm_skill
    normalized_plan["vlm_jump_level"] = vlm_jump_level
    normalized_plan["final_skill"] = final_skill
    normalized_plan["final_jump_level"] = chosen_jump_level
    normalized_plan["override_reason"] = None
    normalized_plan["candidate_table"] = dict(candidates)
    normalized_plan["risk_table"] = dict(risks)
    normalized_plan["constraints"] = {"max_jump_level": max_level, "urgency": urgency}

    return normalized_plan, [f"Jump Level: {chosen_jump_level}"]
