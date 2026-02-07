from __future__ import annotations

from agents.crowsphere.mario_jump_rules import (
    JUMP_PARAMS,
    LANDING_DANGER_MARGIN,
    MARIO_SPEED_PER_STEP,
    POST_LANDING_RUN_PX,
    estimate_landing_danger,
    estimate_has_low_ceiling,
    is_enemy_threat_name,
    min_jump_level_for_total_dx,
    select_safest_jump_level,
)
from agents.crowsphere.mario_preprocess.jump_level_helpers import level_for_distance, level_for_height
from agents.crowsphere.mario_preprocess.types import Position, ThreatInfo


def choose_jump_for_nearest_threat(
    *,
    nearest_threat: ThreatInfo | None,
    pit_info: tuple[int, int] | None,
    mario_pos: Position | None,
    bricks: list[Position] | None,
    question_blocks: list[Position] | None,
    all_threats: list[ThreatInfo] | None,
    sentinel_height_diff_warning: bool,
    has_low_ceiling_override: bool | None = None,
) -> tuple[int, str, str] | None:
    if nearest_threat is None:
        return None

    has_low_ceiling = estimate_has_low_ceiling(mario_pos, bricks, question_blocks, lookbehind_px=40)
    if has_low_ceiling_override is not None:
        has_low_ceiling = bool(has_low_ceiling_override)

    dist = float(nearest_threat.distance)
    height = int(nearest_threat.height)
    name = str(nearest_threat.name)
    is_enemy = is_enemy_threat_name(name)
    max_level = 4 if has_low_ceiling else 6

    # ========== Stairs + Pit "step-up" adjustment ==========
    def _adjust_stairs_level_for_pit(level: int) -> tuple[int, str]:
        if name != "Stairs" or (pit_info is None) or (mario_pos is None):
            return level, ""
        pit_start, pit_end = pit_info
        p0 = int(pit_start) - int(mario_pos.x)
        p1 = int(pit_end) - int(mario_pos.x)
        if p1 <= 0:
            return level, ""

        if int(mario_pos.y) >= 70 and float(dist) <= 12.0 and 20 <= p0 <= 55:
            return 0, " [stairs+pit:step_up walk]"

        try:
            base_landing_dx = int(JUMP_PARAMS[int(level)][0]) if int(level) > 0 else 0
        except Exception:
            base_landing_dx = 0
        if p0 < base_landing_dx < p1:
            for cand in range(1, int(max_level) + 1):
                cand_landing = int(JUMP_PARAMS[cand][0])
                if cand_landing < p0:
                    return cand, f" [stairs+pit:step_up lvl{cand}]"
        return level, ""

    level_for_height_needed = level_for_height(height)

    if is_enemy:
        enemy_min_level = min_jump_level_for_total_dx(
            int(dist) + int(LANDING_DANGER_MARGIN) + 1,
            max_level=int(max_level),
        )
        base_level = max(int(level_for_height_needed), int(enemy_min_level))

        if sentinel_height_diff_warning:
            VLM_HEIGHT_DIFF_MIN_LEVEL = 4
            if int(base_level) < int(VLM_HEIGHT_DIFF_MIN_LEVEL):
                base_level = min(int(VLM_HEIGHT_DIFF_MIN_LEVEL), int(max_level))
    else:
        level_for_distance_needed = level_for_distance(int(dist))
        base_level = max(int(level_for_height_needed), int(level_for_distance_needed))

    # 对 Stairs（台阶/砖块墙）做一个保守处理：
    # - 当“需要非常大的跳”（base_level>=5）但距离还不算紧急（>=40px）时，
    #   直接大跳往往更不稳定（随机/非 1-1 关卡中模板匹配可能误报高度，或落点信息不足）。
    # - 更稳做法是先逼近/观察一小步，再在下一步决定是否需要大跳。
    # 该规则只在“没有 pit 证据”时启用，避免影响 1-1 末段的 stairs+pit 逻辑（pit 分支优先返回）。
    if name == "Stairs" and pit_info is None and (not has_low_ceiling) and float(dist) >= 40.0:
        if int(base_level) >= 5:
            return (0, f"Stairs at {int(dist)}px (tall), approach for better window", "LOW")

    def _safe_level_for_visible_enemies(
        min_level: int,
        *,
        allow_wait: bool,
    ) -> tuple[int, str]:
        enemies = [
            t
            for t in (all_threats or [])
            if is_enemy_threat_name(str(t.name)) and t.distance > 0
        ]
        if not enemies:
            return int(min_level), ""

        is_dangerous, _, _ = estimate_landing_danger(int(min_level), enemies)
        if not is_dangerous:
            return int(min_level), ""

        lvl, rsn = select_safest_jump_level(
            int(min_level),
            enemies,
            max_level=int(max_level),
            allow_wait=bool(allow_wait),
        )
        return int(lvl), str(rsn)

    def _cruise_jump_wont_hit_obstacle(
        obstacle_dist: float,
        *,
        level: int = 3,
        margin_px: int = 12,
    ) -> bool:
        try:
            total_dx = int(JUMP_PARAMS[int(level)][0]) + int(POST_LANDING_RUN_PX)
        except Exception:
            return False
        return float(total_dx) < (float(obstacle_dist) - float(margin_px))

    # =========== Phase 1: Pipe+Enemy 联合落点安全约束（带等待/逼近策略） ===========
    if name == "Pipe" and dist < 80:
        enemies_behind_pipe = [
            t for t in (all_threats or []) if is_enemy_threat_name(str(t.name)) and t.distance > dist
        ]
        if enemies_behind_pipe:
            allow_wait = dist >= 40
            safe_level, reason = select_safest_jump_level(
                int(base_level),
                enemies_behind_pipe,
                max_level=int(max_level),
                allow_wait=bool(allow_wait),
            )

            if dist < 30:
                urgency = "CRITICAL"
            elif dist < 50:
                urgency = "HIGH"
            else:
                urgency = "MEDIUM"

            return (
                int(safe_level),
                f"{name} at {int(dist)}px + enemy behind, {reason}",
                urgency,
            )
    # =========== End Phase 1 ===========

    PANIC_ENEMY_DIST = 18
    PANIC_MIN_LEVEL = 4

    if dist < 30:
        if is_enemy:
            level = min(int(max_level), int(base_level))
        else:
            if name == "Stairs" and pit_info and mario_pos:
                level = min(int(max_level), int(base_level))
            else:
                level = min(int(max_level), int(base_level) + 1)
        if is_enemy and dist < PANIC_ENEMY_DIST:
            level = max(int(level), min(int(PANIC_MIN_LEVEL), int(max_level)))
        if has_low_ceiling and is_enemy and dist <= 25:
            # 低天花板下，过小的跳（lvl1）在贴脸敌人场景里更容易“没离地就被撞”或落点仍贴脸。
            # lvl2 通常仍不会顶头，但能更稳定地越过/踩过最近的敌人。
            # 但在极近距离时，lvl2 的起跳阶段更容易发生侧向碰撞；
            # 此时用更短的 lvl1 往往更能“及时离地并踩到”。（经验阈值：≈14px）
            if float(dist) <= 14.0:
                level = min(1, int(max_level))
            else:
                level = min(2, int(max_level))
        if is_enemy:
            level, rsn = _safe_level_for_visible_enemies(int(level), allow_wait=False)
            suffix = f", {rsn}" if rsn else ""
        else:
            suffix = ""
        level, pit_suffix = _adjust_stairs_level_for_pit(int(level))
        return (
            int(level),
            f"URGENT! {name} at {int(dist)}px, jump NOW{suffix}{pit_suffix}",
            "CRITICAL",
        )

    if dist < 60:
        level = min(int(max_level), int(base_level))
        if is_enemy:
            visible_enemy_count = sum(
                1
                for t in (all_threats or [])
                if is_enemy_threat_name(str(t.name)) and t.distance > 0
            )
            if has_low_ceiling and visible_enemy_count == 1 and 40 <= dist <= 55:
                level = 2
                level, rsn = _safe_level_for_visible_enemies(int(level), allow_wait=False)
                suffix = f", {rsn}" if rsn else ""
                return (
                    int(level),
                    f"{name} at {int(dist)}px under low ceiling, take small jump{suffix}",
                    "MEDIUM",
                )

            allow_wait = dist >= 40
            if has_low_ceiling and visible_enemy_count >= 2:
                allow_wait = False
            level, rsn = _safe_level_for_visible_enemies(int(level), allow_wait=bool(allow_wait))
            suffix = f", {rsn}" if rsn else ""
        else:
            suffix = ""
        level, pit_suffix = _adjust_stairs_level_for_pit(int(level))
        return (
            int(level),
            f"{name} at {int(dist)}px, jump now (base={int(base_level)},max={int(max_level)}){suffix}{pit_suffix}",
            "HIGH",
        )

    if dist < 80:
        next_dist = float(dist) - float(MARIO_SPEED_PER_STEP)
        should_jump_now = next_dist < 60
        if sentinel_height_diff_warning and is_enemy:
            should_jump_now = True

        if should_jump_now:
            level = min(int(max_level), int(base_level))
            if is_enemy:
                allow_wait = dist >= 40 and not sentinel_height_diff_warning
                if has_low_ceiling:
                    visible_enemy_count = sum(
                        1
                        for t in (all_threats or [])
                        if is_enemy_threat_name(str(t.name)) and t.distance > 0
                    )
                    if visible_enemy_count >= 2:
                        allow_wait = False
                level, rsn = _safe_level_for_visible_enemies(int(level), allow_wait=bool(allow_wait))

                if allow_wait and (not has_low_ceiling) and (not sentinel_height_diff_warning) and level >= 5 and dist >= 55:
                    return (
                        0,
                        f"{name} at {int(dist)}px, approach for safer window (avoid lvl{int(level)})",
                        "LOW",
                    )
                suffix = f", {rsn}" if rsn else ""
                suffix += f" (base={int(base_level)},max={int(max_level)})"
                if sentinel_height_diff_warning:
                    suffix += " [VLM]"
            else:
                suffix = ""
            level, pit_suffix = _adjust_stairs_level_for_pit(int(level))
            return (
                int(level),
                f"{name} at {int(dist)}px, prepare to jump{suffix}{pit_suffix}",
                "MEDIUM",
            )

        return (0, f"{name} at {int(dist)}px, approach", "LOW")

    if dist < 120:
        if sentinel_height_diff_warning and is_enemy:
            level = min(int(max_level), int(base_level))
            level, rsn = _safe_level_for_visible_enemies(int(level), allow_wait=True)
            suffix = f", {rsn}" if rsn else ""
            return (int(level), f"[VLM] height-diff warning: {name} at {int(dist)}px{suffix}", "MEDIUM")

        if name in {"Pipe", "Stairs"}:
            level = min(int(max_level), int(base_level))
            if name == "Pipe":
                level, rsn = _safe_level_for_visible_enemies(int(level), allow_wait=True)
                suffix = f", {rsn}" if rsn else ""
            else:
                suffix = ""
            level, pit_suffix = _adjust_stairs_level_for_pit(int(level))
            return (int(level), f"{name} at {int(dist)}px, advance{suffix}{pit_suffix}", "LOW")

        if is_enemy and not has_low_ceiling:
            return (3, f"{name} at {int(dist)}px, speed jump", "LOW")
        return (0, f"{name} at {int(dist)}px, watch ahead", "LOW")

    if sentinel_height_diff_warning and is_enemy:
        return (0, f"[VLM] height-diff warning: {name} at {int(dist)}px, prepare early", "LOW")

    if name in {"Pipe", "Stairs"} and not has_low_ceiling and _cruise_jump_wont_hit_obstacle(dist):
        return (3, f"{name} far (>{int(dist)}px), cruise jump", "NONE")
    if is_enemy and not has_low_ceiling:
        return (3, f"{name} far (>{int(dist)}px), speed jump", "NONE")
    return (0, f"threats distant (>{int(dist)}px), safe", "NONE")
