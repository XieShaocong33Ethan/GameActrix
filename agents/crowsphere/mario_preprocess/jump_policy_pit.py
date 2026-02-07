from __future__ import annotations

from agents.crowsphere.mario_jump_rules import JUMP_PARAMS, MARIO_SPEED_PER_STEP, POST_LANDING_RUN_PX
from agents.crowsphere.mario_preprocess.types import Position, ThreatInfo


def choose_jump_for_pit(
    *,
    pit_info: tuple[int, int] | None,
    mario_pos: Position | None,
    nearest_threat: ThreatInfo | None,
    all_threats: list[ThreatInfo] | None,
    on_stairs: bool,
    stair_block_count: int,
) -> tuple[int, str, str] | None:
    # 优先处理坑：跳跃落点必须超过坑的另一边
    if pit_info is None or mario_pos is None:
        return None

    pit_start = pit_info[0]
    pit_end = pit_info[1]
    pit_width = pit_end - pit_start
    dist_to_pit = pit_start - mario_pos.x

    if not (0 < dist_to_pit < 80):
        return None

    blocking_solid: ThreatInfo | None = None
    if all_threats:
        solids = [t for t in all_threats if t.name in {"Pipe", "Stairs"} and 0 < float(t.distance) < float(dist_to_pit)]
        blocking_solid = min(solids, key=lambda t: float(t.distance)) if solids else None

    if blocking_solid is not None:
        # 台阶坑：允许“穿过台阶列的模板匹配”继续处理 pit，否则会被永远卡在非 pit 分支里一路慢走到死区。
        can_ignore_stairs_block = bool(
            blocking_solid.name == "Stairs"
            and nearest_threat is not None
            and nearest_threat.name == "Stairs"
            and (on_stairs or (0 < float(nearest_threat.distance) <= 16.0))
            and int(mario_pos.y) >= 70
            and int(dist_to_pit) <= 70
        )
        if not can_ignore_stairs_block:
            return None

    # 台阶坑提前起跳窗口：
    # 在官方 1-1 末段，坑的模板坐标会有抖动（dist_to_pit 可能在 30~50px 之间跳）。
    # 如果一直等到 dist_to_pit 很小才起跳，宏动作（多帧 right+A）常会来不及，直接从边缘掉下去。
    # 台阶坑提前起跳窗口：
    # 在官方 1-1 末段，坑的模板坐标会有抖动（dist_to_pit 可能在 30~50px 之间跳）。
    # 如果一直等到 dist_to_pit 很小才起跳，宏动作（多帧 right+A）常会来不及，直接从边缘掉下去。
    # 额外 run-up（仅在“大台阶段”启用）：
    # 当 Mario 已经在较高的台阶顶（y≈109）且坑距离 26~35px 时，
    # 直接大跳（lvl6）在部分 warmup 相位下会出现“位移折损/碰壁→落入坑”的不稳定失败。
    # 该问题主要出现在 1-1 末段的大台阶（模板会返回很多 stair blocks）。
    # 更稳策略：先走一步获得更好的助跑/站位，再在下一步跨坑。
    if (
        int(mario_pos.y) >= 109
        and nearest_threat is not None
        and nearest_threat.name == "Stairs"
        and (on_stairs or (0 < float(nearest_threat.distance) <= 16.0))
        and 26 <= int(dist_to_pit) <= 35
        and int(pit_width) <= 40
        and int(stair_block_count) >= 15
    ):
        return (
            0,
            f"stairs pit ({pit_width}px) at {int(dist_to_pit)}px on high step, run-up one more step",
            "HIGH",
        )

    if (
        int(mario_pos.y) >= 70
        and nearest_threat is not None
        and nearest_threat.name == "Stairs"
        and (on_stairs or (0 < float(nearest_threat.distance) <= 16.0))
        and int(dist_to_pit) <= 35
    ):
        urg = "CRITICAL" if int(dist_to_pit) < 30 else "HIGH"
        return (
            6,
            f"stairs pit ({pit_width}px) at {int(dist_to_pit)}px, jump now (lvl6)",
            urg,
        )

    # 运行助跑：当 Mario 已在台阶顶/高台上，坑还在 35-55px 左右的中距离时，
    # 直接大跳（lvl6）经常因为速度不足/起跳点偏差导致落入坑。
    # 更稳做法：先走一步获得更好的起跳位置与速度，再在下一步跨坑。
    if (
        int(mario_pos.y) >= 70
        and nearest_threat is not None
        and nearest_threat.name == "Stairs"
        and (on_stairs or (0 < float(nearest_threat.distance) <= 12.0))
        and 36 <= int(dist_to_pit) <= 55
    ):
        # 回归：在 1-1 末段的大台阶（stair blocks 很多）里，继续跑一步往往会把 Mario 推进到
        # “已离地/无法起跳”的不可恢复死区（随后无论 jump_level 多少都会掉坑）。
        #
        # 因此：当我们明确处于“大台阶”时，宁可更早起跳也不要再拖延。
        if int(stair_block_count) >= 10 and int(pit_width) <= 40:
            return (
                6,
                f"stairs pit ({pit_width}px) at {int(dist_to_pit)}px, jump now (lvl6)",
                "HIGH",
            )
        return (
            0,
            f"pit ({pit_width}px) at {int(dist_to_pit)}px, run-up on stairs before jump",
            "HIGH",
        )

    # 早跳跨坑（利用台阶高度作为“发射台”）：
    # 在 warmup=20~30 的某些 seed 中，若继续爬到台阶顶（x≈2440）再跳，会进入“任何 jump_level 都会掉坑”的死区。
    # 经验上：在台阶顶附近但还没进入死区（dist_to_pit≈50~70px）时，用一次中等跳（lvl2）
    # 直接跨过台阶坑更稳定（跳跃过程中会有更长的空中右移，实际位移远大于 ground 近似）。
    if (
        int(mario_pos.y) >= 70
        and nearest_threat is not None
        and nearest_threat.name == "Stairs"
        and (on_stairs or (0 < float(nearest_threat.distance) <= 16.0))
        and 56 <= int(dist_to_pit) <= 70
        and int(pit_width) <= 40
    ):
        return (
            2,
            f"stairs pit ({pit_width}px) at {int(dist_to_pit)}px, early launch lvl2",
            "CRITICAL",
        )

    # Pit clearance must be determined by the *landing point* (JUMP_PARAMS[level][0]).
    # Post-landing run (POST_LANDING_RUN_PX) happens on the ground; it does NOT help if the landing
    # point itself is inside the pit. Using total_dx here can under-shoot and LAND_IN_PIT.
    needed_distance = dist_to_pit + pit_width + 5
    if (
        int(mario_pos.y) >= 70
        and nearest_threat is not None
        and nearest_threat.name == "Stairs"
        and (on_stairs or (0 < float(nearest_threat.distance) <= 16.0))
    ):
        needed_distance += 15

    def landing_dx_for_level(level: int) -> int:
        if int(level) <= 0:
            return int(MARIO_SPEED_PER_STEP)
        return int(JUMP_PARAMS[int(level)][0])

    max_landing_dx = landing_dx_for_level(6)
    if int(needed_distance) > int(max_landing_dx):
        # 坑太远（当前起跳点无法保证越过坑尾）：
        # 直接连续 slow-walk（jump_level=0）会浪费大量 step，且在 RandomStages / lava 中更容易错过可跳窗口。
        # 更稳的做法：做一次“安全小跳”逼近（保证落点仍在坑前），争取更快进入可跨越距离范围。
        SAFETY_MARGIN_PX = 10
        approach_level: int | None = None
        for lvl in range(6, 0, -1):
            if landing_dx_for_level(lvl) < int(dist_to_pit) - int(SAFETY_MARGIN_PX):
                approach_level = int(lvl)
                break
        if approach_level is not None:
            return (
                int(approach_level),
                f"pit ({pit_width}px) at {dist_to_pit}px, too far to clear, approach hop lvl{int(approach_level)}",
                "LOW",
            )
        return (0, f"pit ({pit_width}px) at {dist_to_pit}px, wait closer", "LOW")

    chosen_level = 6
    for lvl in range(1, 7):
        if landing_dx_for_level(lvl) >= int(needed_distance):
            chosen_level = int(lvl)
            break

    urg = "CRITICAL" if int(dist_to_pit) < 30 else ("HIGH" if int(dist_to_pit) < 55 else "MEDIUM")
    return (int(chosen_level), f"pit ({pit_width}px) at {dist_to_pit}px, jump lvl{int(chosen_level)}", urg)
