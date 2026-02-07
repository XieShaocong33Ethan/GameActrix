from __future__ import annotations

from typing import Protocol, Sequence


# 物理常数（以评测环境 SuperMarioEnv 的步长为准）
MARIO_SPEED_PER_STEP = 7  # Mario 每步向右移动约 7 像素

# 物体尺寸（像素）
GOOMBA_SIZE = 16
KOOPA_SIZE = 24

# Threat 分类约定：
# - Pipe / Stairs 属于“实体障碍”（solid），需要按高度决定跳跃级别
# - 其他（包括未知 Monster）都按“敌人/危险物”处理，用落点安全规则规避
SOLID_THREAT_NAMES = {"Pipe", "Stairs"}


def is_solid_threat_name(name: str) -> bool:
    return str(name) in SOLID_THREAT_NAMES


def is_enemy_threat_name(name: str) -> bool:
    return not is_solid_threat_name(str(name))


def is_stompable_enemy_name(name: str) -> bool:
    """仅对这些敌人开放 STOMP（踩怪）策略；未知敌人默认不踩更安全。"""
    return str(name) in {"Goomba", "Koopa", "Koopa_shadow"}

# 跳跃参数 (水平距离, 垂直高度)
JUMP_PARAMS = {
    0: (0, 0),
    1: (42, 35),
    2: (56, 46),
    3: (63, 53),
    4: (70, 60),
    5: (77, 65),
    6: (84, 68),
}


# SuperMarioEnv 的 step() 在跳跃后会继续执行 action=0，直到 y_pos 连续 3 次相同才认为“落地稳定”。
# 这 3 次相同里包含“刚落地的那一次”，因此“落地后额外向右走”的步数通常是 2（而不是 3）。
# 这段额外位移会显著影响“落地后 run-in 撞怪”的判定，因此必须与官方环境对齐。
POST_LANDING_FRAMES = 2
POST_LANDING_RUN_PX = MARIO_SPEED_PER_STEP * POST_LANDING_FRAMES  # 约 14px（满速时）


# 连续敌人判定：两只敌人之间的间距很小，单次小跳容易“落地后撞上下一只”
ENEMY_CLUSTER_GAP_PX = 40
ENEMY_CLUSTER_LOOKAHEAD_PX = 140
ENEMY_CLUSTER_CLEARANCE_PX = 4  # 额外安全余量（经验值）


class PositionLike(Protocol):
    x: int
    y: int


class ThreatLike(Protocol):
    name: str
    distance: float


def enemy_width_px(name: str) -> int:
    if name == "Koopa":
        return KOOPA_SIZE
    # 默认按 Goomba 处理
    return GOOMBA_SIZE


def detect_ground_enemy_cluster(all_threats: Sequence[ThreatLike] | None) -> list[ThreatLike]:
    """
    返回以“最近地面敌人”为起点的连续敌人簇（至少 2 个），否则返回空列表。

    判定口径：
    - 只考虑“敌人类威胁”（非 Pipe/Stairs）；未知 Monster 也会被纳入
    - 只看前方一定范围（lookahead）
    - 相邻两只敌人距离差 <= gap 视为“连续”
    """
    if not all_threats:
        return []

    enemies = [t for t in all_threats if is_enemy_threat_name(str(t.name)) and t.distance > 0]
    enemies.sort(key=lambda t: t.distance)
    enemies = [t for t in enemies if t.distance <= ENEMY_CLUSTER_LOOKAHEAD_PX]
    if len(enemies) < 2:
        return []

    cluster: list[ThreatLike] = [enemies[0]]
    for t in enemies[1:]:
        if t.distance - cluster[-1].distance <= ENEMY_CLUSTER_GAP_PX:
            cluster.append(t)
            continue
        break

    return cluster if len(cluster) >= 2 else []


def estimate_has_low_ceiling(
    mario_pos: PositionLike | None,
    bricks: Sequence[PositionLike] | None,
    question_blocks: Sequence[PositionLike] | None,
    *,
    lookbehind_px: int = 0,
) -> bool:
    """
    经验规则：如果前方/头顶附近存在"低天花板"，则避免高跳（容易顶头碰砖）。

    注意：这里不追求完美物理准确，只用保守范围覆盖常见低天花板（y≈95 附近）。

    Args:
        lookbehind_px: 向后检测的范围（默认 0）。设置正值时，也检测 Mario 身后的砖块，
                       用于"前瞻性低天花板检测"——当 Mario 刚跳过来时，身后的砖块仍应算作低天花板。
    """
    if not mario_pos:
        return False

    # 注意：Mario 只会向右走，因此“身后较远处的单块低砖”不应单独触发 low ceiling，
    # 否则会在离开 corridor 后仍长期把 max_jump_level clamp 到 4，导致后续清怪/跨坑窗口丢失。
    #
    # 经验规则（strong evidence）：
    # - 前方出现低砖（dx>=0）
    # - 或贴身身后出现低砖（-16<=dx<=-1，常见于模板抖动）
    # - 或身后较远处同时出现 >=2 块低砖（更像 corridor，而不是噪声）
    lookbehind_px_int = max(0, int(lookbehind_px))
    back_margin_px = 24  # 经验值：约 1.5 块砖宽度
    min_dx = -min(int(back_margin_px), int(lookbehind_px_int))

    low_dxs: list[int] = []
    for block in list(bricks or []) + list(question_blocks or []):
        dx = int(block.x) - int(mario_pos.x)
        if int(min_dx) <= int(dx) <= 80 and 80 < int(block.y) < 130:
            low_dxs.append(int(dx))

    if not low_dxs:
        return False

    has_ahead = any(int(dx) >= 0 for dx in low_dxs)
    has_near_behind = any(-16 <= int(dx) <= -1 for dx in low_dxs)
    far_behind_count = sum(1 for dx in low_dxs if -24 <= int(dx) <= -17)
    return bool(has_ahead or has_near_behind or (far_behind_count >= 2))


def min_jump_level_for_total_dx(total_dx_needed: int, *, max_level: int) -> int:
    """
    根据"单步总前进距离"的粗估选择 jump_level。

    粗估：jump 总水平位移（JUMP_PARAMS[level][0]） + 落地后额外前进（POST_LANDING_RUN_PX）
    """
    for lvl in range(0, max_level + 1):
        jump_dx = JUMP_PARAMS[lvl][0]
        if jump_dx + POST_LANDING_RUN_PX >= total_dx_needed:
            return lvl
    return max_level


# 落点安全约束（B1 新增）
LANDING_DANGER_MARGIN = 15  # px，落点危险窗口

# v13: 起跳碰撞检测（Takeoff Collision）
# 当敌人非常近时，跳跃动作的前几帧（takeoff 阶段）Mario 还没跳起来，可能直接撞上
TAKEOFF_COLLISION_THRESHOLD_PX = 12  # 敌人距离小于此值时，跳跃有起跳碰撞风险

# v13: 宏动作持续时间相关常量（用于敌人逼近裕量）
# 敌人移动速度估计（Goomba/Koopa 约 1-2 px/frame）
ENEMY_SPEED_PER_FRAME = 1.5  # 保守估计

# 每个 jump_level 对应的预估子步数（frames）
# 来源：SuperMarioEnv 的 step() 实际执行 frames + 落地确认
MACRO_SUBSTEPS_EST = {
    0: 1,    # walk: 1 frame
    1: 10,   # short jump
    2: 12,
    3: 14,
    4: 16,
    5: 18,
    6: 20,   # max jump
}


def estimate_enemy_approach_during_macro(jump_level: int) -> int:
    """
    估算宏动作执行期间敌人可能逼近的距离（像素）。
    
    用于给 PANIC_AHEAD / LANDING_DANGER 添加保守裕量。
    """
    substeps = MACRO_SUBSTEPS_EST.get(int(jump_level), 15)
    return int(substeps * ENEMY_SPEED_PER_FRAME)


def estimate_landing_displacement(jump_level: int) -> int:
    """
    估算跳跃后 Mario 相对起跳位置的水平位移（像素）。
    
    = 跳跃本身的水平位移 + 落地后惯性滑行
    """
    jump_level = max(0, min(6, int(jump_level)))
    jump_dx = JUMP_PARAMS[jump_level][0]
    # walk(0) does not have the post-landing confirmation run segment
    # (that segment is only relevant after a jump/landing).
    if jump_level == 0:
        return MARIO_SPEED_PER_STEP
    return jump_dx + POST_LANDING_RUN_PX


def estimate_landing_danger(
    jump_level: int,
    threats: Sequence[ThreatLike] | None,
    *,
    danger_margin: int = LANDING_DANGER_MARGIN,
) -> tuple[bool, float | None, str]:
    """
    评估某个 jump_level 的落点是否危险（会撞上敌人）。
    
    Args:
        jump_level: 跳跃级别 (0-6)
        threats: 当前可见威胁列表（需包含 name 和 distance 属性）
        danger_margin: 落点危险窗口（默认 15px）
    
    Returns:
        (is_dangerous, min_enemy_dist_after_landing, reason_str)
        - is_dangerous: 落点是否危险
        - min_enemy_dist_after_landing: 落地后距离最近敌人的距离（可能为负，表示在敌人之前）
        - reason_str: 判定理由
    
    计算公式：
        delta_x = JUMP_PARAMS[jump_level][0] + POST_LANDING_RUN_PX
        enemy_dist_after = enemy.distance - delta_x
        is_dangerous = abs(enemy_dist_after) < danger_margin
    """
    if not threats:
        return False, None, "no_threats"
    
    # 只考虑“敌人类威胁”（非 Pipe/Stairs）；未知 Monster 也会被纳入
    ground_enemies = [t for t in threats if is_enemy_threat_name(str(t.name)) and t.distance > 0]
    if not ground_enemies:
        return False, None, "no_ground_enemies"
    
    jump_level = max(0, min(6, int(jump_level)))
    delta_x = estimate_landing_displacement(jump_level)
    landing_dx = int(JUMP_PARAMS[jump_level][0]) if jump_level > 0 else 0
    
    min_dist_after: float | None = None
    closest_enemy_name: str = ""

    # After the full macro-action displacement (jump + post-landing confirmation run),
    # if an enemy is still ahead but too close, the landing is risky.
    min_ahead_dist_after: float | None = None
    closest_ahead_enemy_name: str = ""

    # Even when the enemy is *not* ahead after the full displacement (dist_after <= 0),
    # we can still die during the post-landing confirmation run segment:
    #   - Mario lands at landing_dx (JUMP_PARAMS[level][0])
    #   - then runs right for POST_LANDING_RUN_PX while y stabilizes
    # If an enemy sits between (landing_dx, landing_dx + POST_LANDING_RUN_PX],
    # Mario will run into it on the ground (cannot stomp), which is effectively fatal.
    run_in_after_landing: float | None = None
    run_in_enemy_name: str = ""
    for enemy in ground_enemies:
        dist_after = enemy.distance - delta_x
        if min_dist_after is None or abs(dist_after) < abs(min_dist_after):
            min_dist_after = dist_after
            closest_enemy_name = enemy.name
        # 只把“动作结束后仍在前方”的敌人视为落点风险来源：
        # 如果 dist_after < 0，说明宏动作结束时 Mario 已经超过敌人，通常不会在确认段继续撞上它。
        if dist_after >= 0:
            if min_ahead_dist_after is None or dist_after < min_ahead_dist_after:
                min_ahead_dist_after = dist_after
                closest_ahead_enemy_name = enemy.name

        # Run-in after landing: applies to both single-enemy and multi-enemy scenes.
        # Only relevant for jump actions (jump_level>0); walking does not have this segment.
        if jump_level > 0:
            dist_after_landing = enemy.distance - float(landing_dx)
            # Treat the post-landing confirmation run segment as dangerous.
            # 注意：真正会“落地后跑过去撞死”的窗口长度约为 POST_LANDING_RUN_PX。
            # 这里仅加一个很小的 buffer，避免把“其实仍有足够间隔”的情况误判为必撞
            # （例如 CRITICAL 密集敌人场景下的 lvl6 大跳，误判会导致错误规避）。
            RUN_IN_MARGIN_PX = 4.0
            RUN_IN_WINDOW_PX = float(POST_LANDING_RUN_PX) + RUN_IN_MARGIN_PX
            if 0 < dist_after_landing <= RUN_IN_WINDOW_PX:
                if run_in_after_landing is None or dist_after_landing < run_in_after_landing:
                    run_in_after_landing = dist_after_landing
                    run_in_enemy_name = enemy.name
    
    if min_dist_after is None:
        return False, None, "no_ground_enemies"
    
    # 判定危险：
    # - 1) run-in after landing：敌人在落地点之后的确认段内（必撞）
    # - 2) macro end：动作结束后敌人仍在前方且过近（0~danger_margin）
    is_dangerous = run_in_after_landing is not None
    if (not is_dangerous) and (min_ahead_dist_after is not None) and (0 < float(min_ahead_dist_after) < float(danger_margin)):
        is_dangerous = True

    if is_dangerous and run_in_after_landing is not None:
        reason = f"landing_danger_run_in:{run_in_enemy_name}@{run_in_after_landing:.0f}px"
    elif is_dangerous:
        reason = f"landing_danger:{closest_ahead_enemy_name}@{float(min_ahead_dist_after):.0f}px"
    else:
        reason = f"safe:{closest_enemy_name}@{min_dist_after:.0f}px"
    
    return is_dangerous, min_dist_after, reason


def find_safe_jump_levels(
    min_level_for_obstacle: int,
    threats: Sequence[ThreatLike] | None,
    *,
    max_level: int = 6,
    danger_margin: int = LANDING_DANGER_MARGIN,
) -> list[tuple[int, float | None, str]]:
    """
    找出能跨越障碍物且落点安全的所有 jump_level。
    
    Args:
        min_level_for_obstacle: 跨越当前障碍物（如 Pipe）所需的最小级别
        threats: 当前可见威胁列表
        max_level: 最大允许级别
        danger_margin: 落点危险窗口
    
    Returns:
        list of (level, min_enemy_dist_after, reason)，按 level 升序排列
    """
    safe_levels: list[tuple[int, float | None, str]] = []
    
    for level in range(min_level_for_obstacle, max_level + 1):
        is_dangerous, min_dist, reason = estimate_landing_danger(
            level, threats, danger_margin=danger_margin
        )
        if not is_dangerous:
            safe_levels.append((level, min_dist, reason))
    
    return safe_levels


def select_safest_jump_level(
    min_level_for_obstacle: int,
    threats: Sequence[ThreatLike] | None,
    *,
    max_level: int = 6,
    danger_margin: int = LANDING_DANGER_MARGIN,
    allow_wait: bool = True,
) -> tuple[int, str]:
    """
    选择能跨越障碍物且落点最安全的 jump_level。
    
    策略（Phase 1 改进）：
    1. 优先选择"安全且推进最小"的 level
    2. 如果都危险且 allow_wait=True，返回 jump_level=0（等待/逼近）
    3. 如果都危险且 allow_wait=False，选择"落点在敌人之后且裕量最大"的 level
    
    Args:
        min_level_for_obstacle: 跨越当前障碍物所需的最小级别
        threats: 当前可见威胁列表
        max_level: 最大允许级别
        danger_margin: 落点危险窗口
        allow_wait: 是否允许"等待/逼近"策略（全危险时返回0）
    
    Returns:
        (selected_level, reason)
    """
    safe_levels = find_safe_jump_levels(
        min_level_for_obstacle, threats, max_level=max_level, danger_margin=danger_margin
    )
    
    if safe_levels:
        # 选择安全且推进最小的级别
        best = safe_levels[0]  # 已按 level 升序，第一个就是最小的
        return best[0], f"safe_landing:{best[2]}"
    
    # ===== Phase 1 核心改动 =====
    # 所有级别都危险时，优先"等待/逼近"而非强选危险跳法
    if allow_wait:
        return 0, "all_dangerous_wait:approach_for_safe_window"
    
    # 如果不允许等待（例如障碍物太近必须跳），选择"越过敌人且裕量最大"的
    best_level = min_level_for_obstacle
    best_reason = "all_dangerous_forced"
    best_after: float | None = None
    best_before: float | None = None

    for level in range(min_level_for_obstacle, max_level + 1):
        _, min_dist, reason = estimate_landing_danger(level, threats, danger_margin=danger_margin)
        if min_dist is None:
            continue
        # 落在敌人之后：min_dist <= 0，越负代表越过得越多
        if min_dist <= 0:
            if best_after is None or min_dist < best_after:
                best_after = min_dist
                best_level = level
                best_reason = f"forced_after:{reason}"
            continue
        # 落在敌人之前：min_dist > 0，越大代表离敌人越远
        if best_after is None:
            if best_before is None or min_dist > best_before:
                best_before = min_dist
                best_level = level
                best_reason = f"forced_before:{reason}"

    return best_level, best_reason
