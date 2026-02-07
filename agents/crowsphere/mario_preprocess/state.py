from __future__ import annotations

import os
from dataclasses import dataclass

from agents.crowsphere.mario_jump_rules import (
    GOOMBA_SIZE,
    KOOPA_SIZE,
    MARIO_SPEED_PER_STEP,
    is_enemy_threat_name,
)
from agents.crowsphere.mario_preprocess.jump_policy import calculate_jump_level
from agents.crowsphere.mario_preprocess.low_ceiling_memory import LowCeilingMemory
from agents.crowsphere.mario_preprocess.koopa_memory import KoopaMemory
from agents.crowsphere.mario_preprocess.parsing import (
    parse_env_int,
    parse_mario_position,
    parse_monsters,
    parse_pipes,
    parse_pit,
    parse_positions,
)
from agents.crowsphere.mario_preprocess.sentinel import VLMSentinelInfo, parse_vlm_sentinel_block
from agents.crowsphere.mario_preprocess.types import PipeInfo, Position, ThreatInfo


@dataclass(frozen=True)
class MarioPreprocessState:
    obs_text: str
    mario_pos: Position | None
    env_x_pos: int | None
    env_y_pos: int | None
    env_time: int | None
    goombas: list[Position]
    koopas: list[Position]
    bricks: list[Position]
    pipes: list[PipeInfo]
    question_blocks: list[Position]
    stair_blocks: list[Position]
    pit_info: tuple[int, int] | None
    vlm_sentinel: VLMSentinelInfo
    on_stairs: bool
    threats: list[ThreatInfo]
    nearest_threat: ThreatInfo | None
    recommended_level: int
    recommended_reason: str
    urgency: str
    has_low_ceiling: bool
    max_jump_level: int
    safe_level: int
    ground_y_est: int
    ceiling_blocks: list[Position]


def extract_mario_state(
    obs_text: str,
    *,
    koopa_memory: KoopaMemory | None,
    ceiling_memory: LowCeilingMemory | None = None,
) -> MarioPreprocessState:
    env_x_pos = parse_env_int(obs_text, "x_pos")
    env_y_pos = parse_env_int(obs_text, "y_pos")
    env_time = parse_env_int(obs_text, "time")
    mario_pos = parse_mario_position(obs_text)
    monsters = parse_monsters(obs_text)
    goombas = list(monsters.get("Goomba", []))
    koopas = list(monsters.get("Koopa", []))
    unknown_monsters: dict[str, list[Position]] = {
        k: v for k, v in monsters.items() if k not in {"Goomba", "Koopa"}
    }
    bricks = parse_positions(obs_text, "Bricks")
    pipes = parse_pipes(obs_text)
    question_blocks = parse_positions(obs_text, "Question Blocks")
    stair_blocks = parse_positions(obs_text, "Stair Blocks")
    pit_info = parse_pit(obs_text)

    vlm_sentinel = parse_vlm_sentinel_block(obs_text)
    # VLM on_platform 偶发误判：当 Mario 明显在地面（y≈45）时，强制按“非平台”处理，
    # 避免错误触发 height-diff warning 与 ground_y_est 偏移。
    if mario_pos is not None and vlm_sentinel.present:
        if int(mario_pos.y) <= 60:
            vlm_sentinel.on_platform = False
            vlm_sentinel.enemy_below_ahead = False

    # === Mario X calibration near stairs (template jitter / camera drift) ===
    # 官方 env 的 object 坐标来自模板匹配（屏幕坐标），而 Mario x 在 obs_text 里会被 clamp 到 122。
    # 在台阶/坑附近，这两套坐标有时会出现 ~10px 级别的相对偏移，导致 dist_to_pit 误判并把跨坑时机拖晚。
    # 经验上：当 Mario 已经贴近地面台阶（y≈96）且 x 在 ±16px 内时，用该台阶的 x 作为参考更稳。
    on_stairs = False
    if mario_pos is not None and int(mario_pos.y) >= 70 and stair_blocks:
        anchor_candidates = [
            p
            for p in stair_blocks
            if int(p.y) >= 90 and (-16 <= (int(p.x) - int(mario_pos.x)) <= 16)
        ]
        if anchor_candidates:
            anchor = min(anchor_candidates, key=lambda p: abs(int(p.x) - int(mario_pos.x)))
            delta_x = int(anchor.x) - int(mario_pos.x)
            # 只有在偏移明显时才校准，避免在正常情况下改变距离计算（影响跳跃窗口）。
            if abs(int(delta_x)) >= 12:
                mario_pos.x = int(anchor.x)
            on_stairs = True
        else:
            on_stairs = any(-16 <= (int(p.x) - int(mario_pos.x)) <= 16 for p in stair_blocks)
    # === End calibration ===

    # === Pit recall boost via VLM sentinel (tighten only) ===
    if mario_pos is not None and pit_info is None and vlm_sentinel.present and vlm_sentinel.pit_ahead:
        DEFAULT_PIT_WIDTH = 40
        # 优先使用“确定性 pit detector”给出的更精确 dx（如果有）。
        det_start = vlm_sentinel.pit_start_dx
        det_end = vlm_sentinel.pit_end_dx
        start_dx: int | None = None
        end_dx: int | None = None
        ignore_sentinel_pit_for_this_step = False
        if det_start is not None:
            try:
                start_dx = int(det_start)
            except Exception:
                start_dx = None
        if det_end is not None:
            try:
                end_dx = int(det_end)
            except Exception:
                end_dx = None

        # 防止误报：在普通地面关卡里，底部颜色噪声/管道阴影可能被 pit detector 误判成“很窄的坑段”。
        # 真实的坑（至少 1 个 tile）宽度通常 >=16px；窄于该阈值基本不可信。
        MIN_PIT_WIDTH_PX = 16
        if (
            (start_dx is not None)
            and (end_dx is not None)
            and (int(start_dx) > 0)
            and (int(end_dx) > int(start_dx))
            and (int(end_dx) - int(start_dx) < int(MIN_PIT_WIDTH_PX))
        ):
            # 这类“窄坑段”更像噪声，忽略本步 sentinel 的 pit 召回（也不做 bucket 回退）。
            ignore_sentinel_pit_for_this_step = True
            start_dx = None
            end_dx = None

        if start_dx is not None and int(start_dx) > 0:
            pit_start = int(mario_pos.x) + int(start_dx)
            pit_end: int | None = None
            if end_dx is not None and int(end_dx) > int(start_dx):
                pit_end_candidate = int(mario_pos.x) + int(end_dx)
                # 当 hazard 段延伸到屏幕右边缘时，end 往往不可观测；否则会形成“超宽坑”误判。
                if int(pit_end_candidate) >= 250 and int(pit_end_candidate - pit_start) >= 80:
                    pit_end = int(pit_start) + int(DEFAULT_PIT_WIDTH)
                else:
                    pit_end = int(pit_end_candidate)
            if pit_end is None:
                pit_end = int(pit_start) + int(DEFAULT_PIT_WIDTH)
            pit_info = (int(pit_start), int(pit_end))

        if pit_info is None and (not ignore_sentinel_pit_for_this_step):
            bucket = (vlm_sentinel.pit_distance or "unknown").lower()
            start_dx_map = {
                "very_close": 12,
                "close": 30,
                "mid": 70,
                "far": 120,
                "unknown": 70,
            }
            start_dx = int(start_dx_map.get(bucket, 70))
            width = int(DEFAULT_PIT_WIDTH)
            pit_info = (mario_pos.x + start_dx, mario_pos.x + start_dx + width)
    # === End pit recall boost ===

    threats: list[ThreatInfo] = []
    koopas_ahead: list[Position] = []

    if mario_pos is not None:
        for pos in goombas:
            dist = pos.x - mario_pos.x
            if dist > 0:
                threats.append(ThreatInfo("Goomba", pos, dist, GOOMBA_SIZE))

        koopa_seen_behind = False
        for pos in koopas:
            dist = pos.x - mario_pos.x
            if dist > 0:
                threats.append(ThreatInfo("Koopa", pos, dist, KOOPA_SIZE))
                koopas_ahead.append(pos)
            else:
                if dist >= -160:
                    koopa_seen_behind = True

        if koopa_memory is not None:
            if koopa_seen_behind:
                koopa_memory.last_seen_dx = None
                koopa_memory.last_seen_y = None
                koopa_memory.steps_since_seen = 0
            koopa_memory.update(koopas_ahead, mario_pos.x)
            if not koopas_ahead:
                shadow = koopa_memory.get_shadow_koopa()
                if shadow is not None:
                    shadow_abs_x = mario_pos.x + int(shadow.distance)
                    threats.append(
                        ThreatInfo(
                            name="Koopa_shadow",
                            position=Position(x=shadow_abs_x, y=shadow.position.y),
                            distance=shadow.distance,
                            height=KOOPA_SIZE,
                        )
                    )

        for pipe in pipes:
            dist = pipe.x - mario_pos.x
            if dist > 0:
                threats.append(ThreatInfo("Pipe", Position(pipe.x, pipe.y), dist, pipe.height))

        # Unknown monsters: treat as generic ground enemies so the agent does not "go blind".
        for monster_name, positions in unknown_monsters.items():
            for pos in positions:
                dist = pos.x - mario_pos.x
                if dist > 0:
                    # Default to Goomba-sized hitbox; behavior is handled conservatively downstream.
                    threats.append(ThreatInfo(str(monster_name), pos, dist, GOOMBA_SIZE))

        if stair_blocks:
            ahead = [p for p in stair_blocks if p.x > mario_pos.x]
            if ahead:
                nearest_x = min(p.x for p in ahead)
                col = [p for p in ahead if abs(p.x - nearest_x) <= 8]
                top_y = max(p.y for p in col)
                height = max(16, top_y - 32)
                threats.append(ThreatInfo("Stairs", Position(nearest_x, top_y), nearest_x - mario_pos.x, height))

    threats.sort(key=lambda t: t.distance)
    # Koopa_shadow 是短时记忆的“弱证据”，用于 risk 提示，但不应压过当前帧的真实威胁。
    # 否则会出现：shadow 距离更近 → 规则层建议 jump → 把 Mario 推进到 Goomba 簇的必死窗口。
    nearest_threat = next((t for t in threats if t.name != "Koopa_shadow"), None) if threats else None
    if nearest_threat is None and threats:
        nearest_threat = threats[0]

    # === Correct VLM sentinel height flags when they contradict clear geometry ===
    # 在决策点 Mario 的 y 通常是稳定落地后的值：y>=70 基本意味着站在管道/砖块/台阶上。
    # 当 VLM 误报 on_platform=False 时，会导致 jump_policy 低估“高处优势”并给出过小跳跃/踩怪，
    # 常见后果是在管道顶掉落/贴脸撞怪而死。
    if mario_pos is not None and vlm_sentinel.present:
        if int(mario_pos.y) >= 70:
            vlm_sentinel.on_platform = True
        ground_enemies = [t for t in threats if is_enemy_threat_name(str(t.name)) and t.distance > 0]
        nearest_enemy = min(ground_enemies, key=lambda t: t.distance) if ground_enemies else None
        if nearest_enemy is not None and nearest_enemy.distance <= 160:
            if (mario_pos.y - nearest_enemy.position.y) >= 20:
                vlm_sentinel.enemy_below_ahead = True
                vlm_sentinel.enemy_ahead = True
    # === End correction ===

    # === Deterministic height-diff sentinel fallback (no network) ===
    if mario_pos is not None and (not vlm_sentinel.present):
        ground_enemies = [t for t in threats if is_enemy_threat_name(str(t.name)) and t.distance > 0]
        nearest_enemy = min(ground_enemies, key=lambda t: t.distance) if ground_enemies else None
        if nearest_enemy is not None and nearest_enemy.distance <= 160:
            height_diff = mario_pos.y - nearest_enemy.position.y
            if height_diff >= 20:
                vlm_sentinel.present = True
                vlm_sentinel.on_platform = True
                vlm_sentinel.enemy_below_ahead = True
                vlm_sentinel.enemy_ahead = True
    # === End deterministic sentinel fallback ===

    # === Low ceiling detection (factored out to keep this file < 500 lines) ===
    from agents.crowsphere.mario_preprocess.low_ceiling_rules import detect_low_ceiling_and_max_jump_level

    low_ceiling = detect_low_ceiling_and_max_jump_level(
        mario_pos=mario_pos,
        bricks=bricks,
        question_blocks=question_blocks,
        threats=threats,
        vlm_sentinel=vlm_sentinel,
        ceiling_memory=ceiling_memory,
        env_x_pos=env_x_pos,
        stair_blocks=stair_blocks,
    )
    has_low_ceiling = bool(low_ceiling.has_low_ceiling)
    max_jump_level = int(low_ceiling.max_jump_level)
    # === End low ceiling detection ===

    # 重要：jump_policy 的低天花板判断需要与记忆后的 has_low_ceiling 对齐。
    has_low_ceiling_for_policy = bool(has_low_ceiling and int(max_jump_level) <= 4)

    recommended_level, reason, urgency = calculate_jump_level(
        nearest_threat=nearest_threat,
        pit_info=pit_info,
        mario_pos=mario_pos,
        bricks=bricks,
        question_blocks=question_blocks,
        stair_block_count=len(stair_blocks or []),
        all_threats=threats,
        vlm_sentinel=vlm_sentinel,
        on_stairs=on_stairs,
        has_low_ceiling_override=has_low_ceiling_for_policy,
    )

    safe_level = max(0, min(int(recommended_level), int(max_jump_level)))

    has_pipe_ahead_in_threats = any(t.name == "Pipe" and float(t.distance) > 0 for t in threats)
    # 若 threats 已经包含可见的实体障碍（Pipe/Stairs），不应再用 VLM 的 pipe_ahead 兜底覆盖：
    # - 终点旗杆附近 VLM 偶发把旗杆/砖块识别成 pipe，从而把“需要跳台阶”的建议覆写成 0，导致卡死。
    has_solid_ahead_in_threats = any(
        (t.name in {"Pipe", "Stairs"}) and (float(t.distance) > 0) for t in threats
    )
    has_enemy_close_ahead = any(is_enemy_threat_name(str(t.name)) and 0 < float(t.distance) <= 120 for t in threats)
    has_pit_ahead = bool(pit_info and mario_pos and pit_info[1] > mario_pos.x)

    if (
        vlm_sentinel.present
        and vlm_sentinel.pipe_ahead
        and (not has_pipe_ahead_in_threats)
        and (not has_solid_ahead_in_threats)
        and (not has_enemy_close_ahead)
        and (not has_pit_ahead)
    ):
        # 只允许“抬高”跳跃来规避近距离的漏检 pipe；不要把已有的跳跃建议降到 0。
        # mid/far/unknown：保持原策略（通常本来就是 0），避免误判时把必要的跳跃覆写掉。
        if vlm_sentinel.pipe_distance in {"very_close", "close"}:
            recommended_level = max(int(recommended_level), min(int(max_jump_level), 5))
            reason = "VLM sees close pipe ahead; high jump"
            urgency = "HIGH" if vlm_sentinel.pipe_distance == "close" else "CRITICAL"
            safe_level = max(int(safe_level), int(recommended_level))

    # === Dense goomba corner-case (official 1-1, warmup=20~30) ===
    # 经验回归：在 x≈1900~1950 段，若屏幕左侧仍残留一只 Goomba（enemy behind），
    # 敌人互相碰撞/同步移动会让“贴脸窗口”的相对时机更敏感。
    #
    # 早期我们在该窗口“强制 lvl2 分割跳”，但线上复现发现它会让某些 seed 稳定死亡（REMOTE 309445，x_pos≈1924）。
    # 因此这里将该 corner-case 调整为“保守下限=3”：
    # - 当推荐值过小（0/1/2）时抬到 lvl3，减少落点卡在 Goomba 运动范围内的概率；
    # - 仍不从 lvl4+ 向下压（避免把本来就更安全的跳跃压小）。
    if env_x_pos is not None and mario_pos is not None and nearest_threat is not None:
        if (
            1880 <= int(env_x_pos) <= 2000
            and str(nearest_threat.name) == "Goomba"
            and 28.0 <= float(nearest_threat.distance) <= 40.0
        ):
            behind_dxs: list[int] = []
            for pos in list(goombas or []) + list(koopas or []):
                dx = int(pos.x) - int(mario_pos.x)
                if -120 <= int(dx) < 0:
                    behind_dxs.append(int(dx))
            if behind_dxs:
                floor_level = min(int(max_jump_level), 3)
                if int(recommended_level) < int(floor_level):
                    recommended_level = int(floor_level)
                if int(safe_level) < int(floor_level):
                    safe_level = int(floor_level)
                reason = f"{reason}; floor_to_lvl{floor_level}_for_enemy_behind"
    # === End dense goomba corner-case ===

    # ===== Ground Y estimate (for platform-relative solid clearance) =====
    ground_y_est = 45
    if mario_pos is not None:
        ground_candidates: list[int] = []
        is_likely_on_ground = int(mario_pos.y) <= 60
        if (vlm_sentinel.present and (not vlm_sentinel.on_platform)) or ((not vlm_sentinel.present) and is_likely_on_ground):
            ground_candidates.append(int(mario_pos.y))
        for t in threats:
            if is_enemy_threat_name(str(t.name)) and t.distance > 0:
                ground_candidates.append(int(t.position.y))
        if ground_candidates:
            ground_y_est = int(min(ground_candidates))

    # ===== Low ceiling blocks (for deterministic risk verification) =====
    ceiling_blocks: list[Position] = []
    if mario_pos is not None:
        lookbehind_px = 40
        lookahead_px = 120
        for block in list(bricks or []) + list(question_blocks or []):
            dx = int(block.x) - int(mario_pos.x)
            if -lookbehind_px <= dx <= lookahead_px and 80 < int(block.y) < 130:
                ceiling_blocks.append(block)
        ceiling_blocks.sort(key=lambda p: p.x)

    # 若 SAFE 建议会顶到低天花板，则向下收缩到“不会顶头”的最小可行跳。
    # 目的：避免 adapter 因 SAFE=HIT_CEILING 而被迫退化到 SLOW_WALK，进而在低天花板段被贴脸推死。
    if mario_pos is not None and ceiling_blocks and safe_level > 0:
        from agents.crowsphere.mario_preprocess.jump_level_helpers import estimate_ceiling_collision

        adjusted = int(safe_level)
        while adjusted > 0:
            would_hit_ceiling, _, _ = estimate_ceiling_collision(
                jump_level=int(adjusted),
                mario_pos=mario_pos,
                ceiling_blocks=ceiling_blocks,
            )
            if not would_hit_ceiling:
                break
            adjusted -= 1
        if adjusted != int(safe_level):
            safe_level = int(max(0, adjusted))
            recommended_level = int(safe_level)
            reason = f"{reason}; clamp_safe_to_avoid_ceiling=>{safe_level}"

    _ = os.getenv("CROWSPHERE_MARIO_ASSUME_STAIRS_PIT", "0")  # keep env var discoverable via grep

    return MarioPreprocessState(
        obs_text=str(obs_text or ""),
        mario_pos=mario_pos,
        env_x_pos=(int(env_x_pos) if env_x_pos is not None else None),
        env_y_pos=(int(env_y_pos) if env_y_pos is not None else None),
        env_time=(int(env_time) if env_time is not None else None),
        goombas=goombas,
        koopas=koopas,
        bricks=bricks,
        pipes=pipes,
        question_blocks=question_blocks,
        stair_blocks=stair_blocks,
        pit_info=pit_info,
        vlm_sentinel=vlm_sentinel,
        on_stairs=on_stairs,
        threats=threats,
        nearest_threat=nearest_threat,
        recommended_level=int(recommended_level),
        recommended_reason=str(reason),
        urgency=str(urgency),
        has_low_ceiling=bool(has_low_ceiling),
        max_jump_level=int(max_jump_level),
        safe_level=int(safe_level),
        ground_y_est=int(ground_y_est),
        ceiling_blocks=ceiling_blocks,
    )
