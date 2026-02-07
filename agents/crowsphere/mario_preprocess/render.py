from __future__ import annotations

from agents.crowsphere.mario_jump_rules import (
    is_enemy_threat_name,
    is_solid_threat_name,
    is_stompable_enemy_name,
)
from agents.crowsphere.mario_preprocess.candidates import MarioCandidates
from agents.crowsphere.mario_preprocess.jump_level_helpers import level_for_height
from agents.crowsphere.mario_preprocess.state import MarioPreprocessState


def render_preprocessed_obs_text(
    *,
    state: MarioPreprocessState,
    candidates: MarioCandidates,
    risk_entries: list[dict],
    mode: str,
) -> str:
    mode_norm = str(mode or "full").strip().lower()
    if mode_norm not in {"full", "evidence_only", "raw"}:
        mode_norm = "full"

    mario_pos = state.mario_pos
    goombas = state.goombas
    koopas = state.koopas
    bricks = state.bricks
    pipes = state.pipes
    stair_blocks = state.stair_blocks
    pit_info = state.pit_info
    vlm_sentinel = state.vlm_sentinel
    threats = state.threats
    nearest_threat = state.nearest_threat
    recommended_level = int(state.recommended_level)
    reason = str(state.recommended_reason)
    urgency = str(state.urgency)
    max_jump_level = int(state.max_jump_level)

    safe_level = int(candidates.safe_level)
    fast_level = candidates.fast_level
    stomp_level = candidates.stomp_level
    dense_safe_level = candidates.dense_safe_level
    stairs_jump_level = candidates.stairs_jump_level

    blocks: list[str] = []

    blocks.append("[Mario State]")
    if mario_pos:
        blocks.append(f"Position: ({mario_pos.x}, {mario_pos.y})")
    else:
        blocks.append("Position: Unknown")
    if state.env_x_pos is not None:
        blocks.append(f"World x_pos: {int(state.env_x_pos)}")
    if state.env_time is not None:
        blocks.append(f"Time: {int(state.env_time)}")
    blocks.append("")

    blocks.append("[Threat Analysis]")
    if nearest_threat:
        blocks.append(f"Nearest: {nearest_threat.name} at ({nearest_threat.position.x}, {nearest_threat.position.y})")
        blocks.append(f"  Distance: {int(nearest_threat.distance)}px")
        blocks.append(f"  Height: {nearest_threat.height}px")
        blocks.append(f"  Urgency: {urgency}")
        needed_level = level_for_height(int(nearest_threat.height))
        blocks.append(f"  Min jump needed: level {needed_level}")
    else:
        blocks.append("No immediate threats ahead")

    if len(threats) > 1:
        blocks.append(f"Next threats: {', '.join(f'{t.name}@{int(t.distance)}px' for t in threats[1:3])}")

    if pit_info and mario_pos:
        pit_start, pit_end = pit_info
        if pit_end > mario_pos.x:
            pit_width = pit_end - pit_start
            blocks.append(f"Pit ahead: width={pit_width}px (start={pit_start}, end={pit_end})")
    blocks.append("")

    if vlm_sentinel.present:
        blocks.append("[VLM Sentinel Status]")
        blocks.append(f"  on_platform: {vlm_sentinel.on_platform}")
        blocks.append(f"  enemy_below_ahead: {vlm_sentinel.enemy_below_ahead}")
        blocks.append(f"  enemy_ahead: {vlm_sentinel.enemy_ahead}")
        blocks.append(f"  enemy_count_ahead: {vlm_sentinel.enemy_count_ahead}")
        blocks.append(f"  dense_enemies_ahead: {vlm_sentinel.dense_enemies_ahead}")
        blocks.append(f"  pipe_ahead: {vlm_sentinel.pipe_ahead} ({vlm_sentinel.pipe_distance})")
        blocks.append(f"  pit_ahead: {vlm_sentinel.pit_ahead} ({vlm_sentinel.pit_distance})")
        blocks.append(f"  stairs_ahead: {vlm_sentinel.stairs_ahead}")
        blocks.append(f"  stairs_pit_ahead: {vlm_sentinel.stairs_pit_ahead}")
        blocks.append(f"  low_ceiling: {vlm_sentinel.low_ceiling}")
        blocks.append("")

    if mode_norm == "full":
        blocks.append("[Action]")
        blocks.append(f">>> jump_level={recommended_level} <<<")
        blocks.append(f"Reason: {reason}")
        blocks.append(f"Constraints: max_jump_level={max_jump_level}")
        blocks.append("")
    else:
        blocks.append("[Constraints]")
        blocks.append(f"Constraints: max_jump_level={max_jump_level}")
        blocks.append("")

    blocks.append("[Decision Support]")
    if pit_info and mario_pos:
        intent = "EVADE"
    elif nearest_threat and is_enemy_threat_name(str(nearest_threat.name)):
        # 未知敌人默认不建议 ATTACK（避免踩到“不可踩/带刺”的敌人）。
        if is_stompable_enemy_name(str(nearest_threat.name)) and safe_level < 3:
            intent = "ATTACK"
        else:
            intent = "EVADE"
    elif nearest_threat and is_solid_threat_name(str(nearest_threat.name)):
        intent = "PROGRESS"
    else:
        intent = "PROGRESS"
    blocks.append(f"Intent: {intent}")
    blocks.append("Candidates:")
    safe_note = "最稳（工具安全选项）"
    blocks.append(f"- SAFE: jump_level={safe_level} intent={intent} note=\"{safe_note}\"")
    if fast_level is not None:
        blocks.append("- FAST: jump_level={jl} intent=PROGRESS note=\"赶路/省步数；会避开 Pipe/Stairs 撞击与落点危险\"".format(jl=int(fast_level)))
    if stomp_level is not None:
        blocks.append(f"- STOMP: jump_level={int(stomp_level)} intent=ATTACK note=\"尝试踩最近的地面敌人（若 Risk=OK）\"")
    if dense_safe_level is not None:
        if int(dense_safe_level) == 0:
            blocks.append(
                "- DENSE_SAFE: jump_level=0 intent=EVADE note=\"VLM检测到密集敌人：最慢速度行走（非原地等待），降低贴脸风险/让视野逐步展开\""
            )
        else:
            dense_note = "VLM检测到密集敌人，大跳跨过敌人簇" if int(dense_safe_level) >= 5 else "VLM检测到密集敌人，小跳避开敌人群"
            blocks.append(f"- DENSE_SAFE: jump_level={int(dense_safe_level)} intent=EVADE note=\"{dense_note}\"")
    if stairs_jump_level is not None:
        blocks.append(f"- STAIRS_JUMP: jump_level={int(stairs_jump_level)} intent=EVADE note=\"VLM检测到台阶坑，高跳越过\"")
    blocks.append("- SLOW_WALK: jump_level=0 intent=PROGRESS note=\"最慢速度行走（非原地等待）\"")
    blocks.append("")

    blocks.append("[Risk Report]")
    for r in risk_entries:
        name = r.get("name")
        jl = r.get("jump_level")
        dx = r.get("dx_est")
        flags: list[str] = []
        if r.get("takeoff_collision"):
            flags.append(f"TAKEOFF_COLLISION@{r.get('takeoff_enemy_dist')}px")
        if r.get("would_hit_ceiling"):
            flags.append(f"HIT_CEILING@{r.get('ceiling_block_dist')}px")
        if r.get("would_hit_solid"):
            solid = r.get("solid") or {}
            flags.append(f"HIT_{solid.get('name')}@{solid.get('dist_px')}px")
        if r.get("pit_landing"):
            flags.append("LAND_IN_PIT")
        if r.get("pit_too_close_to_jump"):
            flags.append("PIT_TOO_CLOSE_TO_JUMP")
        if r.get("pit_edge_too_close_after"):
            flags.append(f"PIT_EDGE_TOO_CLOSE@{r.get('pit_edge_dist_after')}px")
        if r.get("pit_edge_near_after"):
            flags.append(f"NEAR_PIT_EDGE@{r.get('pit_edge_near_dist_after')}px")
        if r.get("landing_danger"):
            flags.append(f"LANDING_DANGER({r.get('landing_danger_reason')})")
        if r.get("panic_ahead_after_landing"):
            flags.append(f"PANIC_AHEAD@{r.get('panic_dist_after')}px")
        flag_str = "OK" if not flags else ", ".join(flags)
        blocks.append(f"- {name}: jl={jl} dx≈{dx}px => {flag_str}")
    blocks.append("")

    blocks.append("[Objects]")
    if goombas:
        blocks.append(f"Goomba: {', '.join(f'({p.x},{p.y})' for p in goombas[:3])}")
    if koopas:
        blocks.append(f"Koopa: {', '.join(f'({p.x},{p.y})' for p in koopas[:3])}")
    unknown_enemies = [
        t
        for t in threats
        if is_enemy_threat_name(str(t.name))
        and (str(t.name) not in {"Goomba", "Koopa", "Koopa_shadow"})
        and float(t.distance) > 0
    ]
    if unknown_enemies:
        blocks.append(
            "Unknown Monster: "
            + ", ".join(f"{t.name}@({t.position.x},{t.position.y})" for t in unknown_enemies[:3])
        )
    if pipes:
        blocks.append(f"Pipe: {', '.join(f'({p.x},{p.y},h={p.height})' for p in pipes[:3])}")
    if bricks:
        blocks.append(f"Brick: {', '.join(f'({p.x},{p.y})' for p in bricks[:3])}")
    if stair_blocks:
        blocks.append(f"Stairs: {', '.join(f'({p.x},{p.y})' for p in stair_blocks[:3])}")

    return "\n".join(blocks).strip()
