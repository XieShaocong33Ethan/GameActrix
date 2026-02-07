from __future__ import annotations

import re
from typing import Any, Callable

from .pokemon_tools_field_local_map import _best_move_target_from_local_map
from .pokemon_tools_navigation import (
    _extract_map_on_screen,
    _find_horizontal_extreme_reachable_in_map_on_screen,
    _find_northernmost_reachable_in_explored_map,
)
from .pokemon_tools_parsing import _contains_token, _parse_walkable_dirs, _pick_alternative_dir


def append_exit_reds_house_candidates(
    *,
    nav: dict[str, Any],
    pos: tuple[int, int] | None,
    visible_warps: list[tuple[int, int]],
    map_y_max: int | None,
    subtask_step_count: int,
    last_action: str,
    note: Any,
    max_keys_per_step: int,
    add_candidate: Any,
    add_env_tool: Any,
) -> None:
    exit_wp = nav.get("exit_warppoint") if isinstance(nav.get("exit_warppoint"), dict) else None
    exit_steps = int(exit_wp.get("steps", -1)) if isinstance(exit_wp, dict) else -1
    exit_xy: tuple[int, int] | None = None
    if isinstance(exit_wp, dict) and isinstance(exit_wp.get("x"), int) and isinstance(exit_wp.get("y"), int):
        exit_xy = (int(exit_wp["x"]), int(exit_wp["y"]))

    visible_warp_set = {(int(xw), int(yw)) for (xw, yw) in visible_warps}
    exit_visible = bool(exit_xy and exit_xy in visible_warp_set)

    # Prefer key-path navigation using processed_map (works even if the door isn't explored on-screen yet).
    exit_keys = nav.get("next_keys_to_exit_warppoint")
    if isinstance(exit_keys, list) and exit_keys:
        add_candidate(
            exit_keys,
            priority=100,
            why="[Subtask: exit_reds_house] 按 processed_map 最短路径走向出口（比 warp_tool 更鲁棒）",
        )

    # If we are already on the exit WarpPoint at the boundary, press the boundary direction to leave.
    edge_hint = nav.get("edge_warp_hint_dir")
    if exit_steps == 0 and edge_hint:
        add_candidate(
            [str(edge_hint)],
            priority=99,
            why="[Subtask: exit_reds_house] 已站在出口 WarpPoint 边界 → 按边界方向触发换图离开",
        )

    # Only use warp_with_warp_point when the target WarpPoint is visible (explored_map knows it's a WarpPoint),
    # or when we're already standing on it (exit_steps==0).
    if exit_xy and (exit_visible or exit_steps == 0):
        add_env_tool(
            "warp_with_warp_point",
            {"x_dest": int(exit_xy[0]), "y_dest": int(exit_xy[1])},
            priority=90,
            why="[Subtask: exit_reds_house] 出口 WarpPoint 已可见/已站上 → 用 warp_tool 一步完成换图",
        )

    # Fallback strategy: if no nav data available, try going down (typical exit location in RedsHouse1f)
    if not exit_keys and not visible_warps and not exit_wp:
        # Note: Official starter-kit ZIPs may not ship `processed_map/`, so we need a
        # robust fallback that can still leave RedsHouse2f without any WarpPoint hints.
        #
        # Strategy:
        # - If we're already on the bottom edge and repeatedly trying "down" doesn't move us,
        #   probe horizontally along the edge and retry "down" (stairs/edge-warp alignment).
        # - Otherwise, keep a conservative "down" attempt and provide alternatives.

        note_str = str(note or "").lower()
        stuck = ("unchanged" in note_str) or ("oscillating" in note_str)

        last_dir = ""
        last_tokens = str(last_action or "").strip().lower().split()
        if last_tokens and last_tokens[-1] in {"up", "down", "left", "right"}:
            last_dir = last_tokens[-1]

        at_bottom = bool(pos) and isinstance(map_y_max, int) and pos[1] == int(map_y_max)

        # Prefer short key sequences here: long repeats can waste steps when blocked.
        base_try_down = ["down"]

        if at_bottom and stuck and last_dir == "down":
            # Probe: move sideways a bit, then try "down" again.
            dist = max(1, min(3, int(max_keys_per_step) - 1))
            prefer = "left" if (int(subtask_step_count) % 2 == 0) else "right"
            other = "right" if prefer == "left" else "left"

            add_candidate(
                [prefer] * dist + ["down"],
                priority=90,
                why=f"[Subtask: exit_reds_house] Bottom edge + stuck on 'down' → probe {prefer}×{dist} then down",
            )
            add_candidate(
                [other] * dist + ["down"],
                priority=85,
                why=f"[Subtask: exit_reds_house] Bottom edge + stuck on 'down' → probe {other}×{dist} then down",
            )
            add_candidate(
                base_try_down,
                priority=60,
                why="[Subtask: exit_reds_house] 低优先级：再试一次 down（已在边界但可能需要重新对齐）",
            )
            add_candidate(
                ["up"] * max(1, min(2, int(max_keys_per_step))),
                priority=55,
                why="[Subtask: exit_reds_house] 低优先级：向上挪动以改变对齐/视野",
            )
            return

        # Default fallback: always keep a single-step "down" try, then explore alternatives.
        add_candidate(
            base_try_down,
            priority=85,
            why="[Subtask: exit_reds_house] No exit WarpPoint detected → try down (single step, cheaper than repeats)",
        )

        # If we appear stuck on the last direction, avoid repeating it as the top choice.
        alt_dirs = ["left", "right", "up"]
        if stuck and last_dir in alt_dirs:
            alt_dirs = [d for d in alt_dirs if d != last_dir] + [last_dir]

        for i, direction in enumerate(alt_dirs[:3]):
            add_candidate(
                [direction] * max(1, min(3, int(max_keys_per_step))),
                priority=80 - i * 5,
                why=f"[Subtask: exit_reds_house] Backup: try going {direction} to find stairs/exit",
            )


def append_find_oak_or_reach_viridian_city_candidates(
    *,
    obs_text: str,
    sections: dict[str, str],
    nav: dict[str, Any],
    pos: tuple[int, int] | None,
    accepted_subtask: str,
    is_indoor_map: bool,
    subtask_step_count: int,
    max_keys_per_step: int,
    add_candidate: Any,
    add_env_tool: Any,
) -> None:
    # overworld_map_transition only works in outdoor maps, NOT indoor maps
    if is_indoor_map:
        # 先离开室内（例如 Oak's Lab），回到野外再执行“向北走”
        exit_keys = nav.get("next_keys_to_nearest_warppoint")
        if isinstance(exit_keys, list) and exit_keys:
            add_candidate(
                exit_keys,
                priority=100,
                why=f"[Subtask: {accepted_subtask}] 室内地图 → 先按 processed_map 路径走到最近 WarpPoint 离开",
            )
        return

    local_target = _best_move_target_from_local_map(
        map_info_text=sections.get("Map Info", ""),
        pos=pos,
        direction="north",
    )
    if local_target and pos:
        add_env_tool(
            "move_to",
            {"x_dest": int(local_target[0]), "y_dest": int(local_target[1])},
            priority=104,
            why=f"[Subtask: {accepted_subtask}] 基于 Local Map 选视野内北向落脚点 ({local_target[0]},{local_target[1]}) → move_to 绕过房门/障碍",
        )

    north_keys = nav.get("next_keys_to_north_edge") if isinstance(nav, dict) else None
    if isinstance(north_keys, list) and north_keys:
        add_candidate(
            north_keys,
            priority=110,
            why=f"[Subtask: {accepted_subtask}] processed_map 路径：先走到北边界再触发换图/Oak 事件",
        )

    # 野外导航：基于 Map on Screen 用 A* 找最北可达点，用 move_to 绕过障碍物
    # 直接从 obs_text 中提取 "Map on Screen:" 部分（不是 [Section] 格式）
    map_on_screen_data = _extract_map_on_screen(obs_text)
    northernmost_reachable = _find_northernmost_reachable_in_explored_map(
        explored_map_text=map_on_screen_data,
        current_pos=pos,
    )

    moved_north = False
    if northernmost_reachable and pos:
        north_x, north_y = northernmost_reachable
        # 只有当目标在当前位置北边时才使用 move_to
        if north_y < pos[1]:
            moved_north = True
            add_env_tool(
                "move_to",
                {"x_dest": int(north_x), "y_dest": int(north_y)},
                priority=104,
                why=f"[Subtask: {accepted_subtask}] 基于 Map on Screen 用 A* 找到最北可达点 ({north_x},{north_y}) → move_to 绕过障碍物",
            )

    # 如果北上被障碍封死（north_y >= 当前 y），则横向探索以寻找绕行通路，避免死循环调用 overworld_map_transition。
    if not moved_north and pos and pos[1] != 0:
        prefer = "west" if (subtask_step_count % 2 == 0) else "east"
        horizontal_target = _find_horizontal_extreme_reachable_in_map_on_screen(
            explored_map_text=map_on_screen_data,
            current_pos=pos,
            prefer=prefer,
        )
        if horizontal_target:
            hx, hy = horizontal_target
            add_env_tool(
                "move_to",
                {"x_dest": int(hx), "y_dest": int(hy)},
                priority=104,
                why=f"[Subtask: {accepted_subtask}] 北向受阻 → 先向{prefer}侧探索到 ({hx},{hy}) 以发现可绕行通路",
            )

    # Backup: overworld_map_transition (only valid on the map edge).
    can_transition_north = bool(pos) and pos[1] == 0
    if can_transition_north:
        add_env_tool(
            "overworld_map_transition",
            {"direction": "north"},
            priority=100,
            why=f"[Subtask: {accepted_subtask}] 已到北边界 → overworld_map_transition(north) 触发换图",
        )
    # Very conservative fallback: only move one step up, and only when we don't have
    # a structured plan (move_to / processed_map path). This avoids repeatedly walking
    # into nearby doors (WarpPoint) and oscillating between indoor/outdoor maps.
    if not local_target and not north_keys and not moved_north:
        add_candidate(
            ["up"],
            priority=40,
            why=f"[Subtask: {accepted_subtask}] Fallback: move north by one step (only when no move_to/path available)",
        )


def append_get_starter_pokemon_candidates(
    *,
    map_name: str,
    map_name_lower: str,
    is_indoor_map: bool,
    subtask_step_count: int,
    visible_sprites: list[str],
    max_keys_per_step: int,
    rotate_list: Callable[[list[tuple[int, int]], int], list[tuple[int, int]]],
    warppoints_for_map: Callable[[str], list[tuple[int, int]]],
    best_exit_warp: Callable[[], tuple[int, int] | None],
    add_candidate: Any,
    add_env_tool: Any,
) -> None:
    if "oakslab" in map_name_lower:
        pokeballs = sorted([s for s in visible_sprites if "POKE_BALL" in str(s).upper()])
        if pokeballs:
            add_env_tool(
                "interact_with_object",
                {"object_name": pokeballs[0]},
                priority=100,
                why=f"[Subtask: get_starter_pokemon] 在 Oak's Lab → 交互 {pokeballs[0]} 领取初始宝可梦",
            )
        oaks = sorted([s for s in visible_sprites if _contains_token(s, "OAK")])
        # Hidden tests may rename NPC ids. Avoid placeholder object_name; only interact with what we can see.
        if oaks:
            add_env_tool(
                "interact_with_object",
                {"object_name": oaks[0]},
                priority=95,
                why="[Subtask: get_starter_pokemon] 备用：与 Oak 交互推进剧情",
            )
        else:
            others = [s for s in visible_sprites if "POKE_BALL" not in str(s).upper()]
            if others:
                add_env_tool(
                    "interact_with_object",
                    {"object_name": others[0]},
                    priority=90,
                    why=f"[Subtask: get_starter_pokemon] 备用：交互可见对象 {others[0]} 以推进剧情（适应重命名）",
                )
        add_candidate(["a"], priority=80, why="[Subtask: get_starter_pokemon] 备用：按 a 交互/确认")
    elif "pallettown" in map_name_lower and not is_indoor_map:
        warps = warppoints_for_map(map_name)
        rotated = rotate_list(warps, subtask_step_count)
        for i, (xw, yw) in enumerate(rotated[:3]):
            add_env_tool(
                "warp_with_warp_point",
                {"x_dest": int(xw), "y_dest": int(yw)},
                priority=100 - i,
                why=f"[Subtask: get_starter_pokemon] PalletTown 进门找 Oak's Lab → WarpPoint ({xw},{yw})",
            )
        add_candidate(["up"] * max_keys_per_step, priority=70, why="[Subtask: get_starter_pokemon] 备用：往北探索找实验室")
    else:
        exit_xy = best_exit_warp()
        if exit_xy:
            add_env_tool(
                "warp_with_warp_point",
                {"x_dest": int(exit_xy[0]), "y_dest": int(exit_xy[1])},
                priority=90,
                why="[Subtask: get_starter_pokemon] 进错屋/室内 → 先离开回到室外再找 Oak's Lab",
            )


def append_finish_rival_battle_candidates(
    *,
    nav: dict[str, Any],
    visible_warps: list[tuple[int, int]],
    max_keys_per_step: int,
    add_candidate: Any,
    add_env_tool: Any,
) -> None:
    # 经典流程：走向出口会被对手拦下并触发战斗；直接对话不一定能触发。
    exit_keys = nav.get("next_keys_to_nearest_warppoint")
    if isinstance(exit_keys, list) and exit_keys:
        add_candidate(
            exit_keys,
            priority=100,
            why="[Subtask: finish_rival_battle] 往出口方向前进以触发对手战（processed_map 导航）",
        )
    add_candidate(["a"], priority=80, why="[Subtask: finish_rival_battle] 备用：按 a 交互/推进")
    if visible_warps:
        xw, yw = visible_warps[0]
        add_env_tool(
            "warp_with_warp_point",
            {"x_dest": int(xw), "y_dest": int(yw)},
            priority=70,
            why="[Subtask: finish_rival_battle] 备用：尝试进/出门以推进剧情",
        )


def append_anti_stuck_recover_candidates(
    *,
    last_action: str,
    max_keys_per_step: int,
    add_candidate: Any,
) -> None:
    alt = _pick_alternative_dir(last_action=last_action)
    if alt:
        add_candidate([alt] * max_keys_per_step, priority=65, why="[Subtask: anti_stuck_recover] Try different direction to break stuck")


def append_explore_frontier_candidates(
    *,
    sections: dict[str, str],
    add_candidate: Any,
) -> None:
    walkable = _parse_walkable_dirs(sections.get("Map Info", ""))
    if walkable:
        for i, d in enumerate(walkable[:3]):
            add_candidate([d] * 2, priority=85 - i * 5, why=f"[Subtask: explore_frontier] Explore {d} direction")


def append_standard_field_candidates(
    *,
    sections: dict[str, str],
    map_name: str,
    accepted_subtask: str,
    is_indoor_map: bool,
    current_party: str,
    pos: tuple[int, int] | None,
    nav: dict[str, Any],
    note: Any,
    last_action: str,
    max_keys_per_step: int,
    add_candidate: Any,
    add_env_tool: Any,
) -> None:
    # Special early-game heuristic (fallback only): PalletTown with no Pokemon → try going north.
    # 注意：Current Party 里常常会出现 "No more Pokemons" 作为列表结束标记，即便队伍非空也会出现。
    party_has_any_pokemon = bool(re.search(r"^Name:\s*\S", current_party or "", flags=re.MULTILINE))
    if (
        "PalletTown" in map_name
        and (not party_has_any_pokemon)
        and (not is_indoor_map)
        and accepted_subtask not in ("find_oak", "reach_viridian_city")
    ):
        can_transition_north = bool(pos) and pos[1] == 0
        local_target = _best_move_target_from_local_map(
            map_info_text=sections.get("Map Info", ""),
            pos=pos,
            direction="north",
        )
        if local_target and pos:
            add_env_tool(
                "move_to",
                {"x_dest": int(local_target[0]), "y_dest": int(local_target[1])},
                priority=97,
                why="PalletTown 且还没有宝可梦 → 先 move_to 到北向安全落脚点，避免误入房门",
            )
        if can_transition_north:
            add_env_tool(
                "overworld_map_transition",
                {"direction": "north"},
                priority=98,
                why="PalletTown 且还没有宝可梦 → 已到北边界，触发向北换图/Oak 事件",
            )

    # Exit warppoint path (if identifiable): highest priority for indoor navigation.
    exit_keys = nav.get("next_keys_to_exit_warppoint")
    exit_steps = nav.get("exit_warppoint", {}).get("steps", 0) if isinstance(nav.get("exit_warppoint"), dict) else 0
    if (
        accepted_subtask == "return_parcel_to_oak"
        and "OaksLab" in map_name
    ):
        # Do not suggest exiting when we still need to talk to Oak inside the lab.
        exit_keys = None

    if isinstance(exit_keys, list) and exit_keys:
        add_candidate(exit_keys, priority=90, why="Move toward exit WarpPoint (processed_map shortest path)")

    # Edge WarpPoint hint: only use if NOT inside a building with an identifiable exit.
    # This prevents the agent from mistakenly pressing "right" on a staircase warp
    # when the real exit is at the bottom of the map.
    edge_hint = nav.get("edge_warp_hint_dir")
    if edge_hint:
        # For pure navigation subtasks, only trust the edge hint if it matches the intended
        # high-level direction; otherwise it often causes wrong map transitions (e.g. Route1→PalletTown).
        if accepted_subtask in ("find_oak", "reach_viridian_city") and (not is_indoor_map) and str(edge_hint) != "up":
            edge_hint = None
        if accepted_subtask == "return_parcel_to_oak" and (not is_indoor_map) and str(edge_hint) != "down":
            edge_hint = None
        # When we still need to do something inside Oak's Lab (deliver the parcel), do NOT
        # suggest immediately exiting via the door warp (it causes PalletTown↔OaksLab oscillation).
        if accepted_subtask == "return_parcel_to_oak" and "OaksLab" in map_name:
            edge_hint = None

    if edge_hint:
        # If we have a clear exit path with >0 steps, the edge hint might be misleading (e.g., stairs)
        # Only trust edge_hint if: (a) no exit path, or (b) exit_steps == 0 meaning we're ON the exit
        if not exit_keys or exit_steps == 0:
            add_candidate([str(edge_hint)], priority=88, why="Standing on WarpPoint at map edge → press boundary direction to exit")
        else:
            # We're on a warp but there's a different exit → this warp is probably stairs, not exit
            add_candidate([str(edge_hint)], priority=40, why="Standing on WarpPoint (possibly stairs) → edge direction available but exit path preferred")

    # If we already have a strong processed_map edge path for the current milestone navigation
    # subtask, avoid adding distractor candidates (nearest warp / generic anti-stuck).
    #
    # Rationale: On overworld maps, the "nearest WarpPoint" is often a building door. When the
    # model sees an "oscillating" stuck hint, it may pick a 1-step alternative like "up" and
    # repeatedly enter/exit the door, getting stuck near milestone 6→7 (ViridianMart↔ViridianCity).
    has_strong_edge_path = False
    if accepted_subtask in ("find_oak", "reach_viridian_city"):
        nk = nav.get("next_keys_to_north_edge")
        has_strong_edge_path = isinstance(nk, list) and bool(nk)
    elif accepted_subtask == "return_parcel_to_oak":
        sk = nav.get("next_keys_to_south_edge")
        has_strong_edge_path = isinstance(sk, list) and bool(sk)

    # Nearest warppoint path.
    # For milestone navigation subtasks, "nearest WarpPoint" is usually a house/lab door and causes
    # oscillation (enter/exit) without advancing. Prefer edge-path navigation instead.
    #
    # However, in some evaluated states the model may repeatedly insist on walking to the nearest
    # WarpPoint even when it's a suboptimal plan. If we *completely* omit this option from
    # candidates, strict candidate enforcement can hard-fail the whole evaluation. So for
    # `return_parcel_to_oak` we keep it as a low-priority escape hatch.
    nearest_keys = nav.get("next_keys_to_nearest_warppoint")
    if isinstance(nearest_keys, list) and nearest_keys:
        if accepted_subtask in ("find_oak", "reach_viridian_city"):
            pass
        elif accepted_subtask == "return_parcel_to_oak":
            # For return_parcel_to_oak on overworld maps, the nearest WarpPoint is typically a
            # building door (Mart/Center/House). Offering it as a candidate often causes the model
            # to walk into doors and oscillate instead of heading south back to Pallet Town.
            #
            # We therefore only keep this fallback when we are indoors (need a door to leave),
            # or when processed_map is unavailable.
            has_processed_map = bool(nav.get("has_processed_map"))
            if is_indoor_map or (not has_processed_map):
                add_candidate(
                    nearest_keys,
                    priority=55,
                    why="Fallback: move toward nearest WarpPoint (kept for robustness; may be a detour)",
                )
        else:
            add_candidate(nearest_keys, priority=75, why="Move toward nearest WarpPoint (processed_map shortest path)")

    # If we seem stuck, suggest changing direction.
    if isinstance(note, str) and note:
        # When we already have a strong edge path (north/south) for the current milestone navigation
        # subtask, do not offer a generic 1-step alternative. It often points toward nearby doors
        # (WarpPoint) and causes indoor/outdoor oscillation.
        if not has_strong_edge_path:
            alt = _pick_alternative_dir(last_action=last_action)
            if alt:
                add_candidate([alt], priority=65, why=f"Anti-stuck: {note} → try different direction ({alt})")

    # Fallback exploration: if we can read Walkable Dirs line from obs_text, propose one.
    # Avoid offering low-signal local-neighbor fallbacks when we already have a strong
    # processed_map edge path for the current milestone navigation subtask. These
    # fallbacks often cause oscillation (e.g., choosing 'down' on Route1 instead of following north_keys).
    if not has_strong_edge_path:
        walkable = _parse_walkable_dirs(sections.get("Map Info", ""))
        if walkable:
            add_candidate([walkable[0]], priority=50, why="Fallback: pick a walkable direction from local neighbors")

    # Always include a safe pass candidate.
    add_candidate([], priority=10, why="Fallback: pass (no-op) if unsure / waiting animation")
