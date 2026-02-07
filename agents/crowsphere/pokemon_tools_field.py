from __future__ import annotations

import re
from typing import Any, Callable

from .pokemon_tools_navigation import _load_coll_map_case_insensitive
from .pokemon_tools_parsing import _contains_token
from .pokemon_tools_field_subtasks_basic import (
    append_anti_stuck_recover_candidates,
    append_explore_frontier_candidates,
    append_exit_reds_house_candidates,
    append_find_oak_or_reach_viridian_city_candidates,
    append_finish_rival_battle_candidates,
    append_get_starter_pokemon_candidates,
    append_standard_field_candidates,
)
from .pokemon_tools_field_subtasks_parcel import (
    append_obtain_oaks_parcel_candidates,
    append_return_parcel_to_oak_candidates,
)


def append_field_state_candidates(
    *,
    obs_text: str,
    sections: dict[str, str],
    map_name: str,
    pos: tuple[int, int] | None,
    nav: dict[str, Any],
    accepted_subtask: str,
    mem: dict[str, Any],
    last_action: str,
    note: Any,
    current_party: str,
    visible_warps: list[tuple[int, int]],
    visible_signs_by_warp: dict[tuple[int, int], list[str]],
    visible_sprites: list[str],
    max_keys_per_step: int,
    add_candidate: Any,
    add_env_tool: Any,
) -> None:
    map_name_lower = (map_name or "").lower()
    is_indoor_map = any(kw in map_name_lower for kw in ["house", "lab", "mart", "center", "gym", "gate", "room"])
    map_info_block = sections.get("Map Info", "")
    map_y_max: int | None = None
    m_size = re.search(r"\(x_max , y_max\):\s*\((\d+),\s*(\d+)\)", map_info_block or "")
    if m_size:
        try:
            map_y_max = int(m_size.group(2))
        except Exception:
            map_y_max = None

    subtask_step_count = int(mem.get("subtask_step_count", 0) or 0)

    def _rotate_list(items: list[tuple[int, int]], offset: int) -> list[tuple[int, int]]:
        if not items:
            return []
        off = offset % len(items)
        return items[off:] + items[:off]

    def _warppoints_for_map(name: str) -> list[tuple[int, int]]:
        coll_map = _load_coll_map_case_insensitive(name)
        if not coll_map:
            return []
        warps = [(x, y) for y, row in enumerate(coll_map) for x, v in enumerate(row) if str(v) == "WarpPoint"]
        # Prefer scanning north-side doors first (smaller y), then x.
        warps.sort(key=lambda p: (p[1], p[0]))
        return warps

    def _best_exit_warp() -> tuple[int, int] | None:
        exit_wp = nav.get("exit_warppoint") if isinstance(nav.get("exit_warppoint"), dict) else None
        if isinstance(exit_wp, dict) and isinstance(exit_wp.get("x"), int) and isinstance(exit_wp.get("y"), int):
            return int(exit_wp["x"]), int(exit_wp["y"])
        nearest_wp = nav.get("nearest_warppoint") if isinstance(nav.get("nearest_warppoint"), dict) else None
        if isinstance(nearest_wp, dict) and isinstance(nearest_wp.get("x"), int) and isinstance(nearest_wp.get("y"), int):
            return int(nearest_wp["x"]), int(nearest_wp["y"])
        if visible_warps:
            return int(visible_warps[0][0]), int(visible_warps[0][1])
        return None

    # Interactions on visible sprites are helpful, but should not dominate pure navigation subtasks.
    sprite_ranked: list[tuple[int, str, str]] = []
    for sprite_name in visible_sprites:
        upper_name = str(sprite_name).upper()
        pr = 0
        why = ""
        if _contains_token(upper_name, "OAK"):
            if accepted_subtask in ("find_oak", "get_starter_pokemon", "return_parcel_to_oak"):
                pr = 99
                why = "看到 SPRITE_OAK → 当前子任务需要与 Oak 交互推进里程碑"
            else:
                pr = 40
                why = f"看到 SPRITE_OAK → 可交互（当前子任务={accepted_subtask}，通常不优先）"
        elif "BLUE" in upper_name and accepted_subtask == "finish_rival_battle":
            pr = 99
            why = "看到 SPRITE_BLUE（对手）→ 交互可能触发/推进对手战"
        elif accepted_subtask == "obtain_oaks_parcel" and "CLERK" in upper_name:
            pr = 99
            why = "在商店目标中看到店员 → 交互拿包裹"
        elif accepted_subtask == "get_starter_pokemon" and "POKE_BALL" in upper_name:
            pr = 98
            why = "在初始宝可梦目标中看到精灵球 → 交互领取初始宝可梦"
        else:
            # During pure navigation subtasks, random NPC interactions are almost always a distraction
            # (e.g., talking to Route1 youngsters) and waste the limited 200-step budget.
            if accepted_subtask in ("find_oak", "reach_viridian_city", "return_parcel_to_oak"):
                continue
            pr = 80 if accepted_subtask in ("get_starter_pokemon", "obtain_oaks_parcel", "return_parcel_to_oak") else 50
            why = f"可交互对象 {sprite_name} → 交互（低优先级备选）"
        sprite_ranked.append((pr, sprite_name, why))

    sprite_ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
    for pr, sprite_name, why in sprite_ranked[:3]:
        add_env_tool(
            "interact_with_object",
            {"object_name": sprite_name},
            priority=pr,
            why=why,
        )

    # Subtask-specific candidates
    if accepted_subtask == "exit_reds_house":
        append_exit_reds_house_candidates(
            nav=nav,
            pos=pos,
            visible_warps=visible_warps,
            map_y_max=map_y_max,
            subtask_step_count=subtask_step_count,
            last_action=last_action,
            note=note,
            max_keys_per_step=max_keys_per_step,
            add_candidate=add_candidate,
            add_env_tool=add_env_tool,
        )
    elif accepted_subtask in ("find_oak", "reach_viridian_city"):
        append_find_oak_or_reach_viridian_city_candidates(
            obs_text=obs_text,
            sections=sections,
            nav=nav,
            pos=pos,
            accepted_subtask=accepted_subtask,
            is_indoor_map=is_indoor_map,
            subtask_step_count=subtask_step_count,
            max_keys_per_step=max_keys_per_step,
            add_candidate=add_candidate,
            add_env_tool=add_env_tool,
        )
    elif accepted_subtask == "return_parcel_to_oak":
        append_return_parcel_to_oak_candidates(
            obs_text=obs_text,
            sections=sections,
            map_name=map_name,
            map_name_lower=map_name_lower,
            pos=pos,
            nav=nav,
            is_indoor_map=is_indoor_map,
            map_y_max=map_y_max,
            mem=mem,
            last_action=last_action,
            note=note,
            visible_warps=visible_warps,
            visible_sprites=visible_sprites,
            subtask_step_count=subtask_step_count,
            max_keys_per_step=max_keys_per_step,
            best_exit_warp=_best_exit_warp,
            add_candidate=add_candidate,
            add_env_tool=add_env_tool,
        )
    elif accepted_subtask == "obtain_oaks_parcel":
        append_obtain_oaks_parcel_candidates(
            obs_text=obs_text,
            sections=sections,
            map_name=map_name,
            map_name_lower=map_name_lower,
            pos=pos,
            nav=nav,
            is_indoor_map=is_indoor_map,
            mem=mem,
            visible_warps=visible_warps,
            visible_signs_by_warp=visible_signs_by_warp,
            visible_sprites=visible_sprites,
            subtask_step_count=subtask_step_count,
            max_keys_per_step=max_keys_per_step,
            rotate_list=_rotate_list,
            best_exit_warp=_best_exit_warp,
            add_candidate=add_candidate,
            add_env_tool=add_env_tool,
        )
    elif accepted_subtask == "get_starter_pokemon":
        append_get_starter_pokemon_candidates(
            map_name=map_name,
            map_name_lower=map_name_lower,
            is_indoor_map=is_indoor_map,
            subtask_step_count=subtask_step_count,
            visible_sprites=visible_sprites,
            max_keys_per_step=max_keys_per_step,
            rotate_list=_rotate_list,
            warppoints_for_map=_warppoints_for_map,
            best_exit_warp=_best_exit_warp,
            add_candidate=add_candidate,
            add_env_tool=add_env_tool,
        )
    elif accepted_subtask == "finish_rival_battle":
        append_finish_rival_battle_candidates(
            nav=nav,
            visible_warps=visible_warps,
            max_keys_per_step=max_keys_per_step,
            add_candidate=add_candidate,
            add_env_tool=add_env_tool,
        )
    elif accepted_subtask == "anti_stuck_recover":
        append_anti_stuck_recover_candidates(
            last_action=last_action,
            max_keys_per_step=max_keys_per_step,
            add_candidate=add_candidate,
        )
    elif accepted_subtask == "explore_frontier":
        append_explore_frontier_candidates(
            sections=sections,
            add_candidate=add_candidate,
        )

    # Standard candidates (lower priority, always available)
    append_standard_field_candidates(
        sections=sections,
        map_name=map_name,
        accepted_subtask=accepted_subtask,
        is_indoor_map=is_indoor_map,
        current_party=current_party,
        pos=pos,
        nav=nav,
        note=note,
        last_action=last_action,
        max_keys_per_step=max_keys_per_step,
        add_candidate=add_candidate,
        add_env_tool=add_env_tool,
    )
