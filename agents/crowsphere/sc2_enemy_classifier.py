from __future__ import annotations

import re
from typing import Any, Mapping


def _norm_unit_name(name: str) -> str:
    """Normalize unit names like 'siege_tank'/'Siege Tank' -> 'siegetank'."""
    return re.sub(r"[^a-z0-9]+", "", str(name or "").strip().lower())


# Notes:
# - The env prints `Enemy unittypeid.<lowercase_name>: <count>`; historically these names match
#   python-sc2 `UnitTypeId` names lowercased (e.g. `hydralisk`, `hightemplar`).
# - We keep this list conservative: only units that materially impact our SC2 macro decisions.
# - Unknown units are treated as ground combat for safety (prefer false-positive defense over blind attack).

_WORKERS = {
    "scv",
    "probe",
    "drone",
    "mule",
}

_NONCOMBAT_GROUND = {
    "larva",
    "egg",
    "changeling",
    "changelingzealot",
}

_NONCOMBAT_AIR = {
    # Zerg
    "overlord",
    "overseer",
    # Protoss
    "observer",
    "warpprism",
    # Terran
    "medivac",
}

_BUILDINGS = {
    # Zerg
    "hatchery",
    "lair",
    "hive",
    "extractor",
    "spawningpool",
    "roachwarren",
    "hydraliskden",
    "spire",
    "greaterspire",
    "spire",
    "evolutionchamber",
    "sporecrawler",
    "spinecrawler",
    # Terran
    "commandcenter",
    "orbitalcommand",
    "planetaryfortress",
    "refinery",
    "supplydepot",
    "supplydepotlowered",
    "barracks",
    "factory",
    "starport",
    "engineeringbay",
    "bunker",
    "missileturret",
    # Protoss
    "nexus",
    "pylon",
    "gateway",
    "warpgate",
    "cyberneticscore",
    "forge",
    "photoncannon",
    "shieldbattery",
    "roboticsfacility",
    "stargate",
    "twilightcouncil",
    "templararchive",
    "darkshrine",
    "roboticsbay",
    "fleetbeacon",
    "assimilator",
}

_GROUND_MELEE_THREATS = {
    # Zerg
    "zergling",
    "baneling",
    "ultralisk",
    "queen",
    # Protoss
    "zealot",
    "darktemplar",
}

_GROUND_RANGED_THREATS = {
    # Zerg
    "roach",
    "hydralisk",
    "ravager",
    "lurker",
    "lurkerdenmp",  # sometimes appears in python-sc2 unit ids
    # Terran
    "marine",
    "marauder",
    "reaper",
    "hellion",
    "hellbat",
    "widowmine",
    "widowmineburrowed",
    "siegetank",
    "siegetanksieged",
    "cyclone",
    "thor",
    "ghost",
    # Protoss
    "stalker",
    "adept",
    "sentry",
    "immortal",
    "colossus",
    "disruptor",
    "archon",
    "hightemplar",
}

_AIR_THREATS = {
    # Zerg
    "mutalisk",
    "corruptor",
    "broodlord",
    # Terran
    "vikingfighter",
    "banshee",
    "liberator",
    "liberatorag",
    "battlecruiser",
    # Protoss
    "phoenix",
    "voidray",
    "oracle",
    "carrier",
    "tempest",
    "mothership",
}


def classify_enemy_unit(unit_name: str) -> str:
    """Return category for threat aggregation.

    Categories:
    - ground_ranged
    - ground_melee
    - air
    - noncombat
    - building
    - unknown
    """
    n = _norm_unit_name(unit_name)
    if not n:
        return "noncombat"
    if n in _WORKERS or n in _NONCOMBAT_GROUND or n in _NONCOMBAT_AIR:
        return "noncombat"
    if n in _BUILDINGS:
        return "building"
    if n in _AIR_THREATS:
        return "air"
    if n in _GROUND_RANGED_THREATS:
        return "ground_ranged"
    if n in _GROUND_MELEE_THREATS:
        return "ground_melee"
    return "unknown"


def aggregate_enemy_units(enemy_units: Mapping[str, Any]) -> dict[str, int]:
    """Aggregate per-unit counts into a few threat buckets used by the macro policy."""
    ground_ranged = 0
    ground_melee = 0
    air = 0
    unknown = 0
    for raw_name, raw_count in (enemy_units or {}).items():
        try:
            count = int(raw_count or 0)
        except Exception:
            continue
        if count <= 0:
            continue
        cat = classify_enemy_unit(str(raw_name))
        if cat == "ground_ranged":
            ground_ranged += count
        elif cat == "ground_melee":
            ground_melee += count
        elif cat == "air":
            air += count
        elif cat in {"noncombat", "building"}:
            continue
        else:
            # Unknown units are treated as ground combat for safety.
            unknown += count

    ground_total = ground_ranged + ground_melee + unknown
    total_combat = ground_total + air
    return {
        "ground_ranged": ground_ranged,
        "ground_melee": ground_melee,
        "ground_total": ground_total,
        "air": air,
        "unknown": unknown,
        "total_combat": total_combat,
    }


def earliest_seen_time_seconds(enemy_first_seen_time_seconds: Mapping[str, Any], *, category: str) -> int | None:
    """Return earliest seen time (seconds) for a given unit category, or None if absent."""
    best: int | None = None
    for raw_name, raw_time in (enemy_first_seen_time_seconds or {}).items():
        try:
            t = int(raw_time or 0)
        except Exception:
            continue
        if t <= 0:
            continue
        if classify_enemy_unit(str(raw_name)) != category:
            continue
        best = t if best is None else min(best, t)
    return best
