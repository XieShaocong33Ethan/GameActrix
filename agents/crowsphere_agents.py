from __future__ import annotations

from agents.crowsphere.engine import CrowSphereEngine


class PokemonAgent:
    TRACK = "TRACK1"

    def __init__(self) -> None:
        self._engine = CrowSphereEngine()

    def act(self, obs) -> str:
        return self._engine.act("pokemon_red", obs)


class TwentyFourtyEightAgent:
    TRACK = "TRACK1"

    def __init__(self) -> None:
        self._engine = CrowSphereEngine()

    def act(self, obs) -> str:
        return self._engine.act("twenty_fourty_eight", obs)


class SuperMarioAgent:
    TRACK = "TRACK1"

    def __init__(self) -> None:
        self._engine = CrowSphereEngine()

    def act(self, obs) -> str:
        return self._engine.act("super_mario", obs)


class StarCraftAgent:
    TRACK = "TRACK1"

    def __init__(self) -> None:
        self._engine = CrowSphereEngine()

    def act(self, obs) -> str:
        return self._engine.act("star_craft", obs)
