from __future__ import annotations

from agents.crowsphere.mario_preprocess.candidates import build_candidates
from agents.crowsphere.mario_preprocess.koopa_memory import KoopaMemory
from agents.crowsphere.mario_preprocess.mario_memory import MarioPreprocessMemory
from agents.crowsphere.mario_preprocess.render import render_preprocessed_obs_text
from agents.crowsphere.mario_preprocess.risk_report import build_risk_entries
from agents.crowsphere.mario_preprocess.state import extract_mario_state


def preprocess_super_mario_obs_text(
    obs_text: str,
    *,
    koopa_memory: KoopaMemory | MarioPreprocessMemory | None = None,
    mode: str = "full",
) -> str:
    mode_norm = str(mode or "full").strip().lower()
    if mode_norm not in {"full", "evidence_only", "raw"}:
        mode_norm = "full"
    if mode_norm == "raw":
        return (obs_text or "").strip()

    koopa_mem: KoopaMemory | None = None
    ceiling_mem = None
    if isinstance(koopa_memory, MarioPreprocessMemory):
        koopa_mem = koopa_memory.koopa_memory
        ceiling_mem = koopa_memory.low_ceiling_memory
    else:
        koopa_mem = koopa_memory

    state = extract_mario_state(obs_text or "", koopa_memory=koopa_mem, ceiling_memory=ceiling_mem)
    candidates = build_candidates(state)
    risk_entries = build_risk_entries(state, candidates)
    return render_preprocessed_obs_text(state=state, candidates=candidates, risk_entries=risk_entries, mode=mode_norm)
