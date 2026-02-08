# Final Submission Package

This folder contains the files required by the Orak Game Agent Challenge 2025 submission checklist, plus short
reproducibility notes for reviewers.

## 1) What we provide (mapped to official requirements)

1. Model artifacts and documentation
   - Base model id + exact resolved revision: `submission/MODEL_MANIFEST.json`
   - Runtime / serving / decoding details: `submission/MODEL_ARTIFACTS_AND_RUNTIME.md`
   - If the base model repository is gated, grant access to:
     - `aicrowd` (AIcrowd SA)
     - `orak-krafton-eval` (Krafton evaluation)
2. Runnable agent code
   - Starter-kit runner: `run.py`, `evaluation_utils/`
   - CrowSphere agent: `agents/crowsphere/`, `agents/crowsphere_agents.py`, `agents/config.py`
   - Default evaluation config: `agents/crowsphere_config.json`
3. 2-page design and training PDF
   - PDF: `submission/design_and_training.pdf`
   - Source: `submission/design_and_training.typ`
4. Reproducibility artifacts
   - Retokenizable model-call logs: `submission/eval_artifacts/*/llm_calls.jsonl`
5. Evaluation summaries plus required metadata
   - `EVALUATION_SUMMARY.json/.csv`
   - `PER_EPISODE_BREAKDOWN.json/.csv`
   - `MODEL_DECLARATION.json`
   - `RAW_REQUESTS_README.md`

## 2) Included official REMOTE evaluation artifacts (full score)

To keep the GitHub delivery repository small, we include a single official REMOTE full-score run:

- `submission/eval_artifacts/20260208_163116_online_309561/`
  - Submission `309561`, Session `301f00dff97440d69d21f7ce88dc24e6`
  - Full score across 12 episodes:
    - 2048: 1.0 (model-driven; non-zero inference calls)
    - Super Mario: 1.0
    - Pokemon Red: 7.0
    - StarCraft II: 1.0

## 3) Compliance defaults (avoid misconfiguration)

The evaluation entrypoint loads `agents/crowsphere_config.json` by default (unless `CROWSPHERE_CONFIG` is set).

Key defaults:

- 2048: `twenty_fourty_eight.use_model=true` (the engine raises an error if disabled)
- Pokemon Red: `pokemon.tool_use.bypass_model_when_candidates=false`
- Super Mario: no deterministic action override; invalid outputs trigger retries (model re-chooses)

## 4) Minimal reproduction steps (Linux + NVIDIA GPU)

1) Start vLLM:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0
```

2) Run the starter-kit:

```bash
uv sync
GAME_DATA_DIR=game_logs_local_$(date -u +%Y%m%d_%H%M%S) \
uv run python run.py --local
```

## 5) Build the 2-page PDF (Typst)

```bash
typst compile submission/design_and_training.typ submission/design_and_training.pdf
```
