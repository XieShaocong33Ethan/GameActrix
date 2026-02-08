# Final submission package (Track 2)

## 1) Delivery Checklist

1. Model artifacts and documentation
   - Model: `Qwen/Qwen3-VL-8B-Instruct`
   - Exact revision: see `submission/MODEL_MANIFEST.json` (currently resolved as `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`)
   - Model format and inference specifications: see `submission/MODEL_ARTIFACTS_AND_RUNTIME.md`
   - Needs to be shared as a gated Hugging Face repository with access granted to:
     - `aicrowd` (AIcrowd SA)
     - `orak-krafton-eval` (Krafton evaluation)
2. Runnable agent code
   - This GitHub repository (grant access to `@aicrowd` and `@orak-krafton-eval` if private)
3. 2-page design and training PDF
   - `submission/design_and_training.pdf`
4. Reproducibility artifacts
   - `MODEL_MANIFEST.json` (Model source and file checksums)
   - `llm_calls.jsonl` (Re-tokenizable text + key hashes)
5. Evaluation summaries plus required metadata
   - `MODEL_DECLARATION.json`
   - `EVALUATION_SUMMARY.json/.csv`
   - `PER_EPISODE_BREAKDOWN.json/.csv`
   - `RAW_REQUESTS_README.md`

## 1.1) Prepared perfect-score evaluation artifacts (Official online evaluation, REMOTE)

This repository keeps one copy of the official Track 2 REMOTE full score artifacts:

- Online evaluation deliverables: `submission/eval_artifacts/20260207_222041_online_309532/`
  - Submission `309532`, Session `42e087c10c0a4323aa966a0b8da771cf`
  - Perfect scores for all four games (3 rounds each):
    - 2048: `1.0 / 1.0 / 1.0`
    - Mario: `1.0 / 1.0 / 1.0`
    - Pokemon: `7.0 / 7.0 / 7.0`
    - StarCraft: `1.0 / 1.0 / 1.0`
  - Key files:
    - `EVALUATION_SUMMARY.json/.csv`, `PER_EPISODE_BREAKDOWN.json/.csv`, `MODEL_DECLARATION.json`
    - `llm_calls.jsonl` (Re-tokenizable text, including de-identified image placeholders)
    - `RAW_REQUESTS_README.md`
    - `evaluation.log`, `official_online_eval_309532_console.log`

Track selection note:
- The REMOTE API session track is determined by `agents/crowsphere_agents.py` (class attribute `TwentyFourtyEightAgent.TRACK`),
  which is set to `TRACK2` on this branch.

## 2) Shortest path for reproduction on a server

Prerequisites:
- A Linux machine with NVIDIA GPU
- Start vLLM locally and listen on `http://127.0.0.1:8000/v1`
- vLLM loads the model `Qwen/Qwen3-VL-8B-Instruct` and fixes inference parameters (see `submission/MODEL_ARTIFACTS_AND_RUNTIME.md` for details)

Example steps:

1) Start vLLM (in a separate terminal)

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

2) Run the starter-kit in the root directory of the code package where this directory is located

```bash
uv sync
GAME_DATA_DIR=game_logs_local_$(date -u +%Y%m%d_%H%M%S) \
uv run python run.py --local
```

Run only a single game (Example, StarCraft II):

```bash
SC2PATH=/path/to/StarCraftII \
ORAK_STARCRAFT_RGB_RENDER=0 \
GAME_DATA_DIR=game_logs_sc2_local_$(date -u +%Y%m%d_%H%M%S) \
uv run python run.py --local --games star_craft
```

Notes:
- Pokémon Red's `executables/` and `processed_map/` are already included in this code package and do not require additional building.
- StarCraft II requires a valid SC2 installation directory, specified via `SC2PATH`.
