# Final Submission Package Checklist (Track 2)

This checklist maps the official requirements to concrete files in this repository.

## 1. Model artifacts and documentation

- Base model id and resolved revision
  - `submission/MODEL_MANIFEST.json`
- Model format, numerical type, serving, decoding parameters, prompt template locations
  - `submission/MODEL_ARTIFACTS_AND_RUNTIME.md`

## 2. Runnable agent code

- Orak starter-kit runner
  - `run.py`
  - `evaluation_utils/`
- CrowSphere agent implementation
  - `agents/crowsphere/`
  - `agents/crowsphere_agents.py`
  - `agents/config.py`
- Default agent configuration
  - `agents/crowsphere_config.json`

## 3. 2 page design and training PDF

- PDF
  - `submission/design_and_training.pdf`
- Source
  - `submission/design_and_training.typ`
- Build command (Typst)
  - `typst compile submission/design_and_training.typ submission/design_and_training.pdf`

## 4. Reproducibility artifacts

- Retokenizable model call logs
  - `submission/eval_artifacts/20260207_222041_online_309532/llm_calls.jsonl`

## 5. Evaluation summaries plus required metadata

- Official REMOTE full score summary and per episode breakdown
  - `submission/eval_artifacts/20260207_222041_online_309532/EVALUATION_SUMMARY.json`
  - `submission/eval_artifacts/20260207_222041_online_309532/PER_EPISODE_BREAKDOWN.json`
  - `submission/eval_artifacts/20260207_222041_online_309532/MODEL_DECLARATION.json`
  - `submission/eval_artifacts/20260207_222041_online_309532/RAW_REQUESTS_README.md`

## 6. Extra data needed at runtime

- Pokemon PyBoy artifacts (ROM, RAM, symbol and map files)
  - `executables/pokemon_red/pyboy/`
- Pokemon deterministic navigation data
  - `evaluation_utils/mcp_game_servers/pokemon_red/game/processed_map/`
