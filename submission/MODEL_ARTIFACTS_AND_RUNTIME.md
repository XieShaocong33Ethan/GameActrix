# Model Artifacts and Runtime Details

This file is part of the final submission package. It describes the exact base model reference and the runtime details
needed to reproduce our results without uploading base model weights.

## 1. Base Model Reference

We do not fine tune the base model in the final agent.

Model repository id

`Qwen/Qwen3-VL-8B-Instruct`

Resolved model revision

The exact revision is recorded in `submission/MODEL_MANIFEST.json`.

Current resolved commit hash

`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`

Model format and numerical type

We use Hugging Face safetensors weights and serve the model with vLLM using bfloat16. We do not apply weight quantization.

## 2. Inference Runtime and Key Versions

We run a local vLLM OpenAI compatible server on the evaluation machine and the agent talks to it via HTTP.

Key versions observed on the evaluation server

- Python 3.10.8
- vLLM 0.12.0
- torch 2.9.0
- transformers 4.57.1
- openai 1.100.2
- tiktoken 0.9.0
- pillow 11.2.0

For an automatically captured snapshot of Python executable, platform, environment variables, and key dependencies,
see the evaluation logs in the official run directory, for example

`submission/eval_artifacts/20260207_222041_online_309532/evaluation.log`

## 3. Decoding Parameters and Image Preprocessing

All decoding parameters are controlled by `agents/crowsphere_config.json` and applied by
`agents/crowsphere/engine_generation.py`.

Defaults

- temperature 0
- top_p 1
- presence_penalty 0
- frequency_penalty 0
- max_tokens 512
- max_tokens for Super Mario 256

Image preprocessing is deterministic and controlled by `agents/crowsphere_config.json` and
`agents/crowsphere/image_preproc.py`.

Defaults

- short side 184
- JPEG quality 85

## 4. Prompts and Structured Output Templates

The exact prompt templates are part of the runnable agent code.

Main prompt builders

- `agents/crowsphere/prompt_builder.py`
- `agents/crowsphere/prompt_builder_sc2.py`

Structured JSON output schema and generation parameter wiring

- `agents/crowsphere/engine_generation.py`
- `agents/crowsphere/action_plan.py`

Game adapters that validate and convert ActionPlan into environment actions

- `agents/crowsphere/adapters/super_mario.py`
- `agents/crowsphere/adapters/pokemon_red.py`
- `agents/crowsphere/adapters/star_craft.py`

## 5. Configuration Files

The agent reads configuration using the following priority

1. `CROWSPHERE_CONFIG` environment variable pointing to a JSON file
2. default `agents/crowsphere_config.json`

The final submission uses

`agents/crowsphere_config.json`

## 6. Extra Data and Scripts Needed for Reproduction

Pokemon Red

The PyBoy runtime expects artifacts under `executables/pokemon_red/pyboy/`. This submission package includes
`pokered.gbc` and the companion `.ram/.map/.sym` files in that directory.

Pokemon deterministic navigation uses processed map data

`evaluation_utils/mcp_game_servers/pokemon_red/game/processed_map/`

StarCraft II

The StarCraft II installation path is provided via `SC2PATH` and is expected to exist on the evaluation machine.

## 7. Reproducibility Scripts

To reproduce runs, start a local vLLM server and then run the starter-kit with uv.

Example vLLM command (adjust GPU memory utilization and max model length for your machine)

`vllm serve Qwen/Qwen3-VL-8B-Instruct --trust-remote-code --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 --limit-mm-per-prompt.image 1 --limit-mm-per-prompt.video 0 --host 0.0.0.0 --port 8000 --served-model-name Qwen/Qwen3-VL-8B-Instruct`

Starter-kit runner (local mode)

`uv sync && GAME_DATA_DIR=game_logs_local_$(date -u +%Y%m%d_%H%M%S) uv run python run.py --local`

## 8. Evaluation Summaries

Official REMOTE full score reference

`submission/eval_artifacts/20260207_222041_online_309532/EVALUATION_SUMMARY.json`
