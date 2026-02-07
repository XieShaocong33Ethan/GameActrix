# 最终提交材料目录（Final submission package）

本目录用于整理官方要求的“最终提交包”所需文件与说明，便于统一打包与复现。

## 1) 需要提供给官方的内容（按规则清单对齐）

1. Model artifacts and documentation
   - 模型：`Qwen/Qwen3-VL-8B-Instruct`
   - 精确 revision：见 `submission/MODEL_MANIFEST.json`（当前 resolved 为 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`）
   - 模型格式与推理口径：见 `submission/MODEL_ARTIFACTS_AND_RUNTIME.md`
   - 需要以 gated Hugging Face 仓库形式共享，并授予访问权限：
     - `aicrowd`（AIcrowd SA）
     - `orak-krafton-eval`（Krafton evaluation）
2. Runnable agent code
   - 本 GitLab 私有仓库（需要授予 `@aicrowd` 与 `@orak-krafton-eval` 访问权限）
3. 2-page design and training PDF
   - `submission/design_and_training.pdf`
4. Reproducibility artifacts
   - `MODEL_MANIFEST.json`（模型来源与文件校验和）
   - `llm_calls.jsonl`（可重分词文本 + 关键哈希）
5. Evaluation summaries plus required metadata
   - `MODEL_DECLARATION.json`
   - `EVALUATION_SUMMARY.json/.csv`
   - `PER_EPISODE_BREAKDOWN.json/.csv`
   - `RAW_REQUESTS_README.md`

## 1.1) 本次准备好的满分评测产物（已拷贝到 submission/ 下）

为降低 GitHub 交付仓库体积，本仓库只保留 1 份“官方在线评测（REMOTE）满分”产物：

- 在线评测交付物：`submission/eval_artifacts/20260205_073454_online_309465/`
  - Submission `309465`，Session `b41c3735201a4b82b860c24c036930d6`
  - 四个游戏（每个 3 局）均满分：
    - 2048：`1.0 / 1.0 / 1.0`
    - Mario：`1.0 / 1.0 / 1.0`
    - Pokemon：`7.0 / 7.0 / 7.0`
    - StarCraft：`1.0 / 1.0 / 1.0`
  - 关键文件：
    - `EVALUATION_SUMMARY.json/.csv`、`PER_EPISODE_BREAKDOWN.json/.csv`、`MODEL_DECLARATION.json`
    - `llm_calls.jsonl`（可重分词文本，含去标识图像占位符）
    - `RAW_REQUESTS_README.md`（英文说明）
    - `evaluation.log`、`official_online_eval_309465_console.log`

更完整的本地评测产物与回归对照保存在内部研发仓库中，不在本 GitHub 交付仓库中提供。

合规说明（与本次沟通要求对齐）：
- Pokemon：`bypass_model_when_candidates` 默认与配置均为 `false`，即“有 candidates 也必须由模型选择最终动作”。
- Mario：不再在 adapter 中直接改写模型输出；当触发 `PANIC_AHEAD@<=8px`、`FAST=>OK` 等约束时，通过重试反馈要求模型重新选择。

## 2) 在服务器上复现的最短路径

前提：
- 一台有 NVIDIA GPU 的 Linux 机器
- 本机启动 vLLM 并监听 `http://127.0.0.1:8000/v1`
- vLLM 载入模型 `Qwen/Qwen3-VL-8B-Instruct`，并固定推理参数（详见 `submission/MODEL_ARTIFACTS_AND_RUNTIME.md`）

示例步骤：

1) 启动 vLLM（另开一个终端）

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

2) 在本目录所在的代码包根目录运行 starter-kit

```bash
uv sync
GAME_DATA_DIR=game_logs_local_$(date -u +%Y%m%d_%H%M%S) \
uv run python run.py --local
```

只跑单个游戏（示例，StarCraft II）：

```bash
SC2PATH=/path/to/StarCraftII \
ORAK_STARCRAFT_RGB_RENDER=0 \
GAME_DATA_DIR=game_logs_sc2_local_$(date -u +%Y%m%d_%H%M%S) \
uv run python run.py --local --games star_craft
```

备注：
- Pokémon Red 的 `executables/` 与 `processed_map/` 已包含在此代码包中，无需额外构建。
- StarCraft II 需要可用的 SC2 安装目录，并通过 `SC2PATH` 指定。

## 3) 生成 2 页 PDF（Typst）

Typst 源文件：

- `submission/design_and_training.typ`

生成 PDF（需要本机安装 `typst`）：

```bash
typst compile submission/design_and_training.typ submission/design_and_training.pdf
```
