from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agents.crowsphere.config_loader import load_effective_config
from agents.crowsphere.llm_logging import count_tokens


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []

    def _gen() -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj

    return _gen()


def _infer_provider(base_url: str) -> str:
    url = (base_url or "").lower()
    if "dashscope" in url:
        return "dashscope"
    if "modelscope" in url:
        return "modelscope"
    if "openai" in url:
        return "openai_compatible"
    if url:
        return "openai_compatible"
    return "unknown"


def _infer_parameter_count(model_name: str) -> int | None:
    name = (model_name or "").lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name)
    if not match:
        return None
    try:
        billions = float(match.group(1))
    except Exception:
        return None
    if billions <= 0:
        return None
    return int(billions * 1_000_000_000)


def _normalize_parameter_count(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        v = value.strip()
        if v.isdigit():
            n = int(v)
            return n if n > 0 else None
    return None


def write_model_declaration(
    *,
    output_dir: Path,
    config: dict[str, Any] | None = None,
    llm_calls_path: Path | None = None,
) -> Path:
    """
    生成 MODEL_DECLARATION.json。

    - 优先使用 config.model.* 里的字段；
    - 如果存在 llm_calls.jsonl，会补充「本次评测实际用到的模型列表」以及每个模型的调用次数。
    """
    cfg = config or load_effective_config()
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}

    cfg_name = str(model_cfg.get("name") or "")
    cfg_base_url = str(model_cfg.get("base_url") or "")
    cfg_provider = str(model_cfg.get("provider") or _infer_provider(cfg_base_url))
    cfg_version = str(model_cfg.get("version") or cfg_name or "unknown")
    cfg_parameter_count = _normalize_parameter_count(model_cfg.get("parameter_count")) or _infer_parameter_count(cfg_name)
    cfg_organizer_tier = model_cfg.get("organizer_tier")

    calls_by_key: dict[tuple[str, str], int] = {}
    if llm_calls_path is not None and llm_calls_path.exists():
        for rec in _iter_jsonl(llm_calls_path):
            if rec.get("event") != "llm_call":
                continue
            prov = rec.get("provider") if isinstance(rec.get("provider"), dict) else {}
            base_url = str(prov.get("base_url") or "")
            name = str(prov.get("model") or "")
            if not name and not base_url:
                continue
            key = (base_url, name)
            calls_by_key[key] = calls_by_key.get(key, 0) + 1

    models: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add_model(*, base_url: str, name: str, provider: str, version: str, parameter_count: int | None, organizer_tier: Any, calls: int | None) -> None:
        key = (base_url, name)
        if key in seen:
            return
        seen.add(key)
        item: dict[str, Any] = {
            "name": name,
            "version": version,
            "provider": provider,
            "parameter_count": parameter_count,
        }
        if organizer_tier is not None:
            item["organizer_tier"] = organizer_tier
        if base_url:
            item["base_url"] = base_url
        if calls is not None:
            item["inference_calls"] = int(calls)
        models.append(item)

    # 1) config 里的主模型
    _add_model(
        base_url=cfg_base_url,
        name=cfg_name,
        provider=cfg_provider,
        version=cfg_version,
        parameter_count=cfg_parameter_count,
        organizer_tier=cfg_organizer_tier,
        calls=None,
    )

    # 2) llm_calls.jsonl 里出现过的模型（可能是 fallback / 多模型）
    for (base_url, name), calls in sorted(calls_by_key.items(), key=lambda x: (-x[1], x[0][1], x[0][0])):
        provider = _infer_provider(base_url)
        version = name or "unknown"
        parameter_count = _infer_parameter_count(name)
        organizer_tier = None
        if name == cfg_name and base_url == cfg_base_url:
            provider = cfg_provider
            version = cfg_version
            parameter_count = cfg_parameter_count
            organizer_tier = cfg_organizer_tier
        _add_model(
            base_url=base_url,
            name=name,
            provider=provider,
            version=version,
            parameter_count=parameter_count,
            organizer_tier=organizer_tier,
            calls=calls,
        )

    payload: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "models": models,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "MODEL_DECLARATION.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_requests_readme(*, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "RAW_REQUESTS_README.md"
    text = (
        "# Raw requests / re-tokenizable text\n\n"
        "- `llm_calls.jsonl`：每次模型调用一行（JSONL）。\n"
        "- `request.retoken_text`：把所有 message 的纯文本内容拼接后的字符串；官方可用指定 tokenizer 重分词。\n"
        "- 图片不参与重分词统计：`request.messages` 里 `image_url.url` 已脱敏为占位符；对应图片一致性用 `context.image_sha256` 校验。\n"
        "- `tokenizer.prompt_tokens`：如果本地安装了 tiktoken，会用 `cl100k_base` 计算提示词 token；否则为 null，最终以官方复算为准。\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


@dataclass(frozen=True)
class EpisodeKey:
    game: str
    episode_id: int


def _episode_maps_from_game_states(game_states_path: Path) -> tuple[dict[int, int], dict[int, Any]]:
    """Return (step_id->episode_id, episode_id->final_score). step_id is 1-based line index."""
    step_to_episode: dict[int, int] = {}
    final_scores: dict[int, Any] = {}
    if not game_states_path.exists():
        return step_to_episode, final_scores

    episode_id = 1
    step_id = 0
    with game_states_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            step_id += 1
            step_to_episode[step_id] = episode_id
            result = rec.get("result")
            is_finished = bool(result.get("is_finished")) if isinstance(result, dict) else False
            if is_finished:
                score = rec.get("current_score")
                if score is None and isinstance(result, dict):
                    score = result.get("score")
                final_scores[episode_id] = score
                episode_id += 1
    return step_to_episode, final_scores


def summarize_evaluation(
    *,
    game_data_dir: Path,
    llm_calls_path: Path,
    tokenizer_name: str = "cl100k_base",
    games: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    games_list = games or [
        "twenty_fourty_eight",
        "super_mario",
        "pokemon_red",
        "star_craft",
    ]

    step_to_episode_by_game: dict[str, dict[int, int]] = {}
    final_scores_by_game: dict[str, dict[int, Any]] = {}
    episodes_finished_by_game: dict[str, int] = {}

    for game in games_list:
        path = game_data_dir / game / "game_states.jsonl"
        step_map, final_scores = _episode_maps_from_game_states(path)
        step_to_episode_by_game[game] = step_map
        final_scores_by_game[game] = final_scores
        episodes_finished_by_game[game] = len(final_scores)

    def _to_float(value: Any) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            try:
                return float(v)
            except Exception:
                return None
        return None

    per_episode_stats: dict[EpisodeKey, dict[str, Any]] = {}
    total_calls = 0
    total_tokens: int | None = 0
    token_count_available = True

    for rec in _iter_jsonl(llm_calls_path):
        if rec.get("event") != "llm_call":
            continue
        ctx = rec.get("context") if isinstance(rec.get("context"), dict) else {}
        game = str(ctx.get("game") or "").strip()
        step_id = ctx.get("step_id")
        if not game:
            continue
        total_calls += 1

        prompt_tokens: int | None = None
        tok = rec.get("tokenizer") if isinstance(rec.get("tokenizer"), dict) else {}
        if isinstance(tok.get("prompt_tokens"), int):
            prompt_tokens = int(tok.get("prompt_tokens"))
        if prompt_tokens is None:
            req = rec.get("request") if isinstance(rec.get("request"), dict) else {}
            text = req.get("retoken_text")
            if isinstance(text, str):
                prompt_tokens = count_tokens(text=text, encoding_name=tokenizer_name)
        if prompt_tokens is None:
            token_count_available = False
        if total_tokens is not None:
            total_tokens += int(prompt_tokens or 0)

        episode_id = None
        if isinstance(step_id, int):
            episode_id = step_to_episode_by_game.get(game, {}).get(step_id)
        if not isinstance(episode_id, int) or episode_id <= 0:
            continue

        key = EpisodeKey(game=game, episode_id=episode_id)
        stats = per_episode_stats.get(key)
        if stats is None:
            stats = {"inference_calls": 0, "tokens": 0}
            per_episode_stats[key] = stats
        stats["inference_calls"] += 1
        stats["tokens"] += int(prompt_tokens or 0)

    per_episode_breakdown: list[dict[str, Any]] = []
    evaluation_episodes = 0
    for game in games_list:
        for episode_id in range(1, episodes_finished_by_game.get(game, 0) + 1):
            key = EpisodeKey(game=game, episode_id=episode_id)
            stats = per_episode_stats.get(key, {"inference_calls": 0, "tokens": 0})
            per_episode_breakdown.append(
                {
                    "episode_id": f"{game}-{episode_id}",
                    "game_name": game,
                    "seed": None,
                    "inference_calls": int(stats.get("inference_calls", 0)),
                    "tokens": int(stats.get("tokens", 0)),
                    "final_score": final_scores_by_game.get(game, {}).get(episode_id),
                }
            )
            evaluation_episodes += 1

    mean_calls = (total_calls / evaluation_episodes) if evaluation_episodes else 0.0
    mean_tokens: float | None
    if total_tokens is None:
        mean_tokens = None
    else:
        mean_tokens = (total_tokens / evaluation_episodes) if evaluation_episodes else 0.0

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "evaluation_episodes": evaluation_episodes,
        "total_inference_calls": total_calls,
        "total_tokens": total_tokens if token_count_available else None,
        "mean_calls_per_episode": mean_calls,
        "mean_tokens_per_episode": mean_tokens if token_count_available else None,
        "tokenizer": {
            "name": tokenizer_name,
            "note": "tokens are counted from request.retoken_text; may be null if tiktoken is unavailable",
            "available": bool(token_count_available),
        },
        "scores": {},
    }

    for game in games_list:
        raw_scores = list(final_scores_by_game.get(game, {}).values())
        numeric_scores = [s for s in (_to_float(v) for v in raw_scores) if s is not None]
        summary["scores"][game] = {
            "episodes_finished": int(episodes_finished_by_game.get(game, 0)),
            "avg_final_score": (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else None,
            "min_final_score": min(numeric_scores) if numeric_scores else None,
            "max_final_score": max(numeric_scores) if numeric_scores else None,
        }
    return summary, per_episode_breakdown


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_evaluation_artifacts(
    *,
    output_dir: Path,
    game_data_dir: Path,
    games: list[str] | None = None,
    llm_calls_path: Path | None = None,
) -> dict[str, Path]:
    """
    生成交付物：
    - MODEL_DECLARATION.json
    - EVALUATION_SUMMARY.json (+ CSV)
    - PER_EPISODE_BREAKDOWN.json (+ CSV)
    - RAW_REQUESTS_README.md
    - llm_calls.jsonl（若不存在则创建空文件，方便后续填充/打包）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    llm_path = llm_calls_path or (output_dir / "llm_calls.jsonl")
    llm_path.touch(exist_ok=True)

    model_path = write_model_declaration(output_dir=output_dir, llm_calls_path=llm_path)
    readme_path = write_requests_readme(output_dir=output_dir)

    summary, per_episode = summarize_evaluation(
        game_data_dir=game_data_dir,
        llm_calls_path=llm_path,
        games=games,
    )

    summary_path = output_dir / "EVALUATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_csv_path = output_dir / "EVALUATION_SUMMARY.csv"
    _write_csv(summary_csv_path, [summary])

    per_episode_path = output_dir / "PER_EPISODE_BREAKDOWN.json"
    per_episode_path.write_text(json.dumps(per_episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_episode_csv_path = output_dir / "PER_EPISODE_BREAKDOWN.csv"
    _write_csv(per_episode_csv_path, per_episode)

    return {
        "MODEL_DECLARATION.json": model_path,
        "RAW_REQUESTS_README.md": readme_path,
        "llm_calls.jsonl": llm_path,
        "EVALUATION_SUMMARY.json": summary_path,
        "EVALUATION_SUMMARY.csv": summary_csv_path,
        "PER_EPISODE_BREAKDOWN.json": per_episode_path,
        "PER_EPISODE_BREAKDOWN.csv": per_episode_csv_path,
    }
