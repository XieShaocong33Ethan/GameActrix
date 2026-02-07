from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(Exception):
    pass


class MissingConfigError(ConfigError):
    pass


class InvalidConfigError(ConfigError):
    pass


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    config: dict[str, Any]


def _resolve_eval_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path(eval_root: Path) -> Path:
    return eval_root / "agents" / "crowsphere_config.json"


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MissingConfigError(f"配置文件不存在: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"读取配置文件失败: {path}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidConfigError(f"配置文件不是合法 JSON: {path}") from exc

    if not isinstance(data, dict):
        raise InvalidConfigError(f"配置文件根节点必须是 JSON 对象: {path}")
    return data


def load_effective_config(
    *,
    eval_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    env_var_name: str = "CROWSPHERE_CONFIG",
) -> dict[str, Any]:
    environment = env if env is not None else os.environ
    root = eval_root if eval_root is not None else _resolve_eval_root()

    from_env = (environment.get(env_var_name) or "").strip()
    if from_env:
        path = Path(from_env).expanduser()
        return _load_json_file(path)

    default_path = _default_config_path(root)
    if default_path.exists():
        return _load_json_file(default_path)

    raise MissingConfigError(
        f"未找到配置文件：环境变量 {env_var_name} 未设置，且默认路径不存在: {default_path}"
    )

