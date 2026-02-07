import json
import os
import threading
import time
from typing import Any, Callable


_TRUTHY = {"1", "true", "yes", "y", "on"}


def _env_is_truthy(name: str) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    return raw in _TRUTHY


def live_export_enabled() -> bool:
    return (
        _env_is_truthy("ORAK_LIVE_EXPORT")
        or _env_is_truthy("ORAK_LIVE_EXPORT_ENABLED")
        or _env_is_truthy("LIVE_EXPORT")
    )


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _stringify_mapping(data: dict[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {}
    return {str(k): _json_safe(v) for k, v in data.items()}


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)
    os.replace(tmp_path, path)


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _relative_path(base_dir: str, path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, base_dir)
    except Exception:
        return path


class PerGameLiveExporter:
    def __init__(
        self,
        base_dir: str,
        game_id: str,
        port: int,
        get_game_config: Callable[[], dict[str, Any]],
    ) -> None:
        self._base_dir = base_dir
        self._game_id = game_id
        self._port = port
        self._get_game_config = get_game_config
        self._lock = threading.Lock()

        self._game_dir = os.path.join(base_dir, game_id)
        self._frames_dir = os.path.join(self._game_dir, "frames")
        os.makedirs(self._frames_dir, exist_ok=True)

        self._state_path = os.path.join(self._game_dir, "live_state.json")
        self._frame_index = 0
        self._episode_index = 0
        self._failure_count = 0

        self._state: dict[str, Any] = {
            "game_id": game_id,
            "port": port,
            "game_config": self._safe_game_config(),
            "episode": self._episode_index,
            "updated_at": time.time(),
        }
        self._write_state()

    def _safe_game_config(self) -> dict[str, Any]:
        try:
            config = self._get_game_config() or {}
        except Exception:
            config = {}
        return _stringify_mapping(config)

    def _write_state(self) -> None:
        _atomic_write_json(self._state_path, self._state)

    def on_session_start(self) -> None:
        with self._lock:
            self._frame_index = 0
            self._episode_index = 0
            self._failure_count = 0
            self._state.update(
                {
                    "game_config": self._safe_game_config(),
                    "episode": self._episode_index,
                    "updated_at": time.time(),
                    "session_started_at": time.time(),
                    "last_observation": None,
                }
            )
            self._write_state()

    def record_observation(
        self,
        obs_text: str,
        obs_image_bytes: bytes | None,
        game_info: dict[str, Any],
        current_step_in_episode: int,
    ) -> None:
        with self._lock:
            image_path = None
            if obs_image_bytes:
                filename = f"frame_{self._frame_index:06d}.jpg"
                image_path = os.path.join(self._frames_dir, filename)
                try:
                    with open(image_path, "wb") as f:
                        f.write(obs_image_bytes)
                    self._frame_index += 1
                except Exception:
                    image_path = None

            self._state.update(
                {
                    "current_step_in_episode": int(current_step_in_episode),
                    "last_observation": {
                        "text": obs_text,
                        "info": _stringify_mapping(game_info),
                        "image_path": _relative_path(self._base_dir, image_path),
                        "frame_index": self._frame_index - 1 if image_path else None,
                    },
                    "updated_at": time.time(),
                }
            )
            self._write_state()

    def record_step(
        self,
        score: float,
        avg_score: float,
        is_finished: bool,
        max_episodes_reached: bool,
        current_step_in_episode: int,
    ) -> None:
        with self._lock:
            if is_finished:
                self._episode_index += 1

            self._state.update(
                {
                    "score": score,
                    "avg_score": avg_score,
                    "is_finished": is_finished,
                    "max_episodes_reached": max_episodes_reached,
                    "current_step_in_episode": int(current_step_in_episode),
                    "episode": self._episode_index,
                    "updated_at": time.time(),
                }
            )
            self._write_state()

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._state.update(
                {
                    "failure_count": self._failure_count,
                    "last_failure_at": time.time(),
                    "updated_at": time.time(),
                }
            )
            self._write_state()


class EvaluationStateAggregator:
    def __init__(
        self,
        renderer: Any,
        games: list[str],
        game_servers_procs: dict[str, Any],
        proc_start_times: dict[str, float],
        proc_end_times: dict[str, float],
        game_data_dir: str,
        aggregate_path: str,
        interval_seconds: float = 1.0,
    ) -> None:
        self._renderer = renderer
        self._games = games
        self._game_servers_procs = game_servers_procs
        self._proc_start_times = proc_start_times
        self._proc_end_times = proc_end_times
        self._game_data_dir = game_data_dir
        self._aggregate_path = aggregate_path
        self._interval_seconds = interval_seconds

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._write_state_snapshot(final=True)

    def finalize_game_media(self, game_name: str) -> None:
        live_path = os.path.join(self._game_data_dir, game_name, "live_state.json")
        state = _read_json(live_path) or {}
        state["media_finalized"] = True
        state["media_finalized_at"] = time.time()
        _atomic_write_json(live_path, state)

    def finalize_all_media(self) -> None:
        self._write_state_snapshot(final=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._write_state_snapshot()
            self._stop_event.wait(self._interval_seconds)

    def _collect_game_state(self, game_name: str) -> dict[str, Any]:
        renderer_state = getattr(self._renderer, "state", None)
        status = None
        score = None
        if renderer_state is not None:
            status = renderer_state.server_status_by_game.get(game_name)
            score = renderer_state.scores_by_game.get(game_name)

        proc = self._game_servers_procs.get(game_name)
        return_code = None
        pid = None
        if proc is not None:
            pid = getattr(proc, "pid", None)
            return_code = proc.poll()
            if return_code is not None and game_name not in self._proc_end_times:
                self._proc_end_times[game_name] = time.time()

        live_path = os.path.join(self._game_data_dir, game_name, "live_state.json")
        live_state = _read_json(live_path)

        return {
            "status": status,
            "score": score,
            "start_time": self._proc_start_times.get(game_name),
            "end_time": self._proc_end_times.get(game_name),
            "pid": pid,
            "return_code": return_code,
            "live_state": live_state,
        }

    def _write_state_snapshot(self, final: bool = False) -> None:
        payload = {
            "updated_at": time.time(),
            "final": final,
            "games": {game: self._collect_game_state(game) for game in self._games},
        }
        _atomic_write_json(self._aggregate_path, payload)
