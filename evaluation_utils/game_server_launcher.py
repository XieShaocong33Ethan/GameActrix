from multiprocessing import Process
import os
import subprocess
import time
import shutil
import json

from evaluation_utils.commons import GAME_SERVER_PORTS, GAME_DATA_DIR
from evaluation_utils.renderer import Renderer
from evaluation_utils.live_export import live_export_enabled

LIVE_EXPORT_ENABLED = live_export_enabled()
if LIVE_EXPORT_ENABLED:
    from evaluation_utils.live_export import EvaluationStateAggregator
else:
    EvaluationStateAggregator = None


class GameLauncher:
    def __init__(self, renderer: Renderer, games: list[str] | None = None):
        self.renderer = renderer
        # If no specific games are provided, default to all known games
        self.games = games or list(GAME_SERVER_PORTS.keys())
        self.game_servers_procs = {}
        self.output_files = {}
        self.proc_start_times: dict[str, float] = {}
        self.proc_end_times: dict[str, float] = {}

        self._live_export_enabled = LIVE_EXPORT_ENABLED
        self._live_export_aggregator: EvaluationStateAggregator | None = None
        if self._live_export_enabled and EvaluationStateAggregator is not None:
            self._live_export_aggregator = EvaluationStateAggregator(
                renderer=self.renderer,
                games=self.games,
                game_servers_procs=self.game_servers_procs,
                proc_start_times=self.proc_start_times,
                proc_end_times=self.proc_end_times,
                game_data_dir=GAME_DATA_DIR,
                aggregate_path=os.path.join(GAME_DATA_DIR, "evaluation_state.json"),
            )

        # Initialize all game servers as queued in the renderer
        for game in self.games:
            self.renderer.set_server_status(game, "queued")
            self.renderer.set_score(game, 0)

    def __del__(self):
        self.force_stop_all_games()
    
    def clean_game_data_dir(self):
        if os.path.exists(GAME_DATA_DIR):
            shutil.rmtree(GAME_DATA_DIR)
        os.makedirs(GAME_DATA_DIR)

    def _update_scores_from_disk(self):
        """Update renderer with scores read from disk."""
        for game in self.games:
            results_path = os.path.join(GAME_DATA_DIR, game, "game_results.json")
            score_val = 0
            try:
                if os.path.exists(results_path):
                    with open(results_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        score_val = int(data.get("score", 0))
            except Exception:
                score_val = 0
            self.renderer.set_score(game, score_val)
    
    def launch_game_server(self, game_name: str):
        if game_name in self.game_servers_procs:
            return self.game_servers_procs[game_name]

        self.renderer.set_server_status(game_name, "launching")

        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        game_server_dir = os.path.join(app_dir, "evaluation_utils", "mcp_game_servers", game_name)
        game_server_script = os.path.join(game_server_dir, "server.py")
        game_data_dir = os.path.join(GAME_DATA_DIR, game_name)
        if not os.path.exists(game_data_dir):
            os.makedirs(game_data_dir)

        # Clean frames directory when live export is enabled
        if self._live_export_enabled:
            frames_dir = os.path.join(game_data_dir, "frames")
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir, ignore_errors=True)

        cmd = [
            "python",
            game_server_script,
        ]
        env = os.environ.copy()
        env["PORT"] = str(GAME_SERVER_PORTS[game_name])
        env["GAME_DATA_DIR"] = game_data_dir
        env["PYTHONPATH"] = os.path.join(app_dir, "evaluation_utils") + os.pathsep + app_dir
        env["GAME_ID"] = game_name

        log_file_path = os.path.join(game_data_dir, "game_server.log")
        self.output_files[game_name] = open(log_file_path, "w")

        proc = subprocess.Popen(cmd, env=env, stdout=self.output_files[game_name], stderr=self.output_files[game_name])
        self.game_servers_procs[game_name] = proc
        self.proc_start_times[game_name] = time.time()

        return proc

    def start_game_servers(self, games: list[str] | None = None):
        self.renderer.event("Initializing game servers...")

        game_list = games or self.games

        for game_name in game_list:
            self.launch_game_server(game_name)
            # Avoid long per-game delays; servers should come up in parallel.
            # A tiny stagger helps prevent resource spikes on some systems.
            time.sleep(0.05)

        time.sleep(1.5)
        self.renderer.event("All game servers launched successfully")
        if self._live_export_aggregator:
            self._live_export_aggregator.start()
    
    def clean_up_game_server(self, game_name: str):
        """
        Terminate a game server process and close any associated resources.

        Important: cleanup must NOT depend on the existence of game_results.json.
        Servers can crash or runs can be interrupted before results are written,
        and we still need to ensure processes and file handles are released.
        """
        proc = self.game_servers_procs.get(game_name)
        if proc is not None:
            try:
                # If still running, try graceful terminate first, then hard kill.
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                else:
                    # Reap process if already exited.
                    try:
                        proc.wait(timeout=0.2)
                    except Exception:
                        pass
            finally:
                # Always remove the proc entry even if terminate/kill raised.
                self.game_servers_procs.pop(game_name, None)

        f = self.output_files.pop(game_name, None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
    
    def stop_game_server(self, game_name: str, silent: bool = False):
        if game_name in self.game_servers_procs:
            if self.game_servers_procs[game_name].poll() is not None:
                self.clean_up_game_server(game_name)
                return

            if not silent:
                self.renderer.event(f"Shutting down {game_name}")
            # Only set to "stopped" if not already in a terminal state
            current_status = self.renderer.state.server_status_by_game.get(game_name)
            if current_status not in ("completed", "failed", "stopped"):
                self.renderer.set_server_status(game_name, "stopped")
            self.clean_up_game_server(game_name)

    def force_stop_all_games(self):
        for game_name in list(self.game_servers_procs.keys()):
            self.stop_game_server(game_name, silent=True)
        if self._live_export_aggregator:
            self._live_export_aggregator.stop()
    
    def wait_for_games_to_finish(self):
        completed_games: set[str] = set()
        total_games = len(self.game_servers_procs)

        while len(completed_games) < total_games:
            time.sleep(10)
            # Iterate over a snapshot in case we stop/cleanup while iterating.
            for game_name, proc in list(self.game_servers_procs.items()):
                if game_name in completed_games:
                    continue

                results_path = os.path.join(GAME_DATA_DIR, game_name, "game_results.json")
                return_code = proc.poll()

                if return_code is not None:
                    # If the process exited cleanly, allow a small grace period for the
                    # results file to appear (avoid false "crash" on delayed writes).
                    if return_code == 0 and not os.path.exists(results_path):
                        grace_deadline = time.time() + 2.0
                        while time.time() < grace_deadline and not os.path.exists(results_path):
                            time.sleep(0.1)

                    if return_code != 0 or not os.path.exists(results_path):
                        self.renderer.warn(f"Game server {game_name} crashed with return code {return_code}")
                        self.renderer.set_server_status(game_name, "failed")
                        self.force_stop_all_games()
                        return

                if os.path.exists(results_path):
                    time.sleep(5)  # give a buffer for any pending writes
                    self.renderer.set_server_status(game_name, "completed")
                    self._update_scores_from_disk()
                    self.renderer.event(f"Game {game_name} completed")
                    if self._live_export_aggregator:
                        self._live_export_aggregator.finalize_game_media(game_name)
                    completed_games.add(game_name)

        if self._live_export_aggregator:
            self._live_export_aggregator.finalize_all_media()



if __name__ == "__main__":
    from evaluation_utils.renderer import get_renderer

    renderer = get_renderer()
    renderer.start(local=True)

    try:
        game_launcher = GameLauncher(renderer)
        game_launcher.start_game_servers()
        game_launcher.wait_for_games_to_finish()
    finally:
        game_launcher.force_stop_all_games()
        renderer.stop()
