import os
import sys
import argparse

# Setup paths
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, app_dir)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from evaluation_utils.grpc_server import serve
from evaluation_utils.mcp_game_servers.base_game_logic import GameLogic


def main():
    """
    StarCraft II gRPC Server Entry Point

    This server wraps the StarCraft II environment with gRPC for remote access.
    The multiprocessing architecture in star_craft_env.py is preserved:
    - Manager.dict() for transaction state between parent-child processes
    - Lock for synchronizing access to shared transaction dict
    - Event objects for parent-child coordination (isReadyForNextStep, game_end_event, done_event)
    - Separate process for SC2 game execution (sc2_run_game target)

    The gRPC layer provides:
    - Session management with timeout-based expiry
    - Fail-fast locking: non-blocking acquire() immediately rejects concurrent requests
    - Sequential processing: max_workers=1 ensures deterministic state transitions

    When a Step RPC is executed:
    1. gRPC servicer acquires _action_lock (fail-fast)
    2. Calls game.dispatch_action_and_get_score(action)
    3. Inside that method, StarCraftEnv.step() executes:
       - Writes action to transaction dict with lock
       - Waits on isReadyForNextStep or done_event
       - SC2 child process polls transaction dict, executes action
       - Child process sets event when ready or done
    4. Parent process (this server) returns result to client

    This design prevents race conditions while maintaining the multiprocessing architecture.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    port = int(os.getenv("PORT", "33003"))  # StarCraft II default port
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )

    game = GameLogic(config_path)
    print(f"Starting StarCraft II gRPC server on port {port}")
    serve(game, port)


if __name__ == "__main__":
    main()
