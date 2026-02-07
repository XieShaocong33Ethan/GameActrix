import os
import sys
import argparse

# Setup paths - go up to the project root
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, app_dir)

from evaluation_utils.grpc_server import serve
from evaluation_utils.mcp_game_servers.base_game_logic import GameLogic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    port = int(os.getenv("PORT", "33002"))  # 2048 default port
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )

    game = GameLogic(config_path)
    print(f"Starting 2048 gRPC server on port {port}")
    serve(game, port)

if __name__ == "__main__":
    main()
