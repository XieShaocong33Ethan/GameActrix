import asyncio
import argparse

from evaluation_utils.runner import Runner
from evaluation_utils.commons import setup_logging, GAME_DATA_DIR, GAME_SERVER_PORTS
from evaluation_utils.renderer import get_renderer


def main():
    parser = argparse.ArgumentParser(description="Orak Starter Kit Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--session-id", default=None, help="Use existing session id instead of creating a new session")
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    parser.add_argument(
        "--games",
        nargs="+",
        choices=list(GAME_SERVER_PORTS.keys()),
        help="Only run these games (space-separated list). Only supported in LOCAL mode.",
    )
    args = parser.parse_args()

    # Enforce that game selection is only supported in local mode
    if args.games and not args.local:
        parser.error("--games can only be used together with --local")

    setup_logging(verbose=args.verbose)

    # Initialize the centralized renderer
    renderer = get_renderer()
    renderer.start(
        local=args.local,
        session_id=args.session_id,
        game_data_path=GAME_DATA_DIR
    )

    try:
        # Only pass a game subset in local mode; remote mode always runs all games
        selected_games = args.games if args.local else None
        runner = Runner(session_id=args.session_id, local=args.local, renderer=renderer, games=selected_games)
        asyncio.run(runner.evaluate_all_games())

        # Show final summary with total score
        total_score = sum(runner.scores.values())
        renderer.show_final_summary("all_games", total_score)
    except Exception:
        # Mark evaluation as failed
        renderer.complete_evaluation(success=False)
        raise
    finally:
        renderer.stop()


if __name__ == "__main__":
    main()
