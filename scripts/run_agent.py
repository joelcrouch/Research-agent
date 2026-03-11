"""
scripts/run_agent.py

CLI entrypoint. Run with:
    make run                            # uses default query
    make run query="CRISPR gene editing"
    python scripts/run_agent.py --query "long-term potentiation"
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel

from agent.graph import run
from config.logging import configure_logging
from config.settings import settings

console = Console()


def main() -> None:
    configure_logging(settings().log_level)

    parser = argparse.ArgumentParser(description="Academic Research Intelligence Agent")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="long-term potentiation",
        help="Research topic to search for",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            f"[bold blue]Query:[/bold blue] {args.query}",
            title="Research Agent",
            border_style="blue",
        )
    )

    state = run(args.query)

    if state.get("error"):
        console.print(f"\n[bold red]Error:[/bold red] {state['error']}")
        sys.exit(1)

    console.print(f"\n{state['final_response']}")
    console.print(f"\n[dim]Trace ID: {state['trace_id']}[/dim]")
    console.print(f"[dim]Trace written to: logs/trace_{state['trace_id']}.json[/dim]")


if __name__ == "__main__":
    main()
