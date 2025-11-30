"""Command-line interface using OpenAI agent with function calling."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .agent import AgentError, run_agent
from .data_loader import describe_dataset, load_csvs
from .session_state import Turn, initialize_session

console = Console()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with CSV data using OpenAI agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default CSVs in current directory
  python -m agentic_analyst.cli_agent
  
  # Specify custom CSVs
  python -m agentic_analyst.cli_agent --csv data1.csv --csv data2.csv
  
  # Use different model
  python -m agentic_analyst.cli_agent --model gpt-4
        """
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        help="Path to a CSV file (repeat for multiple datasets). "
             "Default: case_study_germany_sample.csv and case_study_germany_treatment_costs_sample.csv",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="Optional dataset name matching the order of --csv arguments.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated artifacts (default: outputs).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model to use (default: from OPENAI_MODEL env or gpt-5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    load_dotenv()
    
    args = parse_args(argv)
    
    # Use default CSVs if none provided
    if args.csv is None:
        default_csvs = [
            "case_study_germany_sample.csv",
            "case_study_germany_treatment_costs_sample.csv"
        ]
        # Check if default files exist
        from pathlib import Path
        existing_defaults = [csv for csv in default_csvs if Path(csv).exists()]
        if not existing_defaults:
            console.print("[bold red]❌ No CSV files found![/bold red]")
            console.print("Please provide CSV files using --csv flag or ensure default files exist:")
            for csv in default_csvs:
                console.print(f"  • {csv}")
            raise SystemExit(1)
        args.csv = existing_defaults
        console.print(f"[dim]Using default CSV files: {', '.join(existing_defaults)}[/dim]")
    
    if args.name and len(args.name) != len(args.csv):
        raise SystemExit("--name count must match --csv count.")
    
    # Load datasets
    console.print("\n[bold cyan]Loading datasets...[/bold cyan]")
    datasets = load_csvs(args.csv, names=args.name)
    session = initialize_session(datasets, output_dir=args.output_dir)
    
    # Determine model
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-5")
    
    # Display loaded datasets in a nice table
    console.print(f"\n[bold green]✓[/bold green] Using OpenAI model: [cyan]{model}[/cyan]\n")
    
    dataset_table = Table(title="📊 Loaded Datasets", show_header=True, header_style="bold magenta")
    dataset_table.add_column("Dataset Name", style="cyan", no_wrap=False)
    dataset_table.add_column("Rows", justify="right", style="yellow")
    dataset_table.add_column("Columns", justify="right", style="yellow")
    dataset_table.add_column("Sample Columns", style="dim")
    
    for ds in datasets.values():
        sample_cols = ", ".join(list(ds.df.columns)[:5])
        if len(ds.df.columns) > 5:
            sample_cols += ", ..."
        dataset_table.add_row(
            ds.name,
            str(len(ds.df)),
            str(len(ds.df.columns)),
            sample_cols
        )
    
    console.print(dataset_table)
    console.print("\n[dim]Commands: /datasets, /history, /help, /exit[/dim]")
    console.print("[dim]Ask questions about your data in natural language.[/dim]\n")
    
    while True:
        try:
            user_text = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        
        if not user_text:
            continue
        
        if user_text.lower() in {"/exit", ":q", "quit"}:
            break
        
        if _handle_command(user_text, session):
            continue
        
        _handle_question(user_text, session, model)


def _handle_command(text: str, session) -> bool:
    lowered = text.lower()
    if lowered == "/datasets":
        dataset_table = Table(title="📊 Available Datasets", show_header=True, header_style="bold magenta")
        dataset_table.add_column("Dataset Name", style="cyan")
        dataset_table.add_column("Rows", justify="right", style="yellow")
        dataset_table.add_column("Columns", justify="right", style="yellow")
        dataset_table.add_column("Sample Columns", style="dim")
        
        for ds in session.datasets.values():
            sample_cols = ", ".join(list(ds.df.columns)[:5])
            if len(ds.df.columns) > 5:
                sample_cols += ", ..."
            dataset_table.add_row(
                ds.name,
                str(len(ds.df)),
                str(len(ds.df.columns)),
                sample_cols
            )
        console.print(dataset_table)
        return True
    if lowered == "/history":
        console.print("\n[bold cyan]📜 Conversation History[/bold cyan]\n")
        for turn in session.chat_history:
            if turn.role == "user":
                console.print(f"[bold blue]You:[/bold blue] {turn.message}")
            else:
                console.print(f"[bold green]Agent:[/bold green] {turn.message}")
        print()
        return True
    if lowered == "/help":
        help_panel = Panel(
            "[cyan]Chat about your data using natural language.[/cyan]\n"
            "The agent will write and execute Python code to answer your questions.\n\n"
            "[bold]Commands:[/bold]\n"
            "  • [yellow]/datasets[/yellow] - Show loaded datasets\n"
            "  • [yellow]/history[/yellow] - View conversation history\n"
            "  • [yellow]/help[/yellow] - Show this help message\n"
            "  • [yellow]/exit[/yellow] - Quit the session",
            title="❓ Help",
            border_style="blue",
        )
        console.print(help_panel)
        return True
    return False


def _handle_question(text: str, session, model: str) -> None:
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]🤔 Agent is thinking...[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]")
    
    try:
        result = run_agent(text, session, model=model, verbose=True)
    except AgentError as exc:
        console.print(f"\n[bold red]❌ Error: {exc}[/bold red]")
        return
    
    # The detailed step logging now happens in real-time inside run_agent()
    # Show final summary of saved files if any
    if result["artifacts"]:
        console.print("\n[bold green]" + "="*80 + "[/bold green]")
        console.print("[bold green]📁 SAVED FILES SUMMARY[/bold green]")
        console.print("[bold green]" + "="*80 + "[/bold green]")
        for artifact in result["artifacts"]:
            console.print(f"  💾 [cyan]{artifact['type']}[/cyan]: [green]{artifact['path']}[/green]")
        print()
    
    # Display response with rich formatting
    print()
    response_panel = Panel(
        Markdown(result['response']),
        title="🤖 Agent Response",
        border_style="green",
        padding=(1, 2),
    )
    console.print(response_panel)
    print()
    
    # Save to history
    session.append_turn(Turn(role="user", message=text))
    session.append_turn(
        Turn(
            role="assistant",
            message=result["response"],
            insights=result["response"],
            artifacts=[],  # Could enhance to track artifact metadata
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])

