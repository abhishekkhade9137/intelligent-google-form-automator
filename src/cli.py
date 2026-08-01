"""
Command Line Interface for running automated Google Form filler campaigns using Typer and Rich.
Enhanced with real-world demographic benchmarks, deterministic Hare-Niemann rebalancing, and noise simulation.
"""
import os
import time
import random
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from typing import Optional

from src.automation import BrowserEngine, FormExtractor, FormFillerEngine
from src.synthesis import AIGenerationEngine
from src.persistence import SubmissionTracker
from src.statistical import get_profile_by_name, PRESET_PROFILES

app = typer.Typer(help="Automated AI Google Form Filler & Dataset Generator")
console = Console()


@app.command()
def run(
    url: str = typer.Option(..., "--url", "-u", help="URL of the target Google Form to complete."),
    count: int = typer.Option(5, "--count", "-n", help="Number of unique responses to generate."),
    provider: str = typer.Option("gemini", "--provider", "-p", help="AI Provider: 'gemini', 'openai', or 'ollama'."),
    model: str = typer.Option("gemini-2.5-flash", "--model", "-m", help="Specific model name (e.g., gemini-2.5-flash, llama3.1)."),
    context: str = typer.Option("General user survey", "--context", "-c", help="Topic or purpose of the survey to guide responses."),
    guidance: Optional[str] = typer.Option(None, "--guidance", "-g", help="Demographic restrictions (e.g., 'Indian names, 21-26yo engineering students')."),
    benchmark: Optional[str] = typer.Option("so_2024_devs", "--benchmark", "-b", help="Real-world demographic benchmark profile ID or name."),
    no_rebalance: bool = typer.Option(False, "--no-rebalance", help="Disable deterministic Hare-Niemann quota rebalancing."),
    no_noise: bool = typer.Option(False, "--no-noise", help="Disable human survey noise (speed-running & short replies)."),
    headless: bool = typer.Option(False, "--headless", help="Run browser in background without UI display."),
    min_delay: int = typer.Option(3, "--min-delay", help="Minimum wait time between submissions in seconds."),
    max_delay: int = typer.Option(8, "--max-delay", help="Maximum wait time between submissions in seconds."),
    db_path: str = typer.Option("submissions_history.db", "--db", help="SQLite file path to record submissions.")
):
    """Executes a multi-submission automated campaign against a target Google Form with mathematical rebalancing."""
    console.print(Panel(
        f"[bold cyan]Automated AI Google Form Campaign Starter[/bold cyan]\n"
        f"Target URL: [green]{url}[/green]\n"
        f"Submissions Goal: [bold yellow]{count}[/bold yellow] | AI Engine: [magenta]{provider.upper()} ({model})[/magenta]\n"
        f"Demographics Guidance: [italic]{guidance or 'Default diversified distribution'}[/italic]\n"
        f"Benchmark Profile: [bold green]{benchmark or 'None'}[/bold green] | Rebalancing: [bold cyan]{not no_rebalance}[/bold cyan]",
        expand=False
    ))

    # Initialize components
    tracker = SubmissionTracker(db_path=db_path)
    generator = AIGenerationEngine(
        provider=provider,
        model_name=model,
        status_callback=lambda m: console.print(f"[dim blue]• {m}[/dim blue]")
    )

    console.print("[yellow][i]Launching Playwright persistent browser session & analyzing DOM...[/i][/yellow]")
    with BrowserEngine(headless=headless) as browser:
        browser.navigate_and_check_auth(url, pause_on_login=not headless)
        
        extractor = FormExtractor(browser.page)
        schema = extractor.extract_schema(url)
        
        console.print(f"\n[bold green][SUCCESS] Discovered Google Form Schema:[/bold green] '{schema.title}' ({len(schema.all_questions)} questions discovered on page 1)")
        
        # 1. Pre-compute and mathematically rebalance entire campaign in memory
        console.print(f"\n[bold cyan]⚡ Phase 1: In-Memory Dataset Synthesis & Hare-Niemann Statistical Rebalancing...[/bold cyan]")
        batch_results = generator.generate_batch_campaign(
            schema=schema,
            count=count,
            context=context,
            demographic_guidance=guidance,
            benchmark_profile=benchmark,
            apply_noise=not no_noise,
            apply_rebalancing=not no_rebalance
        )
        
        console.print(f"[bold green]✅ Pre-flight dataset generated & mathematically verified ({len(batch_results)} profiles ready).[/bold green]\n")
        
        filler = FormFillerEngine(browser, status_callback=lambda m: console.print(f"[dim green]🌐 {m}[/dim green]"))

        # 2. Start batch execution loop with Progress Bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task_id = progress.add_task("Streaming verified answers to Google Form...", total=count)

            for idx, answer_set in enumerate(batch_results):
                persona = answer_set.persona
                progress.update(task_id, description=f"[bold cyan]Submission {idx+1}/{count} - Filling as '{persona.name}'...")
                
                # Ensure browser is on fresh form input URL
                if idx > 0:
                    browser.page.goto(url, wait_until="domcontentloaded")

                success, msg = filler.fill_and_submit(schema, answer_set)
                
                # Log outcome in tracker
                tracker.add_record(url, answer_set, success=success, error_message=None if success else msg)

                # Display summary table of the submitted entry
                table = Table(show_header=True, header_style="bold green" if success else "bold red", title=f"Attempt #{idx+1}: {persona.name} ({persona.age}yo {persona.occupation})")
                table.add_column("Question", style="dim", width=40)
                table.add_column("Verified Answer Value", style="bold")
                
                for ans in answer_set.answers:
                    val_str = ", ".join(ans.value) if isinstance(ans.value, list) else str(ans.value)
                    table.add_row(ans.question_title[:38], val_str)
                
                console.print(table)
                console.print(f"[bold {'green' if success else 'red'}][Result: {msg}][/bold {'green' if success else 'red'}]\n")

                progress.advance(task_id)

                # Randomized human-like inter-submission pause
                if idx < count - 1:
                    delay = random.randint(min_delay, max_delay)
                    progress.update(task_id, description=f"[yellow]Waiting {delay}s before next entry to mimic authentic intervals...")
                    time.sleep(delay)

    console.print(Panel(
        f"[bold green]Campaign Fully Complete![/bold green]\n"
        f"All records have been saved into database: '{db_path}'\n"
        f"To view visualizations and charts, run the lightweight web UI:\n"
        f"[bold cyan]streamlit run app.py[/bold cyan]",
        expand=False
    ))


if __name__ == "__main__":
    app()
