"""CLI entry point for the deep research orchestrator."""

import asyncio

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import get_settings
from .models import ClarificationRequest
from .orchestrator import ResearchOrchestrator

app = typer.Typer(
    name="deep-research",
    help="Deep Research Orchestrator - AI-powered research automation",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


async def handle_clarification(clarification: ClarificationRequest) -> str:
    """Interactive handler for clarification requests."""
    console.print(
        Panel(
            clarification.context or "I need some clarification before proceeding.",
            title="[bold yellow]Clarification Needed[/bold yellow]",
        )
    )

    answers = []
    for i, question in enumerate(clarification.questions, 1):
        console.print(f"\n[bold cyan]Question {i}:[/bold cyan] {question}")
        answer = Prompt.ask("[green]Your answer[/green]")
        answers.append(f"Q: {question}\nA: {answer}")

    return "\n\n".join(answers)


async def run_research(question: str, interactive: bool = True) -> None:
    """Run the research workflow."""
    settings = get_settings()

    if not settings.openai_api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY environment variable not set."
        )
        raise typer.Exit(1)

    orchestrator = ResearchOrchestrator(settings=settings)

    console.print(
        Panel(
            question,
            title="[bold blue]Research Question[/bold blue]",
        )
    )

    with console.status("[bold green]Planning research approach..."):
        result = await orchestrator.run(
            question,
            on_clarification=handle_clarification if interactive else None,
        )

    if result["status"] == "needs_clarification":
        console.print("\n[bold yellow]Clarification needed:[/bold yellow]")
        for q in result["clarification"]["questions"]:
            console.print(f"  • {q}")
        console.print(
            "\n[dim]Run again with answers to continue, "
            "or use --context to provide additional context.[/dim]"
        )
        return

    # Display results
    console.print("\n")
    console.print(
        Panel(
            f"[dim]Executed {len(result['tasks'])} research task(s)[/dim]",
            title="[bold green]Research Complete[/bold green]",
        )
    )

    console.print("\n[bold]Answer:[/bold]\n")
    console.print(Markdown(result["answer"]))


@app.callback(invoke_without_command=True)
def default_research(
    ctx: typer.Context,
    question: str = typer.Argument(
        None,
        help="The research question to investigate",
    ),
    context: str = typer.Option(
        "",
        "--context",
        "-c",
        help="Additional context or constraints for the research",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-n",
        help="Disable interactive clarification prompts",
    ),
) -> None:
    """
    Run a deep research task.

    Provide a research question and the orchestrator will:
    1. Analyze and potentially ask clarifying questions
    2. Decompose into specific research tasks
    3. Execute research in parallel where possible
    4. Synthesize findings into a comprehensive answer
    """
    # If a subcommand was invoked, skip
    if ctx.invoked_subcommand is not None:
        return

    # If no question provided, show help
    if question is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)

    full_question = question
    if context:
        full_question = f"{question}\n\nAdditional context: {context}"

    asyncio.run(run_research(full_question, interactive=not non_interactive))


@app.command()
def list_prompts() -> None:
    """List available prompt templates."""
    from pathlib import Path

    from .prompts import get_prompt_loader

    loader = get_prompt_loader(Path(__file__).parent.parent.parent / "prompts")
    prompts = loader.list_prompts()

    console.print("[bold]Available Prompt Templates:[/bold]\n")
    for name in prompts:
        template = loader.load(name)
        console.print(f"  [cyan]{name}[/cyan]")
        if template.description:
            console.print(f"    {template.description}")
        console.print()


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

