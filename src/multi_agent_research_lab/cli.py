"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    _DEFAULT_QUERIES,
    make_baseline_runner,
    make_multi_agent_runner,
    run_benchmark,
)
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import setup_tracing
from multi_agent_research_lab.services.llm_client import LLMClientError

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    """Khởi tạo chung cho mọi command: logging + LangSmith tracing (nếu có key)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    setup_tracing(settings)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline end-to-end.

    Một model duy nhất làm tất cả: nhận query → trả lời trực tiếp (không tách bước).
    Đây là "mốc so sánh" để benchmark với multi-agent workflow.
    """

    _init()
    request = _parse_query(query)

    # Runner dùng chung với benchmark command → số liệu 2 nơi nhất quán
    try:
        state = make_baseline_runner()(request.query)
    except LLMClientError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))

    # In nhanh các số liệu để ghi lại, phục vụ so sánh với multi-agent
    event = state.trace[-1]["payload"] if state.trace else {}
    cost_text = f"${event['cost_usd']:.6f}" if event.get("cost_usd") is not None else "n/a"
    console.print(
        Panel.fit(
            f"model: {event.get('model')}\n"
            f"latency: {event.get('latency_seconds')}s\n"
            f"tokens: {event.get('input_tokens')} in / {event.get('output_tokens')} out\n"
            f"estimated cost: {cost_text}",
            title="Baseline Metrics",
        )
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow end-to-end."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except LLMClientError as exc:
        console.print(Panel.fit(str(exc), title="LLM Error", style="red"))
        raise typer.Exit(code=1) from exc
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[
        list[str] | None, typer.Option("--query", "-q", help="Research query (lặp lại được)")
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Đường dẫn file markdown report")
    ] = Path("reports/benchmark_report.md"),
) -> None:
    """Benchmark single-agent vs multi-agent trên cùng bộ query, ghi report markdown."""

    _init()
    queries = query or _DEFAULT_QUERIES

    # Tạo runner 1 lần, chạy lại cho từng query (không phải khởi tạo lại client mỗi lần)
    baseline_runner = make_baseline_runner()
    multi_runner = make_multi_agent_runner()

    metrics: list[BenchmarkMetrics] = []
    for q in queries:
        console.print(f"[bold]Benchmark query:[/bold] {q}")
        _, m_baseline = run_benchmark("single-agent", q, baseline_runner)
        _, m_multi = run_benchmark("multi-agent", q, multi_runner)
        metrics.extend([m_baseline, m_multi])
        console.print(
            f"  single-agent: {m_baseline.latency_seconds}s, "
            f"cost=${m_baseline.estimated_cost_usd}"
        )
        console.print(
            f"  multi-agent: {m_multi.latency_seconds}s, cost=${m_multi.estimated_cost_usd}"
        )

    report = render_markdown_report(metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(f"[green]Report written to {output}[/green]")
    console.print(report)


if __name__ == "__main__":
    app()
