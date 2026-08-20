"""Benchmark single-agent vs multi-agent: latency, cost, quality, citation, failure."""

import logging
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.utils.timer import elapsed_timer

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]

# Bộ query mặc định khi chạy `benchmark` không truyền --query.
# Chọn câu hỏi nghiên cứu thật để cả 2 chế độ đều phải tìm + phân tích + viết.
_DEFAULT_QUERIES = [
    "Research GraphRAG state-of-the-art and write a 500-word summary",
    "Compare supervised and hierarchical multi-agent LLM orchestration patterns",
]


def make_baseline_runner(settings: Settings | None = None) -> Runner:
    """Single-agent baseline: MỘT lần gọi LLM làm tất cả (không search, không tách bước).

    Dùng chung với CLI command `baseline` để số liệu 2 nơi nhất quán.
    """
    resolved = settings or get_settings()
    client = LLMClient(settings=resolved)

    system_prompt = (
        "You are a research assistant. Answer the user's question directly, "
        "thoroughly and accurately. Structure your answer with a short summary, "
        "key findings, and a conclusion. If you are unsure about a fact, say so."
    )

    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        with elapsed_timer() as elapsed:
            response = client.complete(system_prompt, query)
        state.final_answer = response.content
        state.add_trace_event(
            "baseline_llm",
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": round(elapsed(), 2),
            },
        )
        return state

    return runner


def make_multi_agent_runner(settings: Settings | None = None) -> Runner:
    """Multi-agent: chạy graph Supervisor → Researcher → Analyst → Writer."""
    # Import muộn để tránh vòng import (workflow imports services, không ngược lại)
    from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow

    workflow = MultiAgentWorkflow(settings=settings)

    def runner(query: str) -> ResearchState:
        return workflow.run(ResearchState(request=ResearchQuery(query=query)))

    return runner


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Chạy 1 runner, đo latency + tổng hợp metric vào BenchmarkMetrics."""
    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:  # benchmark không được chết vì 1 lần chạy lỗi
        logger.exception("benchmark run %s failed for query %.60s", run_name, query)
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(f"{run_name} crashed: {exc}")
        failed = True
    latency = perf_counter() - started

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 2),
        estimated_cost_usd=sum_trace_cost(state),
        quality_score=score_quality(state),
        citation_coverage=citation_coverage(state),
        failure_rate=1.0 if failed else (0.0 if state.final_answer else 1.0),
    )
    return state, metrics


def sum_trace_cost(state: ResearchState) -> float | None:
    """Tổng chi phí ước tính: cộng mọi event có cost_usd trong state.trace."""
    costs = [
        float(event["payload"]["cost_usd"])
        for event in state.trace
        if event.get("payload", {}).get("cost_usd") is not None
    ]
    return round(sum(costs), 6) if costs else None


def citation_coverage(state: ResearchState) -> float | None:
    """Tỷ lệ nguồn được trích dẫn trong final_answer: số [i] xuất hiện / tổng nguồn."""
    if not state.final_answer or not state.sources:
        return None
    cited = sum(
        1 for i in range(1, len(state.sources) + 1) if f"[{i}]" in state.final_answer
    )
    return cited / len(state.sources)


def score_quality(state: ResearchState) -> float:
    """Rubric heuristic 0-10, chấm trên CHẤT LƯỢNG CÂU TRẢ LỜI (công bằng cho cả 2 chế độ).

    - có final_answer: 2 điểm
    - độ dài ≥ 100 từ: 2 điểm; ≥ 200 từ: +1
    - có citation [n]: 2 điểm
    - có cấu trúc (heading/bullet): 1 điểm
    - không có lỗi: 2 điểm
    Ghi chú: đây là rubric tự động để benchmark nhanh; điểm chính thức vẫn
    từ peer review (docs/peer_review_rubric.md).
    """
    score = 0.0
    answer = state.final_answer or ""
    if answer:
        score += 2
    words = len(answer.split())
    if words >= 100:
        score += 2
    if words >= 200:
        score += 1
    if "[" in answer and "]" in answer:
        score += 2
    if any(marker in answer for marker in ("#", "*", "- ")) or answer.count("\n") >= 4:
        score += 1
    if not state.errors:
        score += 2
    return min(score, 10.0)
