"""Benchmark report rendering."""

from collections import defaultdict

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _aggregate(metrics: list[BenchmarkMetrics]) -> dict[str, BenchmarkMetrics]:
    """Gộp nhiều lần chạy cùng run_name → một dòng trung bình."""
    grouped: dict[str, list[BenchmarkMetrics]] = defaultdict(list)
    for item in metrics:
        grouped[item.run_name].append(item)

    aggregated: dict[str, BenchmarkMetrics] = {}
    for name, items in grouped.items():
        aggregated[name] = BenchmarkMetrics(
            run_name=name,
            latency_seconds=round(_avg([m.latency_seconds for m in items]) or 0.0, 2),
            estimated_cost_usd=_avg(
                [m.estimated_cost_usd for m in items if m.estimated_cost_usd is not None]
            ),
            quality_score=_avg(
                [m.quality_score for m in items if m.quality_score is not None]
            ),
            citation_coverage=_avg(
                [m.citation_coverage for m in items if m.citation_coverage is not None]
            ),
            failure_rate=_avg(
                [m.failure_rate for m in items if m.failure_rate is not None]
            ),
            notes=f"averaged over {len(items)} run(s)",
        )
    return aggregated


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render bảng so sánh + các mục phân tích dạng markdown.

    Các mục `<!-- (điền) ... -->` là chỗ chèn phân tích thật
    (failure mode, trace link, kết luận) sau khi xem số liệu + trace UI.
    """
    aggregated = _aggregate(metrics)

    lines = [
        "# Benchmark Report: Single-Agent vs Multi-Agent",
        "",
        f"> Số lần chạy được gộp: {len(metrics)} (mỗi query × mỗi chế độ)",
        "",
        "## Bảng so sánh",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (0-10) | Citation cov. | Failure rate | "
        "Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in aggregated.values():
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines += [
        "",
        "## Phân tích",
        "",
        "<!-- (điền) ai thắng về latency, cost, quality, citation? Vì sao? -->",
        "",
        "## Failure mode gặp phải & cách fix",
        "",
        "<!-- (điền) mô tả 1 failure mode đã gặp (trace minh chứng) và cách bạn fix -->",
        "",
        "## Trace evidence",
        "",
        "<!-- (điền) dán link/screenshot LangSmith hoặc Langfuse -->",
        "",
        "## Kết luận: khi nào dùng multi-agent?",
        "",
        "<!-- (điền) rút ra điều kiện dùng multi-agent từ số liệu trên -->",
    ]
    return "\n".join(lines) + "\n"
