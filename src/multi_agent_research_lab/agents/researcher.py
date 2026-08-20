"""Researcher agent: tìm nguồn + viết research notes."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import (
    LLMClient,
    LLMClientError,
    LLMCompleter,
)
from multi_agent_research_lab.services.search_client import (
    SearchClient,
    SearchClientError,
    SearchProvider,
)
from multi_agent_research_lab.utils.timer import elapsed_timer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a diligent researcher. Based on the provided web sources, write concise research notes.

Rules:
- Every factual claim MUST be followed by [n] referencing the source number it came from.
- If the sources do not support a claim, do NOT include it (no hallucination).
- Group findings by topic; use bullet points.
- Keep the notes under 300 words.
"""


class ResearcherAgent(BaseAgent):
    """Tìm kiếm nguồn qua SearchClient, tổng hợp thành research notes ghi vào state."""

    name = "researcher"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMCompleter | None = None,
        search_client: SearchProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_client
        self._search = search_client

    def _ensure_llm(self) -> LLMCompleter:
        if self._llm is None:
            self._llm = LLMClient(settings=self._settings)
        return self._llm

    def _ensure_search(self) -> SearchProvider:
        if self._search is None:
            self._search = SearchClient(settings=self._settings)
        return self._search

    def run(self, state: ResearchState) -> ResearchState:
        """Điền `state.sources` + `state.research_notes`."""
        # Idempotent: đã có đủ dữ liệu → không làm lại (phòng supervisor route trùng)
        if state.sources and state.research_notes:
            return state

        docs = self._collect_sources(state)

        llm = self._ensure_llm()
        user_prompt = self._build_prompt(state.request.query, docs)

        try:
            with elapsed_timer() as elapsed:
                response = llm.complete(_SYSTEM_PROMPT, user_prompt)
            latency = elapsed()
        except LLMClientError as exc:
            # LLM chết → ném AgentExecutionError để workflow ghi lỗi và supervisor
            # quyết định retry (vòng lặp này sẽ bị chặn bởi guardrail max_iterations)
            state.errors.append(f"researcher LLM failed: {exc}")
            raise AgentExecutionError(f"researcher could not produce notes: {exc}") from exc

        state.sources = docs
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "n_sources": len(docs),
                    "cost_usd": response.cost_usd,
                    "latency_seconds": round(latency, 2),
                },
            )
        )
        state.add_trace_event(
            "researcher_llm",
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": round(latency, 2),
            },
        )
        logger.info(
            "researcher done: %d sources, %d chars of notes", len(docs), len(response.content)
        )
        return state

    def _collect_sources(self, state: ResearchState) -> list[SourceDocument]:
        """Gọi search; nếu search chết → fallback sang mock nguồn (ghi lỗi vào state)."""
        search = self._ensure_search()
        max_results = state.request.max_sources

        try:
            docs = search.search(state.request.query, max_results=max_results)
        except SearchClientError as exc:
            # Guardrail fallback: không để thiếu search làm chết cả pipeline.
            # Dùng nguồn mock (đánh dấu mock=True) và ghi lỗi để trace/benchmark biết.
            state.errors.append(f"search failed, using mock sources: {exc}")
            state.add_trace_event("search_error", {"error": str(exc)})
            docs = SearchClient.mock_search(state.request.query, max_results=min(max_results, 3))
        return docs

    @staticmethod
    def _build_prompt(query: str, docs: list[SourceDocument]) -> str:
        """Ghép query + danh sách nguồn đánh số [1], [2]... thành user prompt."""
        lines = [f"Research question: {query}", "", "Web sources:"]
        for i, doc in enumerate(docs, start=1):
            mock_tag = " [MOCK SOURCE]" if doc.metadata.get("mock") else ""
            lines.append(
                f"[{i}] {doc.title}{mock_tag}\n    URL: {doc.url or 'n/a'}\n    {doc.snippet}"
            )
        lines.append("")
        lines.append("Write the research notes now.")
        return "\n".join(lines)
