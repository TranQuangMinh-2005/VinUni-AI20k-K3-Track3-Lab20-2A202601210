"""Analyst agent: biến research notes thành phân tích có cấu trúc."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import (
    LLMClient,
    LLMClientError,
    LLMCompleter,
)
from multi_agent_research_lab.utils.timer import elapsed_timer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a sharp analyst. Read the research notes and the numbered source list, then produce
analysis notes with these sections:

1. Key claims — main takeaways with evidence strength (strong / moderate / weak)
2. Agreements & disagreements — where sources align or conflict
3. Gaps — what is missing or weakly supported
4. Reliability — caveats about the sources (mock sources, dated info, vendor bias...)

Be concise (under 300 words). Reference sources as [n] where relevant.
"""


class AnalystAgent(BaseAgent):
    """Đọc research_notes → sinh analysis_notes (đánh giá độ tin cậy, tìm lỗ hổng)."""

    name = "analyst"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMCompleter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm_client

    def _ensure_llm(self) -> LLMCompleter:
        if self._llm is None:
            self._llm = LLMClient(settings=self._settings)
        return self._llm

    def run(self, state: ResearchState) -> ResearchState:
        """Điền `state.analysis_notes`."""
        # Idempotent: đã phân tích rồi → không làm lại
        if state.analysis_notes:
            return state

        llm = self._ensure_llm()
        user_prompt = self._build_prompt(state)

        try:
            with elapsed_timer() as elapsed:
                response = llm.complete(_SYSTEM_PROMPT, user_prompt)
            latency = elapsed()
        except LLMClientError as exc:
            state.errors.append(f"analyst LLM failed: {exc}")
            raise AgentExecutionError(f"analyst could not produce analysis: {exc}") from exc

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "cost_usd": response.cost_usd,
                    "latency_seconds": round(latency, 2),
                },
            )
        )
        state.add_trace_event(
            "analyst_llm",
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": round(latency, 2),
            },
        )
        logger.info("analyst done: %d chars of analysis", len(response.content))
        return state

    @staticmethod
    def _build_prompt(state: ResearchState) -> str:
        """Ghép research notes + danh sách nguồn (chỉ title/url để đỡ tốn token)."""
        lines = [f"Research question: {state.request.query}", "", "Research notes:"]
        lines.append(state.research_notes or "(no research notes)")
        lines.append("")
        lines.append("Source list:")
        for i, doc in enumerate(state.sources, start=1):
            mock_tag = " [MOCK]" if doc.metadata.get("mock") else ""
            lines.append(f"[{i}] {doc.title}{mock_tag} — {doc.url or 'n/a'}")
        lines.append("")
        lines.append("Write the analysis notes now.")
        return "\n".join(lines)
