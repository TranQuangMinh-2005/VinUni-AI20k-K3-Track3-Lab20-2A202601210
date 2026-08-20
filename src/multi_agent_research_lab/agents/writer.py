"""Writer agent: tổng hợp research + analysis thành final answer có citation."""

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
You are a senior technical writer. Write a clear, well-structured final answer using ONLY
the provided research notes and analysis notes. Cite sources inline as [n] matching the
numbered source list. Do NOT invent facts that are not in the notes.
"""


class WriterAgent(BaseAgent):
    """Tổng hợp notes → `state.final_answer` kèm citation trỏ về sources."""

    name = "writer"

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
        """Điền `state.final_answer`."""
        # Idempotent: đã viết rồi → không làm lại
        if state.final_answer:
            return state

        llm = self._ensure_llm()
        user_prompt = self._build_prompt(state)

        try:
            with elapsed_timer() as elapsed:
                response = llm.complete(_SYSTEM_PROMPT, user_prompt)
            latency = elapsed()
        except LLMClientError as exc:
            state.errors.append(f"writer LLM failed: {exc}")
            raise AgentExecutionError(f"writer could not produce final answer: {exc}") from exc

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "cost_usd": response.cost_usd,
                    "latency_seconds": round(latency, 2),
                },
            )
        )
        state.add_trace_event(
            "writer_llm",
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": round(latency, 2),
            },
        )
        logger.info("writer done: %d chars of final answer", len(response.content))
        return state

    @staticmethod
    def _build_prompt(state: ResearchState) -> str:
        """Ghép query + audience + notes + danh sách nguồn thành user prompt."""
        lines = [
            f"Research question: {state.request.query}",
            f"Target audience: {state.request.audience}",
            "",
            "Research notes:",
            state.research_notes or "(none)",
            "",
            "Analysis notes:",
            state.analysis_notes or "(none)",
            "",
            "Source list:",
        ]
        for i, doc in enumerate(state.sources, start=1):
            lines.append(f"[{i}] {doc.title} — {doc.url or 'n/a'}")
        lines.append("")
        lines.append("Write the final answer now (cite sources as [n]).")
        return "\n".join(lines)
