"""Critic agent (bonus): kiểm tra final_answer trước khi giao cho user."""

import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
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
You are a critical fact-checker. Review the final answer against the research notes and
source list. Report:
1. Citation check — any claims without a citation?
2. Hallucination check — any claim contradicted by (or absent from) the notes?
3. Coverage — any important finding from the notes missing in the answer?
Be concise (under 150 words). If everything is fine, say so explicitly.
"""


class CriticAgent(BaseAgent):
    """Agent phản biện (tùy chọn, bài bonus).

    Hai lớp kiểm tra:
    1. Deterministic (0 token): citation coverage + phát hiện trích dẫn
       ngoài khoảng nguồn (dấu hiệu hallucination).
    2. LLM review: đối chiếu final_answer với notes để tìm claim thiếu nguồn.

    Kết quả ghi vào `agent_results` + `trace`. KHÔNG ghi `state.errors` và
    KHÔNG chặn pipeline — critic là "người nhắc", không phải "người gác cổng".
    """

    name = "critic"

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
        """Kiểm tra final_answer và ghi findings vào state."""
        # Chưa có bài viết → không có gì để kiểm tra
        if not state.final_answer:
            return state

        # Lớp 1: kiểm tra deterministic (không tốn token)
        checks = self._deterministic_checks(state)
        state.add_trace_event("critic_checks", checks)

        # Lớp 2: LLM review
        llm = self._ensure_llm()
        try:
            with elapsed_timer() as elapsed:
                response = llm.complete(_SYSTEM_PROMPT, self._build_prompt(state))
            latency = elapsed()
        except LLMClientError as exc:
            # Critic fail KHÔNG được chặn pipeline → chỉ ghi nhẹ vào trace rồi bỏ qua
            state.add_trace_event("critic_llm_error", {"error": str(exc)})
            logger.warning("critic LLM failed, skipping review: %s", exc)
            return state

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=self._format_findings(checks, response.content),
                metadata={
                    "cost_usd": response.cost_usd,
                    "latency_seconds": round(latency, 2),
                },
            )
        )
        state.add_trace_event(
            "critic_llm",
            {
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return state

    @staticmethod
    def _deterministic_checks(state: ResearchState) -> dict[str, object]:
        """Đếm citation trong final_answer:
        - nguồn nào bị bỏ sót (missing_sources)?
        - có trích dẫn ngoài khoảng [1..n] không (dấu hiệu bịa nguồn)?
        """
        answer = state.final_answer or ""
        n_sources = len(state.sources)
        cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)} if n_sources else set()
        missing = sorted(set(range(1, n_sources + 1)) - cited)
        out_of_range = sorted(i for i in cited if i < 1 or i > n_sources)
        return {
            "n_sources": n_sources,
            "cited_indices": sorted(cited),
            "missing_sources": missing,
            "out_of_range_citations": out_of_range,
            "coverage": round(len(cited) / n_sources, 3) if n_sources else None,
        }

    @staticmethod
    def _format_findings(checks: dict[str, object], llm_review: str) -> str:
        lines = [
            "Critic findings:",
            f"- deterministic checks: {checks}",
            "- LLM review: " + llm_review.strip(),
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_prompt(state: ResearchState) -> str:
        lines = [
            f"Question: {state.request.query}",
            "",
            "Final answer:",
            state.final_answer or "",
            "",
            "Research notes:",
            state.research_notes or "(none)",
            "",
            "Sources:",
        ]
        for i, doc in enumerate(state.sources, start=1):
            lines.append(f"[{i}] {doc.title}")
        return "\n".join(lines)
