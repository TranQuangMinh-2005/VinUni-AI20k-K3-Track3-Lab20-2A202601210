"""Supervisor / router: agent LLM điều phối worker tiếp theo."""

import json
import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import (
    LLMClient,
    LLMClientError,
    LLMCompleter,
    LLMResponse,
)
from multi_agent_research_lab.utils.timer import elapsed_timer

logger = logging.getLogger(__name__)

# Các lựa chọn route hợp lệ mà supervisor được phép chọn
_AGENT_CHOICES = ("researcher", "analyst", "writer", "done")

# Prompt vai trò: mô tả nhiệm vụ điều phối + định dạng output bắt buộc (JSON).
# Viết tiếng Anh vì LLM tuân thủ format JSON chặt chẽ hơn khi prompt tiếng Anh.
_SYSTEM_PROMPT = """\
You are the supervisor (router) of a multi-agent research system.
Your ONLY job is to decide which worker runs next, based on the current state.

Available workers:
- researcher: searches for sources and writes research notes
- analyst: reads research notes and produces analysis notes
- writer: writes the final answer with citations
- done: stop the workflow

Routing rules:
- If sources/research_notes are missing -> "researcher"
- Else if analysis_notes is missing -> "analyst"
- Else if final_answer is missing -> "writer"
- Else -> "done"

Respond ONLY with a JSON object in this exact shape:
{"next": "<researcher|analyst|writer|done>", "reason": "<one short sentence>"}
Do not output anything else.
"""


class SupervisorAgent(BaseAgent):
    """Điều phối viên dùng LLM: nhìn state hiện tại → quyết định route tiếp theo.

    Thiết kế 3 lớp:
    1. Guardrail: vượt `max_iterations` → dừng ngay, KHÔNG gọi LLM (tiết kiệm token).
    2. Chính: gọi LLM với prompt định sẵn, yêu cầu trả JSON `{"next", "reason"}`.
    3. Fallback: LLM lỗi hoặc trả JSON không hợp lệ → rơi về rule-based
       (luật cứng theo thứ tự pipeline) để hệ thống không bao giờ chết.
    """

    name = "supervisor"

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMCompleter | None = None,
    ) -> None:
        """Nhận Settings + LLMCompleter (inject được để test với LLM giả)."""
        self._settings = settings or get_settings()
        self._llm = llm_client

    def _ensure_llm(self) -> LLMCompleter:
        """Tạo LLMClient thật nếu chưa được inject (lazy để test không bắt buộc API key)."""
        if self._llm is None:
            self._llm = LLMClient(settings=self._settings)
        return self._llm

    def run(self, state: ResearchState) -> ResearchState:
        """Quyết định route tiếp theo và ghi vào `state.next_agent` + `route_history`."""
        # Guardrail: hết ngân sách iteration → dừng ngay, không tốn thêm token
        if state.iteration >= self._settings.max_iterations:
            if state.final_answer is None:
                state.errors.append(
                    f"Stopped by guardrail: max_iterations={self._settings.max_iterations} "
                    "reached before final_answer was produced"
                )
            return self._route(state, "done", "max_iterations reached")

        # Bước chính: hỏi LLM quyết định route
        decision = self._ask_llm(state)

        # Fallback nếu LLM không trả được quyết định hợp lệ
        if decision is None:
            fallback = self._fallback_decision(state)
            decision = fallback
            reason = f"{fallback['reason']} (rule-based fallback)"
        else:
            reason = decision["reason"]

        return self._route(state, decision["next"], reason)

    def _ask_llm(self, state: ResearchState) -> dict[str, str] | None:
        """Gọi LLM để xin quyết định route. Trả None nếu LLM lỗi hoặc output không hợp lệ."""
        llm = self._ensure_llm()

        try:
            with elapsed_timer() as elapsed:  # đo latency cho trace/benchmark
                response: LLMResponse = llm.complete(_SYSTEM_PROMPT, self._state_summary(state))
            latency = elapsed()

            # Ghi token/cost/latency của lần gọi này vào trace → phục vụ benchmark Bước 4
            state.add_trace_event(
                "supervisor_llm",
                {
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_seconds": round(latency, 2),
                },
            )
        except LLMClientError as exc:
            # LLM chết (hết retry) → ghi lỗi, để run() rơi về fallback rule-based
            state.errors.append(f"supervisor LLM call failed: {exc}")
            state.add_trace_event("supervisor_llm_error", {"error": str(exc)})
            return None

        decision = self._parse_decision(response.content)
        if decision is None:
            # Giữ lại 200 ký tự đầu của output hỏng để đọc trace biết LLM trả gì
            state.add_trace_event("supervisor_llm_invalid", {"raw_content": response.content[:200]})
        return decision

    @staticmethod
    def _parse_decision(content: str) -> dict[str, str] | None:
        """Parse JSON từ output LLM. Chấp nhận cả dạng có code fence ```json ... ```.

        Trả None nếu không parse được hoặc `next` không nằm trong danh sách hợp lệ
        (chống "prompt injection" ngầm: LLM tự bịa route lạ cũng bị chặn).
        """
        text = content.strip()
        # Bóc code fence nếu model bọc JSON trong ```json ... ```
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Thử tìm block JSON đầu tiên nằm trong cặp ngoặc nhọn
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        if isinstance(data, dict) and data.get("next") in _AGENT_CHOICES:
            return {
                "next": str(data["next"]),
                "reason": str(data.get("reason", "")).strip(),
            }
        return None

    def _fallback_decision(self, state: ResearchState) -> dict[str, str]:
        """Luật cứng dự phòng theo đúng thứ tự pipeline research → analysis → writing.

        Chỉ dùng khi LLM không trả được quyết định hợp lệ — đảm bảo hệ thống luôn
        có đường đi tiếp (guardrail "fallback" trong rubric).
        """
        if not state.sources or not state.research_notes:
            return {"next": "researcher", "reason": "missing sources/research_notes"}
        if not state.analysis_notes:
            return {"next": "analyst", "reason": "missing analysis_notes"}
        if not state.final_answer:
            return {"next": "writer", "reason": "ready to write final answer"}
        return {"next": "done", "reason": "final_answer already produced"}

    def _state_summary(self, state: ResearchState) -> str:
        """Tóm tắt state thành văn bản ngắn gọn cho LLM đọc.

        Chỉ đưa trạng thái present/missing + lịch sử route, KHÔNG nhồi toàn bộ
        nội dung notes vào prompt (đỡ tốn token, LLM quyết định nhanh hơn).
        """
        lines = [
            f"query: {state.request.query}",
            f"sources: {'present' if state.sources else 'missing'} ({len(state.sources)} items)",
            f"research_notes: {'present' if state.research_notes else 'missing'}",
            f"analysis_notes: {'present' if state.analysis_notes else 'missing'}",
            f"final_answer: {'present' if state.final_answer else 'missing'}",
            f"iteration: {state.iteration}/{self._settings.max_iterations}",
            "route_history: "
            + (" -> ".join(state.route_history) if state.route_history else "(empty)"),
        ]
        if state.errors:
            lines.append("recent_errors: " + "; ".join(state.errors[-2:]))
        return "\n".join(lines)

    def _route(self, state: ResearchState, route: str, reason: str) -> ResearchState:
        """Ghi quyết định route vào state + trace, kèm lý do để debug dễ dàng."""
        state.next_agent = route
        state.record_route(route)  # thêm vào route_history + tăng iteration
        state.add_trace_event(
            "route", {"next": route, "reason": reason, "iteration": state.iteration}
        )
        logger.info("supervisor route=%s reason=%s iteration=%s", route, reason, state.iteration)
        return state
