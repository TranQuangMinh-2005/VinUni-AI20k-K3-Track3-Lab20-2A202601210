"""Unit tests cho SupervisorAgent (LLM-based routing + guardrail + fallback).

Dùng FakeLLM (trong tests/fakes.py) để test không cần gọi API thật.
"""

from fakes import FakeLLM

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClientError


def _settings(max_iterations: int = 6) -> Settings:
    """Settings tường minh để test không phụ thuộc vào giá trị trong .env."""
    return Settings(MAX_ITERATIONS=max_iterations)


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain GraphRAG state of the art"))


def _source() -> SourceDocument:
    return SourceDocument(title="GraphRAG survey", url="https://example.com", snippet="...")


def test_guardrail_stops_without_calling_llm() -> None:
    # Hết ngân sách iteration → dừng ngay, KHÔNG tốn token gọi LLM
    fake = FakeLLM(content='{"next": "researcher", "reason": "x"}')
    state = _state()
    state.iteration = 6
    SupervisorAgent(settings=_settings(max_iterations=6), llm_client=fake).run(state)
    assert fake.calls == 0
    assert state.next_agent == "done"
    assert any("max_iterations" in err for err in state.errors)


def test_routes_based_on_llm_decision() -> None:
    # LLM nói "analyst" → supervisor nghe theo
    fake = FakeLLM(content='{"next": "analyst", "reason": "research done"}')
    state = _state()
    state.sources = [_source()]
    state.research_notes = "notes"
    SupervisorAgent(settings=_settings(), llm_client=fake).run(state)
    assert state.next_agent == "analyst"
    assert state.route_history == ["analyst"]


def test_llm_call_and_route_recorded_in_trace() -> None:
    fake = FakeLLM(content='{"next": "researcher", "reason": "no sources yet"}')
    state = _state()
    SupervisorAgent(settings=_settings(), llm_client=fake).run(state)
    assert state.next_agent == "researcher"
    # Trace phải có: 1) lần gọi LLM (kèm token/cost) 2) quyết định route (kèm lý do)
    assert state.trace[0]["name"] == "supervisor_llm"
    assert state.trace[0]["payload"]["input_tokens"] == 10
    assert state.trace[-1]["name"] == "route"
    assert state.trace[-1]["payload"]["reason"] == "no sources yet"


def test_accepts_json_inside_code_fence() -> None:
    # LLM thường bọc JSON trong ```json ... ``` → phải parse được
    fake = FakeLLM(content='```json\n{"next": "writer", "reason": "analysis ready"}\n```')
    state = _state()
    state.sources = [_source()]
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    SupervisorAgent(settings=_settings(), llm_client=fake).run(state)
    assert state.next_agent == "writer"


def test_falls_back_to_rules_on_invalid_output() -> None:
    # LLM trả text hỏng → fallback rule-based: thiếu sources → researcher
    fake = FakeLLM(content="I'm not sure what to do...")
    state = _state()
    SupervisorAgent(settings=_settings(), llm_client=fake).run(state)
    assert state.next_agent == "researcher"
    assert "fallback" in state.trace[-1]["payload"]["reason"]


def test_falls_back_to_rules_on_llm_error() -> None:
    # LLM chết hoàn toàn (hết retry) → ghi lỗi + fallback, hệ thống không crash
    fake = FakeLLM(error=LLMClientError("provider down"))
    state = _state()
    SupervisorAgent(settings=_settings(), llm_client=fake).run(state)
    assert state.next_agent == "researcher"
    assert any("supervisor LLM call failed" in err for err in state.errors)


def test_rejects_route_outside_allowed_choices() -> None:
    # LLM bịa route lạ → bị chặn, rơi về fallback
    fake = FakeLLM(content='{"next": "hacker", "reason": "muahaha"}')
    state = _state()
    SupervisorAgent(settings=_settings(), llm_client=fake).run(state)
    assert state.next_agent in {"researcher", "analyst", "writer", "done"}
