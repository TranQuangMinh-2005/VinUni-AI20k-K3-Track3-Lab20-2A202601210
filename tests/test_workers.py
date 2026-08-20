"""Unit tests cho 3 worker agent (Researcher, Analyst, Writer)."""

import pytest
from fakes import FakeLLM, FakeSearch

from multi_agent_research_lab.agents import AnalystAgent, CriticAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClientError
from multi_agent_research_lab.services.search_client import SearchClientError


def _settings() -> Settings:
    return Settings(MAX_ITERATIONS=6)


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="What is GraphRAG?"))


# ---------- Researcher ----------

def test_researcher_populates_sources_and_notes() -> None:
    llm = FakeLLM(content="- GraphRAG builds knowledge graphs [1].\n- It improves RAG [2].")
    state = _state()
    ResearcherAgent(settings=_settings(), llm_client=llm, search_client=FakeSearch()).run(state)
    assert len(state.sources) == 2
    assert "[1]" in (state.research_notes or "")
    assert state.trace[0]["name"] == "researcher_llm"


def test_researcher_falls_back_to_mock_sources_on_search_error() -> None:
    # Search chết → researcher vẫn sống: dùng nguồn mock + ghi lỗi vào state
    llm = FakeLLM(content="notes")
    state = _state()
    agent = ResearcherAgent(
        settings=_settings(),
        llm_client=llm,
        search_client=FakeSearch(error=SearchClientError("search api down")),
    )
    agent.run(state)
    assert state.sources and all(doc.metadata.get("mock") for doc in state.sources)
    assert any("search failed" in err for err in state.errors)


def test_researcher_is_idempotent() -> None:
    # Đã có sources + notes → không gọi lại LLM (tránh làm việc trùng lặp)
    llm = FakeLLM(content="should not be called")
    state = _state()
    state.sources = FakeSearch().search("x")
    state.research_notes = "already done"
    ResearcherAgent(settings=_settings(), llm_client=llm, search_client=FakeSearch()).run(state)
    assert llm.calls == 0


def test_researcher_raises_when_llm_dead() -> None:
    # LLM chết hoàn toàn → ném AgentExecutionError để workflow/supervisor xử lý retry
    llm = FakeLLM(error=LLMClientError("provider down"))
    state = _state()
    agent = ResearcherAgent(settings=_settings(), llm_client=llm, search_client=FakeSearch())
    with pytest.raises(AgentExecutionError):
        agent.run(state)


# ---------- Analyst ----------

def test_analyst_populates_analysis_notes() -> None:
    llm = FakeLLM(content="Key claims: ...\nWeak evidence: ...")
    state = _state()
    state.sources = FakeSearch().search("x")
    state.research_notes = "notes [1]"
    AnalystAgent(settings=_settings(), llm_client=llm).run(state)
    assert state.analysis_notes == "Key claims: ...\nWeak evidence: ..."
    assert state.trace[0]["name"] == "analyst_llm"


def test_analyst_is_idempotent() -> None:
    llm = FakeLLM(content="should not be called")
    state = _state()
    state.analysis_notes = "already done"
    AnalystAgent(settings=_settings(), llm_client=llm).run(state)
    assert llm.calls == 0


# ---------- Writer ----------

def test_writer_populates_final_answer() -> None:
    llm = FakeLLM(content="GraphRAG is ... [1][2]")
    state = _state()
    state.sources = FakeSearch().search("x")
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    WriterAgent(settings=_settings(), llm_client=llm).run(state)
    assert state.final_answer == "GraphRAG is ... [1][2]"
    assert state.trace[0]["name"] == "writer_llm"


def test_writer_is_idempotent() -> None:
    llm = FakeLLM(content="should not be called")
    state = _state()
    state.final_answer = "already done"
    WriterAgent(settings=_settings(), llm_client=llm).run(state)
    assert llm.calls == 0


# ---------- Critic (bonus) ----------

def test_critic_flags_missing_and_out_of_range_citations() -> None:
    # [2] bị bỏ sót, [5] vượt ngoài 2 nguồn → critic phải chỉ ra cả hai
    llm = FakeLLM(content="All claims cited.")
    state = _state()
    state.sources = FakeSearch().search("x")
    state.research_notes = "notes"
    state.analysis_notes = "analysis"
    state.final_answer = "GraphRAG is good [1]. See also [5]."
    CriticAgent(settings=_settings(), llm_client=llm).run(state)
    checks = state.trace[0]["payload"]
    assert checks["missing_sources"] == [2]
    assert checks["out_of_range_citations"] == [5]
    assert any(result.agent == AgentName.CRITIC for result in state.agent_results)


def test_critic_skips_when_no_final_answer() -> None:
    llm = FakeLLM(content="should not be called")
    state = _state()
    CriticAgent(settings=_settings(), llm_client=llm).run(state)
    assert llm.calls == 0
