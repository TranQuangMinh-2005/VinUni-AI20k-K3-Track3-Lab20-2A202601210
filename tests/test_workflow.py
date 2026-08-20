"""End-to-end test cho workflow multi-agent (dùng LLM/search giả).

Xác nhận tiêu chí Bước 2: route_history đúng thứ tự
researcher → analyst → writer → done, không còn StudentTodoError.
"""

from fakes import FakeSearch, ScriptedLLM

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClientError, LLMResponse


def test_workflow_runs_full_pipeline_in_order() -> None:
    # Kịch bản 7 lần gọi LLM:
    # supervisor(researcher) → researcher → supervisor(analyst) → analyst
    # → supervisor(writer) → writer → supervisor(done)
    llm = ScriptedLLM(
        [
            '{"next": "researcher", "reason": "no sources yet"}',
            "GraphRAG uses knowledge graphs [1]. It helps multi-hop queries [2].",
            '{"next": "analyst", "reason": "notes ready"}',
            "Key claims: ... Evidence: strong.",
            '{"next": "writer", "reason": "analysis ready"}',
            "GraphRAG is a graph-based RAG approach [1][2].",
            '{"next": "done", "reason": "answer complete"}',
        ]
    )
    workflow = MultiAgentWorkflow(
        settings=Settings(MAX_ITERATIONS=10), llm_client=llm, search_client=FakeSearch()
    )
    state = workflow.run(ResearchState(request=ResearchQuery(query="What is GraphRAG?")))

    # Tiêu chí chính: thứ tự route đúng pipeline
    assert state.route_history == ["researcher", "analyst", "writer", "done"]
    assert state.final_answer == "GraphRAG is a graph-based RAG approach [1][2]."
    assert state.errors == []
    assert llm.calls == 7


class _AnalystAlwaysDown:
    """LLM giả: supervisor luôn route analyst, còn analyst luôn fail → vòng lặp vô hạn."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        if "sharp analyst" in system_prompt:
            raise LLMClientError("analyst is down")
        if "supervisor" in system_prompt:
            return LLMResponse(
                content='{"next": "analyst", "reason": "notes ready"}',
                input_tokens=1,
                output_tokens=1,
                model="fake",
            )
        # Researcher call (được route trước khi kịch bản hỏng bắt đầu)
        return LLMResponse(content="notes [1]", input_tokens=1, output_tokens=1, model="fake")


def test_workflow_guardrail_stops_runaway_loop() -> None:
    # Analyst fail liên tục → supervisor retry analyst mãi → guardrail max_iterations
    # phải dừng graph thay vì chạy vô hạn.
    llm = _AnalystAlwaysDown()
    workflow = MultiAgentWorkflow(
        settings=Settings(MAX_ITERATIONS=6), llm_client=llm, search_client=FakeSearch()
    )
    state = workflow.run(ResearchState(request=ResearchQuery(query="What is GraphRAG?")))

    # Guardrail đã kích hoạt: dừng với route "done" + ghi lỗi, final_answer còn thiếu
    assert state.route_history[-1] == "done"
    assert any("max_iterations" in err for err in state.errors)
    assert state.final_answer is None


def test_workflow_stream_yields_per_node_updates() -> None:
    # Stream mode: mỗi node hoàn thành là 1 update — đây là cơ chế cho UI live pipeline
    llm = ScriptedLLM(
        [
            '{"next": "researcher", "reason": "no sources yet"}',
            "GraphRAG uses knowledge graphs [1].",
            '{"next": "analyst", "reason": "notes ready"}',
            "Key claims: ...",
            '{"next": "writer", "reason": "analysis ready"}',
            "GraphRAG is a graph-based RAG approach [1].",
            '{"next": "done", "reason": "answer complete"}',
        ]
    )
    workflow = MultiAgentWorkflow(
        settings=Settings(MAX_ITERATIONS=10), llm_client=llm, search_client=FakeSearch()
    )
    initial = ResearchState(request=ResearchQuery(query="What is GraphRAG?"))
    updates = list(workflow.stream(initial))

    # Mỗi update là dict {node_name: state_snapshot}
    assert updates and all(isinstance(u, dict) for u in updates)
    node_names = [next(iter(u)) for u in updates]
    assert "supervisor" in node_names
    assert "researcher" in node_names
    assert "writer" in node_names

    # Snapshot cuối cùng chứa đủ kết quả (validate được thành ResearchState)
    final_snapshot = next(iter(updates[-1].values()))
    final_state = ResearchState.model_validate(final_snapshot)
    assert final_state.final_answer is not None
