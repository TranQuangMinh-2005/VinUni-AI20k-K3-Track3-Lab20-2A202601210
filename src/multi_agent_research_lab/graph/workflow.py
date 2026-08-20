"""LangGraph workflow: nối các agent thành đồ thị có vòng lặp kiểm soát.

Sơ đồ luồng (giống sơ đồ trong codelab):

    START → supervisor ──conditional edge──> researcher / analyst / writer
                 ↑                                  │
                 └──────── quay lại kiểm tra ───────┘

    writer (hoặc route "done") → END

Mỗi lần worker xong việc, luồng quay về supervisor để kiểm tra lại state.
Vòng lặp này an toàn vì supervisor có guardrail max_iterations.

BONUS — bật CriticAgent (đã implement trong agents/critic.py) trong graph:
    # thêm vào build():
    #   builder.add_node("critic", self._critic_node)
    #   builder.add_edge("writer", "critic")
    #   builder.add_edge("critic", "supervisor")
    # và thêm method:
    #   def _critic_node(self, state: ResearchState) -> dict[str, Any]:
    #       return self._run_agent(self._agents["critic"], state)
    # (đồng thời thêm CriticAgent vào self._agents)
"""

import logging
from collections.abc import Hashable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMCompleter
from multi_agent_research_lab.services.search_client import SearchClient, SearchProvider

logger = logging.getLogger(__name__)

# Ánh xạ route của supervisor → node đích.
# "done" trỏ về END = dừng graph. (END là hằng số đặc biệt của LangGraph)
_ROUTE_TARGETS: dict[Hashable, str] = {
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
    "done": END,
}


class MultiAgentWorkflow:
    """Dựng và chạy graph multi-agent.

    Orchestration (ai chạy khi nào, vòng lặp ra sao) nằm ở đây;
    logic nội dung của từng agent vẫn nằm trong `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMCompleter | None = None,
        search_client: SearchProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        # Cho phép inject llm_client/search_client để viết test end-to-end với Fake.
        # Khi không inject → dùng client thật (đọc API key từ .env).
        llm = llm_client or LLMClient(settings=self._settings)
        search = search_client or SearchClient(settings=self._settings)

        # Tất cả agent dùng CHUNG một LLMClient (stateless nên an toàn);
        # chỉ Researcher cần thêm SearchClient.
        self._agents: dict[str, BaseAgent] = {
            "supervisor": SupervisorAgent(settings=self._settings, llm_client=llm),
            "researcher": ResearcherAgent(
                settings=self._settings, llm_client=llm, search_client=search
            ),
            "analyst": AnalystAgent(settings=self._settings, llm_client=llm),
            "writer": WriterAgent(settings=self._settings, llm_client=llm),
        }
        self._graph = self.build()

    def build(self) -> CompiledStateGraph[ResearchState, Any, ResearchState, ResearchState]:
        """Dựng graph: thêm node, conditional edge, và stop condition."""
        builder = StateGraph(ResearchState)

        # Mỗi agent là một node. Tất cả node đều đi qua `_run_agent`
        # để xử lý lỗi + chuyển state thành dict cho LangGraph merge.
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("analyst", self._analyst_node)
        builder.add_node("writer", self._writer_node)

        builder.add_edge(START, "supervisor")

        # Supervisor là "trạm trung chuyển": tùy state.next_agent mà rẽ nhánh.
        # Bản đồ rẽ nhánh lấy từ _ROUTE_TARGETS (done → END → dừng graph).
        builder.add_conditional_edges(
            "supervisor",
            self._route_from_state,
            _ROUTE_TARGETS,
        )

        # Worker xong việc → quay về supervisor kiểm tra lại state.
        # Vòng lặp này bị chặn bởi guardrail max_iterations trong SupervisorAgent.
        # Writer cũng quay về để supervisor xác nhận "done" (ghi vào route_history)
        # — nếu writer fail, supervisor sẽ thấy final_answer còn thiếu và cho retry.
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Chạy graph với state đầu vào, trả state cuối cùng.

        `recursion_limit` đặt cao hơn số bước tối đa để guardrail max_iterations
        của supervisor kịp dừng graph một cách "sạch sẽ" trước khi LangGraph tự
        chặn đệ quy (lỗi đó khó đọc hơn).
        """
        recursion_limit = self._settings.max_iterations * 2 + 10
        result = self._graph.invoke(state, config={"recursion_limit": recursion_limit})
        # invoke trả dict → validate lại thành ResearchState để có object đầy đủ
        return ResearchState.model_validate(result)

    # ---------- Node functions ----------
    # Mỗi method là một node của graph. Chúng chỉ làm việc "đóng gói":
    # gọi agent tương ứng qua `_run_agent` (xử lý lỗi + đổi state thành dict).

    def _supervisor_node(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self._agents["supervisor"], state)

    def _researcher_node(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self._agents["researcher"], state)

    def _analyst_node(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self._agents["analyst"], state)

    def _writer_node(self, state: ResearchState) -> dict[str, Any]:
        return self._run_agent(self._agents["writer"], state)

    @staticmethod
    def _route_from_state(state: ResearchState) -> str:
        """Đọc quyết định mới nhất của supervisor từ state.

        next_agent là None (trường hợp bất thường) → mặc định "done" để graph không treo.
        """
        return state.next_agent or "done"

    def _run_agent(self, agent: BaseAgent, state: ResearchState) -> dict[str, Any]:
        """Chạy một agent và trả state dưới dạng dict cho LangGraph merge.

        Hai việc chính:
        1. Trả về dict đầy đủ (model_dump) để LangGraph merge vào state kế tiếp
           (LangGraph không giữ mutation trên object state cũ).
        2. Bắt lỗi khi agent fail → ghi vào state.errors rồi để supervisor quyết định
           tiếp (fallback), thay vì để cả graph crash.
        """
        try:
            updated = agent.run(state)
        except StudentTodoError:
            # Phần chưa implement (Bước 3) → ném tiếp để CLI hiển thị panel "Expected TODO"
            raise
        except Exception as exc:
            # Guardrail fallback: agent fail không làm chết graph —
            # lỗi được ghi lại, supervisor sẽ thấy state còn thiếu và xử lý
            # (hoặc dừng khi hết max_iterations)
            state.errors.append(f"{agent.name} failed: {exc}")
            state.add_trace_event("agent_error", {"agent": agent.name, "error": str(exc)})
            logger.exception("%s failed, recorded error and continuing", agent.name)
            return state.model_dump()
        return updated.model_dump()
