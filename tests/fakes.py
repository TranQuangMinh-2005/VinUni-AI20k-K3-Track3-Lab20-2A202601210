"""Các đối tượng giả dùng chung cho unit test (không cần API key / gọi mạng).

- FakeLLM: trả nội dung định sẵn (hoặc ném lỗi) — test từng agent.
- ScriptedLLM: trả nội dung kế tiếp trong danh sách — test pipeline nhiều lần gọi.
- FakeSearch: trả nguồn cố định (hoặc ném lỗi) — test researcher.
"""

from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClientError, LLMResponse


class FakeLLM:
    """LLM giả: trả nội dung định sẵn hoặc ném lỗi theo kịch bản test."""

    def __init__(self, content: str = "", error: LLMClientError | None = None) -> None:
        self.content = content
        self.error = error
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMResponse(
            content=self.content,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            model="fake-model",
        )


class ScriptedLLM:
    """LLM giả chạy theo kịch bản: mỗi lần gọi trả nội dung kế tiếp trong danh sách.

    Hết kịch bản → trả "done" để graph không treo.
    """

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.calls += 1
        content = (
            self.contents.pop(0)
            if self.contents
            else '{"next": "done", "reason": "script exhausted"}'
        )
        return LLMResponse(content=content, input_tokens=1, output_tokens=1, model="fake")


class FakeSearch:
    """Search giả: trả danh sách nguồn cố định, hoặc ném lỗi theo kịch bản."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if self.error is not None:
            raise self.error
        return [
            SourceDocument(title="Source A", url="https://a.example", snippet="About A."),
            SourceDocument(title="Source B", url="https://b.example", snippet="About B."),
        ]
