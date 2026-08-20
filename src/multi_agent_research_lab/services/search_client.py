"""Search client abstraction cho ResearcherAgent.

Thiết kế: SearchClient nói chuyện với Tavily API qua HTTP (httpx).
Nếu không có API key → dùng mock nguồn (rõ ràng đánh dấu) để lab vẫn chạy được.
"""

import logging
from typing import Any, Protocol

import httpx

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# Endpoint chính thức của Tavily search API
_TAVILY_ENDPOINT = "https://api.tavily.com/search"

# Ngưỡng relevance: bỏ kết quả Tavily trả score thấp (nhiễu)
_MIN_SCORE = 0.5


class SearchClientError(LabError):
    """Lỗi khi gọi search API thất bại (mạng, quota, key sai...)."""


class SearchProvider(Protocol):
    """Giao diện tối thiểu ResearcherAgent cần từ search client (dễ mock trong test)."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]: ...


class SearchClient:
    """Client tìm kiếm qua Tavily, có filter + dedupe kết quả."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Tìm tài liệu liên quan query, trả tối đa `max_results` kết quả đã lọc.

        Raises:
            SearchClientError: khi không có API key hoặc gọi API thất bại.
            (ResearcherAgent sẽ bắt lỗi này để fallback sang mock nguồn.)
        """
        if not self._settings.tavily_api_key:
            raise SearchClientError(
                "TAVILY_API_KEY is not set. Add it to .env or use mock search."
            )

        try:
            # httpx dùng certifi nên không bị lỗi SSL trên macOS như urllib
            response = httpx.post(
                _TAVILY_ENDPOINT,
                json={
                    "api_key": self._settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # httpx.HTTPError: lỗi mạng/timeout/HTTP; ValueError: response không phải JSON
            raise SearchClientError(f"Tavily search failed: {exc}") from exc

        docs = self._to_documents(data, max_results)
        logger.info("search returned %d docs for query %.60s", len(docs), query)
        return docs

    @staticmethod
    def _to_documents(data: dict[str, Any], max_results: int) -> list[SourceDocument]:
        """Chuyển response JSON của Tavily → SourceDocument đã lọc + dedupe."""
        docs: list[SourceDocument] = []
        seen_urls: set[str] = set()

        for item in data.get("results", []):
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            score = float(item.get("score", 0.0))

            # Lọc nhiễu: bỏ kết quả trống, trùng URL, hoặc score quá thấp
            if not title or not url or url in seen_urls or score < _MIN_SCORE:
                continue

            seen_urls.add(url)
            # Cắt snippet ngắn để đỡ tốn token khi đưa vào prompt researcher
            docs.append(
                SourceDocument(
                    title=title,
                    url=url,
                    snippet=content[:500],
                    metadata={"score": round(score, 3)},
                )
            )
            if len(docs) >= max_results:
                break

        return docs

    @staticmethod
    def mock_search(query: str, max_results: int = 5) -> list[SourceDocument]:
        """Nguồn mock tĩnh dùng khi không có Tavily key hoặc API chết.

        Đánh dấu `mock=True` trong metadata để researcher/analyst biết đây
        không phải kết quả web thật (phục vụ giải thích benchmark sau).
        """
        mock_sources = [
            SourceDocument(
                title="Anthropic: Building effective agents",
                url="https://www.anthropic.com/engineering/building-effective-agents",
                snippet=(
                    "Workflows vs agents: predefined code paths vs LLM-driven dynamic control. "
                    "Start simple, add complexity only when justified."
                ),
                metadata={"mock": True},
            ),
            SourceDocument(
                title="LangGraph documentation: multi-agent systems",
                url="https://langchain-ai.github.io/langgraph/concepts/multi_agent/",
                snippet=(
                    "LangGraph supports multi-agent architectures: network, supervisor, "
                    "hierarchical, and custom handoff patterns."
                ),
                metadata={"mock": True},
            ),
            SourceDocument(
                title="GraphRAG: From Local to Global (Microsoft)",
                url="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/",
                snippet=(
                    "GraphRAG uses LLM-derived knowledge graphs to improve retrieval-augmented "
                    "generation over private corpora."
                ),
                metadata={"mock": True},
            ),
        ]
        return mock_sources[: max(1, max_results)]
