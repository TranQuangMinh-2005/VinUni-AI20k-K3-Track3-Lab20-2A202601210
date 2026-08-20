"""Tracing hooks — tích hợp LangSmith.

Nguyên tắc: code không hard-bind một provider duy nhất. Chỗ này là điểm plug-in
cho LangSmith / Langfuse / OpenTelemetry / JSON trace local.

Cách hoạt động:
- Có LANGSMITH_API_KEY trong .env → mọi hàm được gắn `@traceable` tự đẩy trace
  lên LangSmith (mở dashboard smith.langchain.com để xem từng step).
- Không có key → LangSmith tự no-op, hệ thống vẫn chạy bình thường (trace local
  vẫn được ghi vào state.trace như trước).
"""

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, ParamSpec, TypeVar, cast

from langsmith import traceable

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def is_tracing_enabled(settings: Settings | None = None) -> bool:
    """True nếu có LangSmith key trong .env/environment."""
    return bool((settings or get_settings()).langsmith_api_key)


def setup_tracing(settings: Settings | None = None) -> None:
    """Thiết lập biến môi trường cho LangSmith tracing.

    Được gọi 1 lần lúc khởi động CLI. Đặt:
    - LANGSMITH_TRACING=true → bật auto-tracing cho LangGraph/LangChain runs
    - LANGSMITH_PROJECT → tên project hiện trên dashboard
    Không có key → bỏ qua (không crash).
    """
    resolved = settings or get_settings()
    if not is_tracing_enabled(resolved):
        logger.info("LangSmith tracing disabled (no LANGSMITH_API_KEY)")
        return
    # Quan trọng: pydantic-settings chỉ ĐỌC .env, không export ra os.environ —
    # mà LangSmith SDK chỉ nhìn os.environ. Phải set lại ở đây.
    os.environ.setdefault("LANGSMITH_API_KEY", resolved.langsmith_api_key or "")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", resolved.langsmith_project)
    logger.info("LangSmith tracing enabled, project=%s", resolved.langsmith_project)


def traced(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator đẩy một hàm lên LangSmith như một trace/span.

    Dùng cho các hàm quan trọng (ví dụ LLMClient.complete, agent.run):
    mỗi lần gọi sẽ là một node riêng trên trace UI kèm input/output/latency.
    Khi không có key, `traceable` tự no-op nên code không cần if/else.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # traceable trả SupportsLangsmithExtra (hàm có thêm kwarg langsmith_extra)
        # nhưng callable tương đương → cast để giữ chữ ký gọn cho callers
        return cast(Callable[P, R], traceable(run_type="chain", name=name)(func))

    return decorator


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context tối giản (không phụ thuộc provider).

    Đo thời gian chạy một khối code + gắn attributes. Dùng cho các phần
    không phải hàm LLM (ví dụ: search, parse). Kết quả nằm trong state.trace
    của hệ thống; nếu muốn lên LangSmith, dùng `traced()` cho hàm thay vì span này.
    """
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = round(perf_counter() - started, 3)
