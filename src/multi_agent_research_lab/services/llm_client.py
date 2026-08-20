"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.

Thiết kế: mọi agent chỉ nói chuyện với `LLMClient` (interface chung), không import thẳng
SDK của provider. Muốn đổi sang Azure/Anthropic thì chỉ cần sửa mỗi file này.
"""

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.observability.tracing import traced

logger = logging.getLogger(__name__)

# Bảng giá ước tính (USD / 1 triệu token) theo model: (giá input, giá output).
# Dùng để ước lượng chi phí cho benchmark — số làm tròn, không phải giá provider chính xác.
_PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}

# Các loại lỗi "tạm thời" mà retry có thể cứu được:
# - APIError / RateLimitError: server quá tải hoặc vượt rate limit → thử lại sau vài giây
# - APITimeoutError / APIConnectionError: mạng chập chờn → thử lại
# KHÔNG retry lỗi 401 (sai key) hay 400 (prompt sai) vì thử lại cũng vô ích.
_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    APIError,
    APITimeoutError,
    RateLimitError,
    APIConnectionError,
)


@dataclass(frozen=True)
class LLMResponse:
    """Kết quả chuẩn hóa mà mọi agent nhận được (không phụ thuộc provider)."""

    content: str  # nội dung văn bản model sinh ra
    input_tokens: int | None = None  # số token ta gửi lên (để tính chi phí)
    output_tokens: int | None = None  # số token model sinh ra
    cost_usd: float | None = None  # chi phí ước tính bằng USD
    model: str | None = None  # tên model đã dùng (để ghi trace)


class LLMClientError(LabError):
    """Lỗi domain khi gọi LLM thất bại sau khi đã retry hết số lần cho phép."""


class LLMCompleter(Protocol):
    """Giao diện tối thiểu mà agent cần từ LLMClient.

    Tách Protocol này ra để unit test có thể inject "LLM giả" (FakeLLM)
    mà không phải dựng cả LLMClient thật (đòi API key + gọi mạng).
    """

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


class LLMClient:
    """Client gọi OpenAI với retry + timeout + đếm token.

    Các guardrail (retry, timeout, log token) nằm ở đây để agent không phải tự lo.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Khởi tạo client từ Settings (đọc .env).

        Cho phép inject `settings` để dễ viết unit test (không phụ thuộc .env thật).
        """
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            # Chặn sớm với thông báo rõ ràng thay vì để API trả lỗi 401 mơ hồ
            raise LLMClientError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        # SDK openai>=1.x: tạo client mỗi app một lần, tự lấy key từ constructor
        self._client = OpenAI(api_key=self._settings.openai_api_key)

    # Gắn trace LangSmith: mỗi lần gọi LLM thành 1 span riêng trên dashboard
    # (input/output/latency/token tự được ghi lại). Không có key → tự no-op.
    @traced("llm_complete")
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Gọi chat completion và chuẩn hóa kết quả thành `LLMResponse`.

        Args:
            system_prompt: hướng dẫn vai trò cho model (ví dụ "You are a researcher...").
            user_prompt: câu hỏi / dữ liệu cần model xử lý.

        Returns:
            LLMResponse gồm nội dung + token usage + chi phí ước tính.
        """
        try:
            completion = self._call_with_retry(system_prompt, user_prompt)
        except _RETRYABLE_EXCEPTIONS as exc:
            # Chỉ rơi vào đây khi 3 lần retry đều thất bại:
            # chuyển lỗi SDK thành lỗi domain của mình để caller xử lý thống nhất
            raise LLMClientError(f"LLM call failed after retries: {exc}") from exc

        # Lấy nội dung + usage từ response.
        # `usage` có thể là None nếu API không trả về (hoặc đang dùng stream) → xử lý an toàn.
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        cost_usd = self._estimate_cost(input_tokens, output_tokens)

        # Log lại để debug và có dữ liệu cho tracing/benchmark
        logger.info(
            "llm complete model=%s in=%s out=%s cost=%.6f",
            self._settings.openai_model,
            input_tokens,
            output_tokens,
            cost_usd or 0.0,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=self._settings.openai_model,
        )

    @retry(
        # Chỉ retry khi gặp lỗi tạm thời (xem danh sách _RETRYABLE_EXCEPTIONS ở trên)
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),  # tối đa 3 lần thử (1 lần gốc + 2 retry)
        wait=wait_exponential(multiplier=1, min=1, max=10),  # chờ 1s → 2s → 4s giữa các lần
        reraise=True,  # hết số lần vẫn fail thì ném lại lỗi gốc cho `complete()` bắt
    )
    def _call_with_retry(self, system_prompt: str, user_prompt: str) -> Any:
        """Thực hiện một lần gọi API. Decorator `@retry` tự gọi lại khi gặp lỗi tạm thời."""
        return self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Guardrail: timeout cứng (từ TIMEOUT_SECONDS trong .env) để call không treo vô hạn
            timeout=self._settings.timeout_seconds,
        )

    def _estimate_cost(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        """Ước lượng chi phí từ bảng giá cứng.

        Model không có trong bảng giá → trả None (không bịa số) để benchmark xử lý riêng.
        """
        if input_tokens is None or output_tokens is None:
            return None
        prices = _PRICES_PER_MILLION.get(self._settings.openai_model)
        if prices is None:
            return None
        input_price, output_price = prices
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
