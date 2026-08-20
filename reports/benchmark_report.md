# Benchmark Report: Single-Agent vs Multi-Agent

> Số lần chạy được gộp: 4 (mỗi query × mỗi chế độ). Model: `gpt-4o-mini`. Ngày chạy: 2026-08-20.

## Bảng so sánh

| Run | Latency (s) | Cost (USD) | Quality (0-10) | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| single-agent | 7.47 | 0.00030 | 8.0 | — | 0% | averaged over 2 run(s) |
| multi-agent | 21.13 | 0.00120 | 10.0 | 90% | 0% | averaged over 2 run(s) |

## Phân tích

- **Latency**: single-agent thắng (~7.5s vs ~21s, nhanh gấp ~2.8 lần). Multi-agent chậm vì mỗi lần chạy cần 7 lần gọi LLM (4 lần supervisor + researcher + analyst + writer) và 1 lần gọi Tavily search, trong khi baseline chỉ có 1 lần gọi.
- **Cost**: single-agent thắng (~$0.00030 vs ~$0.00120, rẻ gấp ~4 lần). Chi phí tăng đúng bằng số lần gọi thêm; phần lớn chi phí multi-agent nằm ở prompt lớn của researcher/analyst/writer (nhồi sources + notes vào context).
- **Quality (rubric heuristic 0-10)**: multi-agent thắng (10.0 vs 8.0). Khác biệt chính: citation coverage — multi-agent đạt 90% (câu trả lời trích dẫn 9/10 nguồn, tính trung bình) trong khi baseline không hề có nguồn (0%) vì single-agent không search web mà trả lời từ knowledge nội tại.
- **Failure rate**: cả hai 0% trên bộ query benchmark. Điều này không có nghĩa là hệ thống không thể fail — các guardrail (retry, fallback, max_iterations) đã được test riêng bằng unit test (xem mục failure mode).

**Trade-off cốt lõi**: multi-agent đắt hơn và chậm hơn, nhưng mua được (1) bằng chứng nguồn (citation), (2) khả năng tách bước để debug từng khâu qua trace. Baseline chỉ hợp lý khi câu hỏi không cần nguồn/kiểm chứng.

## Failure mode gặp phải & cách fix

**Failure mode 1 — Vòng lặp vô hạn Supervisor ↔ worker khi một agent chết.**
Trong quá trình test, khi bắt `AnalystAgent` luôn ném lỗi LLM (mô phỏng provider down), supervisor LLM vẫn tiếp tục route "analyst" vì `analysis_notes` còn thiếu → vòng lặp analyst → fail → supervisor → analyst... Không có guardrail thì graph sẽ chạy mãi và đốt token.
- **Cách fix**: guardrail `max_iterations` trong `SupervisorAgent.run` (dừng khi `iteration >= MAX_ITERATIONS`, ghi lỗi `"Stopped by guardrail..."` vào `state.errors`) + `recursion_limit` của LangGraph đặt cao hơn `max_iterations*2` để dừng "sạch" trước khi LangGraph tự chặn đệ quy.
- **Minh chứng**: unit test `test_workflow_guardrail_stops_runaway_loop` (tests/test_workflow.py) — route_history kết thúc bằng "done", errors chứa lý do dừng, final_answer là None.

**Failure mode 2 — Supervisor LLM trả JSON không hợp lệ.**
LLM đôi khi trả văn bản tự do hoặc route lạ (không nằm trong researcher/analyst/writer/done) dù prompt yêu cầu JSON.
- **Cách fix**: `_parse_decision` chỉ chấp nhận route trong whitelist `_AGENT_CHOICES`; parse lỏng (bóc code fence ```json, tìm block `{...}` đầu tiên). Nếu vẫn không hợp lệ hoặc LLM call fail → **fallback rule-based** theo thứ tự pipeline (research → analysis → writing), ghi event `supervisor_llm_invalid` vào trace để truy ra.
- **Minh chứng**: unit tests `test_falls_back_to_rules_on_invalid_output` và `test_rejects_route_outside_allowed_choices` (tests/test_supervisor.py).

**Failure mode 3 — Search API chết.**
Nếu Tavily lỗi (quota/network), pipeline vẫn phải chạy được.
- **Cách fix**: `ResearcherAgent._collect_sources` bắt `SearchClientError` → dùng `SearchClient.mock_search()` (nguồn được đánh dấu `mock=True`) và ghi lỗi vào state. Analyst/Writer vẫn hoạt động; benchmark ghi nhận nguồn mock để không nhầm với nguồn thật.
- **Minh chứng**: unit test `test_researcher_falls_back_to_mock_sources_on_search_error` (tests/test_workers.py).

## Trace evidence

- LangSmith project: `multi-agent-research-lab` — https://smith.langchain.com
- Mỗi lần chạy `multi-agent` hiện đủ các span: `supervisor` → `researcher`/`analyst`/`writer` → `_route_from_state`, và mỗi lần gọi LLM là một span `llm_complete` kèm latency + token.
- Lệnh chạy lại để lấy screenshot:

  ```bash
  source .venv/bin/activate
  make run-multi
  # mở https://smith.langchain.com → project multi-agent-research-lab → chọn run mới nhất
  ```

## Kết luận: khi nào dùng multi-agent?

- **Nên dùng multi-agent khi**: task cần nguồn kiểm chứng (research, fact-check), pipeline có bước tách biệt rõ (search → phân tích → viết), cần debug/trace từng khâu, hoặc muốn guardrail & rerun từng bước độc lập (retry 1 agent không phải làm lại cả task).
- **Không nên dùng khi**: câu hỏi đơn giản chỉ cần 1 lần sinh văn bản (multi-agent đắt hơn ~4x, chậm hơn ~2.8x mà không thêm giá trị), latency là yếu tố sống còn (chat thời gian thực), hoặc chi phí token phải cực thấp.
