# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. Case nào nên dùng multi-agent? Vì sao?
2. Case nào không nên dùng multi-agent? Vì sao?

### Trả lời (đã điền — tham khảo số liệu trong reports/benchmark_report.md)

1. **Nên dùng multi-agent khi:**
   - Task cần **nguồn kiểm chứng** (research, fact-check, so sánh thông tin): benchmark cho thấy multi-agent đạt citation coverage 90% còn single-agent là 0% vì không có bước search.
   - Task có **các bước tách biệt rõ** (search → phân tích → viết): mỗi agent có prompt ngắn, vai trò hẹp nên ít loãng context hơn một prompt khổng lồ.
   - Cần **debug & trace từng khâu**: khi output sai, mở trace thấy ngay sai ở researcher hay analyst; single-agent chỉ có 1 span nên khó truy.
   - Cần **guardrail/retry độc lập**: một agent fail có thể retry mà không phải sinh lại toàn bộ câu trả lời.
   - Chấp nhận đánh đổi: chậm hơn ~2.8x và đắt hơn ~4x (theo benchmark của lab).

2. **Không nên dùng multi-agent khi:**
   - Câu hỏi **đơn giản, 1 lượt sinh văn bản là đủ** (tóm tắt, dịch, trả lời từ knowledge nội tại): multi-agent chỉ thêm latency + token mà không thêm giá trị.
   - **Latency sống còn** (chat thời gian thực, autocomplete): 21s vs 7.5s là khác biệt trải nghiệm rõ rệt.
   - **Chi phí token cực kỳ nhạy cảm** (free tier, hàng triệu query/ngày): 7 lần gọi LLM mỗi request nhân lên rất nhanh.
   - Workflow quá **đơn giản để tách vai**: thêm agent chỉ vì "cho giống multi-agent" vi phạm đúng nguyên tắc đầu tiên của lab — không thêm agent nếu không có lý do rõ ràng.
