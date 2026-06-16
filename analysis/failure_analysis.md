# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Sản phẩm: AI Evaluation Factory cho Agent hỗ trợ khách hàng **NovaCloud**.
> Số liệu trích **trực tiếp** từ `reports/summary.json` & `reports/benchmark_results.json`.
> **Run này: `eval_mode = live`** — Agent sinh câu trả lời bằng `gpt-4o-mini`, chấm bằng
> 2 judge thật `gpt-4o` (strict) + `gpt-4o-mini` (lenient).
> *(Không có API key thì pipeline tự chạy chế độ `offline` deterministic, tái lập 100%.)*

## 1. Tổng quan Benchmark (Agent V2 - Optimized, LIVE)

- **Tổng số cases:** 60 (48 fact/reasoning + 12 red-team/edge)
- **Tỉ lệ Pass/Fail:** 59 / 1 → **pass_rate = 98.3%**
- **RAGAS:** Faithfulness = **0.941** · Answer Relevancy = **0.541**
- **Retrieval:** Hit@3 = **98.2%** · MRR = **0.964** (55 case có ground-truth)
- **LLM-Judge consensus:** **4.742 / 5.0**
- **Multi-Judge reliability:** Agreement = **93.8%** · Cohen's Kappa = **0.398** · conflict_rate = 3.3%
- **Abstention accuracy:** **100%** (5/5 câu out-of-context bị từ chối đúng)
- **Performance & Cost:** 60 case trong **37.84s** (async, < 2 phút ✓) · avg_latency **1.0s** · **$0.00093/eval** (tổng $0.0558, 47.6k token)
- **Position-bias audit:** 0/1 (bias_rate = 0.0)

## 2. So sánh Regression (V1 Base → V2 Optimized, LIVE)

| Metric | V1 | V2 | Δ | Nhận xét |
|---|---:|---:|---:|---|
| avg_score | 4.575 | **4.742** | +0.17 | V2 trả lời chuẩn & bám context hơn |
| pass_rate | 0.917 | **0.983** | +0.07 | |
| hit_rate@3 | 0.982 | 0.982 | 0.00 | retrieval đã tốt từ V1 |
| MRR | 0.930 | **0.964** | +0.03 | TF-IDF + rerank đẩy tài liệu đúng lên hạng cao |
| faithfulness | 0.878 | **0.941** | +0.06 | V2 ít bịa hơn |
| **abstention_accuracy** | 0.400 | **1.000** | **+0.60** | đóng góp lớn nhất của V2 |
| avg_latency | 1.16s | **1.00s** | −0.16 | V2 trả lời ngắn, gọn |
| agreement_rate | 0.929 | 0.938 | +0.01 | |
| cohen_kappa | 0.395 | 0.398 | +0.00 | *thấp do điểm cụm ở vùng cao — xem 5-Whys #2* |
| cost_per_eval | $0.00092 | $0.00093 | ~0 | |

➡️ **Release Gate: APPROVE** — 5/5 cổng PASS (Quality · Pass-rate · Retrieval · Judge-reliability · Cost).

## 3. Phân nhóm lỗi (Failure Clustering) — V2 LIVE

| Nhóm lỗi | Số lượng | Case | Nguyên nhân |
|----------|:--------:|------|-------------|
| Khác — sai reasoning, đồng ý tiền đề sai | **1** | case_057 | Generation suy luận sai trên chính sách có điều kiện |

> Pass-rate 98.3% nhờ LLM thật sinh câu trả lời tốt; **lỗi còn lại đều là lỗi *suy luận*,
> không phải lỗi *truy hồi*** (Hit@3 = 98%). Phần 4 phân tích case lỗi + 2 điểm yếu hệ thống
> phát hiện từ phân phối điểm (dù các case đó vẫn "pass").

## 4. Phân tích 5 Whys

### Case #1 — `case_057` (reasoning / false-premise): câu lỗi duy nhất
- **Câu hỏi:** "Tôi huỷ gói sau 3 ngày, vừa được hoàn tiền **vừa** được dùng tiếp đến hết chu kỳ phải không?"
- **Số liệu:** hit_rate = 1.0, MRR = 1.0, relevancy = 0.58, **judge = 1.0/5** (cả gpt-4o & gpt-4o-mini đều cho 1).
- **Đáp án đúng:** Huỷ trong 7 ngày → hoàn 100% tiền, **nhưng khi đã hoàn tiền thì KHÔNG dùng tiếp**; chỉ khi huỷ *sau* 7 ngày (không hoàn tiền) mới dùng đến hết chu kỳ → hai vế **loại trừ nhau**.
- **Agent trả lời (sai):** "Đúng, bạn sẽ được hoàn tiền 100% **và** vẫn có thể sử dụng gói đến hết chu kỳ."
1. **Symptom:** Agent xác nhận một tiền đề sai (hai đặc quyền loại trừ nhau).
2. **Why 1:** LLM "đồng tình" với cách diễn đạt khẳng định của người dùng (sycophancy).
3. **Why 2:** Context đúng được lấy về, nhưng prompt **không buộc** model kiểm tra mâu thuẫn/điều kiện trước khi đồng ý.
4. **Why 3:** Chính sách hoàn tiền là **có điều kiện** (if ≤7 ngày … else …) — đòi suy luận đa bước mà prompt không hướng dẫn.
5. **Why 4:** Không có bước "verify-premise" / self-check trong pipeline sinh câu trả lời.
6. **Root Cause:** **Thiếu ràng buộc suy luận điều kiện & kiểm chứng tiền đề trong prompt** (lỗi Generation, KHÔNG phải Retrieval).

### Case #2 — Nghịch lý "Agreement cao nhưng Kappa thấp" (lỗi độ tin cậy đo lường)
- **Số liệu:** agreement_rate = 0.938 nhưng **Cohen's Kappa = 0.398** ("fair").
1. **Symptom:** Hai chỉ số độ tin cậy "đá nhau".
2. **Why 1:** Hầu hết câu trả lời tốt → điểm judge **cụm ở 4–5**.
3. **Why 2:** Khi nhãn lệch phân phối, xác suất đồng ý **ngẫu nhiên** `p_e` rất cao → Kappa = `(p_o−p_e)/(1−p_e)` bị kéo xuống.
4. **Why 3:** Agreement thô không trừ may rủi nên "trông đẹp" một cách ảo.
5. **Root Cause:** **Dùng sai thước đo cho phân phối lệch** — với tập dễ, nên báo cáo thêm Kappa *theo dải điểm khó* hoặc Krippendorff's alpha, và bổ sung case khó để cân bằng phân phối.

### Case #3 — Hạn chế của metric Relevancy (vd `case_036`, vẫn "pass")
- **Số liệu:** câu trả lời ĐÚNG ("Có, quản trị viên có thể bắt buộc bật 2FA cho cả nhóm") nhưng **relevancy = 0.00**.
1. **Symptom:** Relevancy = 0 cho câu trả lời đúng → metric đánh giá thấp oan.
2. **Why 1:** Relevancy dùng **token-F1** so với `expected_answer` ngắn; đáp án chuẩn nhấn mạnh vế phụ ("ghi vào Nhật ký hoạt động") mà câu hỏi không đòi.
3. **Why 2:** Overlap từ vựng **không nắm được tương đương ngữ nghĩa**.
4. **Root Cause:** **Proxy lexical undervalue** → nên thay/bổ sung relevancy bằng embedding-similarity hoặc LLM-grader cho đúng bản chất.

## 5. Kế hoạch cải tiến (Action Plan) — V3
- [ ] **Verify-premise / self-check** trong prompt sinh: buộc model phát hiện tiền đề sai & chính sách có điều kiện (vá Root Cause case_057).
- [ ] **Relevancy bằng embedding/LLM** thay token-F1 (vá undervalue ở case_036/048).
- [ ] Báo cáo **Kappa theo dải khó** + thêm case khó để phân phối điểm cân bằng (vá đo lường độ tin cậy).
- [ ] **Reranker cross-encoder** sau retrieval để giữ MRR ≥ 0.98 khi mở rộng KB.
- [ ] Mục tiêu V3: pass_rate ≥ 0.99, relevancy ≥ 0.70, giữ abstention = 1.0.

## 6. Đề xuất giảm ~30% chi phí Eval (không giảm độ chính xác)
1. **Cache điểm judge theo `hash(answer)`** — câu trả lời trùng (đặc biệt các abstention giống nhau) không chấm lại.
2. **Tiebreaker có điều kiện** (đã triển khai): chỉ gọi judge thứ 3 khi `|score_a − score_b| > 1` → tiết kiệm ~1/3 lời gọi.
3. **Định tuyến model theo độ khó:** case `easy` chấm bằng `gpt-4o-mini`, chỉ `hard`/`adversarial` mới dùng `gpt-4o` → giảm mạnh chi phí mà giữ độ chính xác ở nơi cần.
