# Reflection cá nhân — Lab Day 14: AI Evaluation Factory

> **Họ tên:** Nguyễn Đức Hiếu — **Mã học viên:** 2A202600680
> **Vai trò trong nhóm:** *(điền: Data / AI-Backend / DevOps-Analyst)*
>
> ⚠️ Đây là báo cáo **cá nhân** — phần "Đóng góp" bên dưới cần bạn chỉnh lại đúng
> những gì *bản thân bạn* đã commit/làm. Phần "Technical Depth" là kiến thức nền,
> có thể giữ và diễn giải lại bằng lời của bạn.

## 1. Engineering Contribution (đóng góp kỹ thuật)
*(Điều chỉnh theo đóng góp thực tế của bạn; minh chứng bằng Git commits.)*

- Module phụ trách chính: **Multi-Judge Consensus Engine** (`engine/llm_judge.py`).
- Việc đã làm:
    - Thiết kế `evaluate_multi_judge`: chấm bằng ≥2 judge, tính `agreement_rate` theo độ lệch điểm.
    - Logic **xử lý xung đột tự động**: khi `|score_a − score_b| > 1` thì gọi judge thứ 3 và lấy **median** (`resolution = tiebreaker_median`).
    - Hàm `cohen_kappa()` đo độ tin cậy giữa 2 rater trên thang 1–5.
    - `check_position_bias()`: đảo vị trí A/B để phát hiện thiên vị.
- Đóng góp khác: nhạch toán token/cost (`engine/llm_client.py`), tích hợp vào `BenchmarkRunner`.

## 2. Technical Depth (chiều sâu kỹ thuật)

**MRR (Mean Reciprocal Rank).** Đo chất lượng *xếp hạng* của retriever: `RR = 1/vị_trí` của tài liệu
đúng đầu tiên, MRR là trung bình trên tập câu hỏi. Khác Hit Rate (chỉ quan tâm "có trúng trong top-k
hay không"), MRR phạt việc đặt tài liệu đúng ở hạng thấp. Trong run của nhóm: Hit@3 không đổi
(0.982) nhưng MRR tăng 0.930 → 0.964 khi V2 dùng TF-IDF + rerank ⇒ tài liệu đúng được đẩy lên hạng cao hơn.

**Cohen's Kappa (κ).** Đo độ đồng thuận giữa 2 judge **đã loại trừ may rủi**:
`κ = (p_o − p_e) / (1 − p_e)`, với `p_o` = tỉ lệ đồng ý quan sát được, `p_e` = tỉ lệ đồng ý kỳ vọng
ngẫu nhiên. κ = 1 là hoàn hảo, 0 là bằng đoán mò. Run LIVE của nhóm cho một bài học đắt giá:
agreement thô = **93.8%** nhưng κ chỉ **0.398** ("fair"). Lý do: model thật trả lời tốt nên điểm
**cụm hết ở 4–5**, khiến `p_e` (đồng ý ngẫu nhiên) rất cao và kéo κ xuống. ⇒ Không được tin agreement
thô; với tập điểm lệch phải báo cáo thêm κ theo dải khó / Krippendorff's alpha.

**Position Bias.** LLM-judge có xu hướng thiên vị câu trả lời ở **vị trí đầu** khi so sánh cặp. Cách phát
hiện: chấm (A,B) rồi đảo thành (B,A); nếu "người thắng" đổi theo vị trí ⇒ judge bị bias. Audit của nhóm:
bias_rate = 0.0 trên mẫu kiểm tra.

**Trade-off Chi phí ↔ Chất lượng.** Judge mạnh (gpt-4o, claude) chính xác nhưng đắt; gọi 2–3 judge cho mọi
case là lãng phí. Ba đòn bẩy giảm ~30% chi phí mà giữ độ chính xác: (1) cache điểm theo `hash(answer)`;
(2) chỉ gọi judge thứ 3 khi có xung đột; (3) định tuyến model theo độ khó (model rẻ cho case easy).
Chi phí thực đo (LIVE, gpt-4o + gpt-4o-mini): **$0.00093/eval**, async 60 case trong **37.8s** (< 2 phút);
chế độ offline deterministic chạy < 0.1s để debug nhanh không tốn API.

## 3. Problem Solving (vấn đề đã gặp & cách xử lý)

- **Vấn đề:** Ban đầu V2 không từ chối câu out-of-context vì cosine TF-IDF bị từ phổ biến
  ("trực tuyến", thương hiệu) kéo lên trên ngưỡng. → **Xử lý:** bổ sung tín hiệu thứ 2 — kiểm tra
  số "token nội dung" thực sự chia sẻ với tài liệu top-1, kèm prompt grounding chặt; abstention_accuracy
  tăng 0.2→0.8 (offline) và **0.4→1.0** (live). *(Bài học: một tín hiệu tin cậy đơn lẻ — chỉ cosine — không đủ.)*
- **Vấn đề:** Lỗi mã hoá tiếng Việt trên console Windows (cp1252). → **Xử lý:** `sys.stdout.reconfigure("utf-8")`
  và ghi file luôn `encoding="utf-8"`.
- **Phát hiện đáng giá nhất:** lỗi còn lại đều ở **Generation/Reasoning** (vd case_057 — LLM đồng ý
  tiền đề sai) chứ không phải Retrieval (Hit@3 = 98%) — minh chứng vì sao phải đo Retrieval *tách riêng*
  trước khi đổ lỗi cho LLM.

## 4. Điều sẽ làm khác lần sau
- Index ở mức **câu** thay vì cả đoạn, thêm reranker cross-encoder.
- Dùng NLI/groundedness làm tín hiệu abstention thứ hai.
- Chấm bằng model thật (cắm `.env`) để đối chiếu với baseline heuristic deterministic.
