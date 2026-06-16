"""
ExpertEvaluator — bộ đánh giá kiểu RAGAS (không phụ thuộc lib nặng), tính cho mỗi case:

    retrieval.hit_rate / retrieval.mrr  : qua RetrievalEvaluator (ground-truth ids)
    faithfulness                        : câu trả lời có bám vào context không (chống hallucination)
    answer_relevancy                    : câu trả lời có khớp expected_answer / câu hỏi không
    abstention_correct                  : với câu cần từ chối, agent có từ chối đúng không

Dùng token-overlap (F1/precision) làm proxy — deterministic, rẻ, tái lập được.
Nếu có LLM, có thể thay faithfulness bằng LLM-grader (đã chừa chỗ mở rộng).
"""

import re
from typing import Dict, List

from engine.retrieval_eval import RetrievalEvaluator

# Stopword tiếng Việt + ký hiệu hay gặp, loại bớt nhiễu khi tính overlap.
STOPWORDS = {
    "là", "và", "của", "có", "không", "được", "cho", "các", "một", "những", "này",
    "đó", "khi", "thì", "ở", "trong", "với", "để", "bạn", "tôi", "nào", "bao", "nhiêu",
    "gì", "thế", "ra", "sau", "trước", "vào", "ngày", "the", "a", "an", "to", "of",
    "hãy", "giúp", "về", "theo", "bằng", "hay", "đúng", "chứ", "phải", "mà", "nó",
}


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"\w+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def overlap_precision(answer: str, reference: str) -> float:
    """Tỉ lệ token nội dung của answer xuất hiện trong reference (đo grounding)."""
    a, r = content_tokens(answer), set(content_tokens(reference))
    if not a:
        return 0.0
    return sum(1 for t in a if t in r) / len(a)


def token_f1(a_text: str, b_text: str) -> float:
    a, b = content_tokens(a_text), content_tokens(b_text)
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    prec = inter / len(sa)
    rec = inter / len(sb)
    return 2 * prec * rec / (prec + rec)


ABSTAIN_MARKERS = ["không có thông tin", "không thể trả lời", "không có dữ liệu",
                   "không hỗ trợ", "không thể làm", "chưa rõ", "nói rõ hơn",
                   "ngoài phạm vi", "không đề cập"]


def looks_like_abstention(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in ABSTAIN_MARKERS)


class ExpertEvaluator:
    def __init__(self, top_k: int = 3):
        self.retrieval = RetrievalEvaluator(top_k=top_k)

    async def score(self, case: Dict, resp: Dict) -> Dict:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = resp.get("retrieved_ids", [])
        contexts = " ".join(resp.get("contexts", []))
        answer = resp.get("answer", "")

        retrieval = self.retrieval.score_case(expected_ids, retrieved_ids)

        # Faithfulness: answer bám context tới đâu (câu từ chối -> coi như grounded).
        if looks_like_abstention(answer):
            faithfulness = 1.0
        else:
            faithfulness = round(overlap_precision(answer, contexts), 4)

        relevancy = round(token_f1(answer, case.get("expected_answer", "")), 4)

        # Abstention: câu không có ground-truth ids cần được từ chối/đính chính.
        need_abstain = len(expected_ids) == 0
        abstention_correct = None
        if need_abstain:
            abstention_correct = 1.0 if looks_like_abstention(answer) else 0.0

        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "retrieval": retrieval,
            "abstention_correct": abstention_correct,
        }
