"""
Đánh giá tầng Retrieval: Hit Rate @k và MRR.

Phải chứng minh Retrieval tốt TRƯỚC khi đánh giá Generation — nếu retriever lấy sai
context thì mọi lỗi Generation chỉ là hệ quả (xem failure_analysis 5-Whys).

Quy ước: nếu expected_retrieval_ids rỗng (câu out-of-context / cần từ chối) thì
KHÔNG tính vào Hit Rate / MRR (trả về None) — đo abstention bằng metric khác.
"""

from typing import List, Dict, Optional


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str],
                           top_k: Optional[int] = None) -> Optional[float]:
        """1.0 nếu có ít nhất 1 expected_id nằm trong top_k retrieved; None nếu không có ground truth."""
        if not expected_ids:
            return None
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> Optional[float]:
        """MRR = 1/vị trí (1-indexed) của expected_id đầu tiên; 0 nếu không thấy; None nếu không có GT."""
        if not expected_ids:
            return None
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def score_case(self, expected_ids: List[str], retrieved_ids: List[str]) -> Dict:
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
        }

    async def evaluate_batch(self, dataset: List[Dict], agent) -> Dict:
        """
        Chạy retrieval cho toàn bộ dataset qua `agent` và tổng hợp Hit Rate / MRR.
        Dùng để báo cáo độc lập tầng Retrieval trước khi chạy Generation.
        """
        hits, mrrs, evaluated = [], [], 0
        for case in dataset:
            resp = await agent.query(case["question"])
            hr = self.calculate_hit_rate(case.get("expected_retrieval_ids", []), resp["retrieved_ids"])
            rr = self.calculate_mrr(case.get("expected_retrieval_ids", []), resp["retrieved_ids"])
            if hr is not None:
                hits.append(hr)
                mrrs.append(rr)
                evaluated += 1
        n = max(1, len(hits))
        return {
            "avg_hit_rate": sum(hits) / n,
            "avg_mrr": sum(mrrs) / n,
            "evaluated_cases": evaluated,
            "skipped_no_ground_truth": len(dataset) - evaluated,
        }
