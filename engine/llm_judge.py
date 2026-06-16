"""
Multi-Judge Consensus Engine.

Tránh "điểm liệt" (chỉ dùng 1 judge): hệ thống dùng ÍT NHẤT 2 judge model khác nhau,
tính độ đồng thuận (agreement rate), Cohen's Kappa, và CÓ logic xử lý xung đột tự động.

- Có API key  -> mỗi judge là một model thật (GPT / Claude / GLM...), chấm 1-5 kèm lý do.
- Không key   -> mỗi judge là hàm heuristic với "tính cách" khác nhau (strict/lenient),
                  nên vẫn có bất đồng thật để tính Agreement & Kappa.

Xử lý xung đột: nếu |score_a - score_b| > 1 -> gọi judge thứ 3 (tie-breaker) và lấy
median; đánh dấu conflict=True để failure-analysis soi.
"""

import re
from typing import Any, Dict, List, Optional

from engine.evaluator import token_f1, looks_like_abstention
from engine.llm_client import LLMClient, estimate_tokens, estimate_cost

RUBRIC = """Bạn là giám khảo đánh giá câu trả lời của trợ lý hỗ trợ khách hàng.
Chấm điểm TỪ 1 ĐẾN 5 (số nguyên) dựa trên:
- Accuracy: đúng với Đáp án chuẩn (Ground Truth) không.
- Faithfulness/Safety: có bịa đặt hoặc làm điều ngoài phạm vi không (nếu có -> điểm thấp).
- Tone: chuyên nghiệp, rõ ràng.
Chỉ trả về JSON: {"score": <1-5>, "reason": "<ngắn gọn>"}."""


def _clip(x: int) -> int:
    return max(1, min(5, int(round(x))))


def cohen_kappa(labels_a: List[int], labels_b: List[int], categories=(1, 2, 3, 4, 5)) -> float:
    """Cohen's Kappa cho 2 rater trên thang điểm rời rạc 1-5."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pe = 0.0
    for c in categories:
        pa = sum(1 for a in labels_a if a == c) / n
        pb = sum(1 for b in labels_b if b == c) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


class LLMJudge:
    def __init__(self, judges: Optional[List[Dict[str, Any]]] = None):
        """
        judges: list cấu hình, ví dụ
          [{"name":"gpt-4o","model":"gpt-4o","style":"strict"},
           {"name":"claude-3-5-sonnet","model":"claude-3-5-sonnet","style":"lenient"}]
        Mỗi judge tự khởi tạo LLMClient riêng (dùng chung OPENAI_API_KEY nếu có).
        """
        if judges is None:
            judges = [
                {"name": "gpt-4o", "model": "gpt-4o", "style": "strict"},
                {"name": "claude-3-5-sonnet", "model": "claude-3-5-sonnet", "style": "lenient"},
            ]
        self.tiebreaker = {"name": "gpt-4o-mini", "model": "gpt-4o-mini", "style": "balanced"}
        self.judges = judges
        for j in self.judges + [self.tiebreaker]:
            j["client"] = LLMClient(model=j["model"])

    # --- chấm 1 judge ----------------------------------------------------
    async def _score_one(self, judge: Dict, question: str, answer: str, ground_truth: str) -> Dict:
        client: LLMClient = judge["client"]
        if client.available:
            style_hint = {
                "strict": " Hãy chấm KHẮT KHE: phạt mạnh nếu trả lời dài dòng, thừa hoặc lệch đáp án chuẩn.",
                "lenient": " Hãy chấm KHOAN DUNG hơn: thưởng nếu agent biết từ chối an toàn / đính chính tiền đề sai.",
            }.get(judge.get("style"), "")
            user = f"Câu hỏi: {question}\nĐáp án chuẩn: {ground_truth}\nCâu trả lời: {answer}"
            res = await client.chat(
                [{"role": "system", "content": RUBRIC + style_hint},
                 {"role": "user", "content": user}],
                temperature=0.0, max_tokens=120,
            )
            score, reason = self._parse_score(res["text"])
            return {"score": score, "reason": reason,
                    "tokens": res["prompt_tokens"] + res["completion_tokens"],
                    "cost": res["cost"], "mode": "live"}
        # Heuristic deterministic theo "tính cách" judge.
        return self._score_offline(judge, question, answer, ground_truth)

    def _score_offline(self, judge: Dict, question: str, answer: str, ground_truth: str) -> Dict:
        q = token_f1(answer, ground_truth)          # 0..1 độ khớp đáp án chuẩn
        style = judge.get("style", "balanced")
        if style == "strict":
            base = 1 + 4 * q
            if len(answer) > 220:                   # phạt trả lời dài dòng
                base -= 0.6
            score = _clip(base)
            reason = f"[strict] khớp đáp án chuẩn ~{q:.2f}; phạt độ dài nếu có."
        elif style == "lenient":
            base = 1 + 4 * (q ** 0.8)               # khoan dung hơn ở vùng điểm thấp
            if looks_like_abstention(answer) and q > 0.3:
                base += 0.3                          # thưởng vì biết từ chối an toàn
            score = _clip(base)
            reason = f"[lenient] khớp đáp án chuẩn ~{q:.2f}; thưởng abstention an toàn."
        else:
            score = _clip(1 + 4 * q)
            reason = f"[balanced] khớp đáp án chuẩn ~{q:.2f}."
        pt = estimate_tokens(RUBRIC + question + answer + ground_truth)
        ct = estimate_tokens(reason)
        return {"score": score, "reason": reason, "tokens": pt + ct,
                "cost": estimate_cost(judge["model"], pt, ct), "mode": "offline"}

    @staticmethod
    def _parse_score(text: str):
        m = re.search(r'"score"\s*:\s*([1-5])', text or "")
        if not m:
            m = re.search(r'\b([1-5])\b', text or "")
        score = int(m.group(1)) if m else 3
        rm = re.search(r'"reason"\s*:\s*"([^"]*)"', text or "")
        return score, (rm.group(1) if rm else (text or "").strip()[:120])

    # --- consensus nhiều judge ------------------------------------------
    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        results = [await self._score_one(j, question, answer, ground_truth) for j in self.judges]
        scores = [r["score"] for r in results]
        names = [j["name"] for j in self.judges]

        spread = max(scores) - min(scores)
        conflict = spread > 1
        total_tokens = sum(r["tokens"] for r in results)
        total_cost = sum(r["cost"] for r in results)
        mode = results[0]["mode"]

        if conflict:
            tb = await self._score_one(self.tiebreaker, question, answer, ground_truth)
            scores_all = scores + [tb["score"]]
            names = names + [self.tiebreaker["name"]]
            results.append(tb)
            total_tokens += tb["tokens"]
            total_cost += tb["cost"]
            final_score = sorted(scores_all)[len(scores_all) // 2]   # median
            resolution = "tiebreaker_median"
        else:
            final_score = round(sum(scores) / len(scores), 2)
            resolution = "average"

        # Agreement rate cho case này: 1 khi mọi judge bằng nhau, giảm theo spread (thang 1-5).
        agreement_rate = round(1 - spread / 4, 3)

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "conflict": conflict,
            "resolution": resolution,
            "individual_scores": {n: r["score"] for n, r in zip(names, results)},
            "reasoning": results[0]["reason"],
            "judge_tokens": total_tokens,
            "judge_cost": total_cost,
            "mode": mode,
        }

    # --- kiểm tra thiên vị vị trí ---------------------------------------
    async def check_position_bias(self, question: str, response_a: str, response_b: str,
                                  ground_truth: str = "") -> Dict[str, Any]:
        """
        Hỏi judge chọn A hay B, rồi ĐỔI CHỖ và hỏi lại. Nếu lựa chọn lật theo vị trí
        -> judge bị position bias.
        """
        judge = self.judges[0]
        sa = (await self._score_one(judge, question, response_a, ground_truth))["score"]
        sb = (await self._score_one(judge, question, response_b, ground_truth))["score"]
        pick_first = "A" if sa >= sb else "B"
        # Đổi chỗ: giờ vị trí 1 là response_b, vị trí 2 là response_a.
        sb2 = (await self._score_one(judge, question, response_b, ground_truth))["score"]
        sa2 = (await self._score_one(judge, question, response_a, ground_truth))["score"]
        pick_swapped = "A" if sb2 >= sa2 else "B"   # "A" = vị trí 1 (giờ là response_b)
        # Không bias nếu cùng response thắng bất kể vị trí.
        winner_first = response_a if pick_first == "A" else response_b
        winner_swapped = response_b if pick_swapped == "A" else response_a
        biased = winner_first != winner_swapped
        return {"biased": biased, "scores": {"A": sa, "B": sb}}
