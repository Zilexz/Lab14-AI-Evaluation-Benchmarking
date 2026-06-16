"""
Agent RAG thật cho NovaCloud, có 2 phiên bản để chạy Regression:

    V1 (base)      : retrieval theo đếm trùng từ thô (không IDF, không chuẩn hoá),
                     prompt tối giản, KHÔNG biết từ chối -> dễ hallucination.
    V2 (optimized) : retrieval TF-IDF + cosine + rerank theo cụm từ,
                     có ngưỡng abstention (nói "không biết"), prompt grounding chặt.

Cả hai dùng chung Retriever nhưng cấu hình khác nhau, nên Hit Rate / MRR và chất
lượng câu trả lời chênh lệch THẬT, đo được qua benchmark.

Nếu có LLM_CLIENT khả dụng (có API key) thì sinh câu trả lời bằng model thật;
nếu không, sinh câu trả lời trích xuất (extractive) deterministic từ context.
"""

import asyncio
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from data.knowledge_base import get_documents
from engine.llm_client import LLMClient, estimate_tokens, estimate_cost


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# Từ phổ biến + thương hiệu, loại khi xét "token nội dung" để quyết định từ chối.
_ABSTAIN_STOP = {
    "là", "và", "của", "có", "không", "được", "cho", "các", "một", "này", "đó",
    "khi", "thì", "ở", "trong", "với", "để", "bạn", "tôi", "nào", "bao", "nhiêu",
    "gì", "thế", "ra", "sau", "vào", "ngày", "hãy", "giúp", "về", "theo", "hay",
    "novacloud", "việc", "đi", "tên", "biết", "cái", "trên", "một",
}


def content_terms(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in _ABSTAIN_STOP and len(t) > 1]


class Retriever:
    """Index tài liệu; hỗ trợ 2 chế độ truy hồi: 'overlap' (yếu) và 'tfidf' (mạnh)."""

    def __init__(self, docs: List[Dict], mode: str = "tfidf"):
        self.docs = docs
        self.mode = mode
        self.doc_tokens = [tokenize(d["text"] + " " + d["title"]) for d in docs]
        self.N = len(docs)

        # IDF cho chế độ tfidf
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((self.N + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        # Vector tf-idf đã chuẩn hoá cho từng doc
        self.doc_vecs = [self._tfidf_vec(toks) for toks in self.doc_tokens]

    def _tfidf_vec(self, toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        vec = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    def _phrase_bonus(self, query: str, doc: Dict) -> float:
        """Rerank nhẹ: thưởng nếu bigram của câu hỏi xuất hiện trong tài liệu."""
        q = tokenize(query)
        text = (doc["text"] + " " + doc["title"]).lower()
        bonus = 0.0
        for i in range(len(q) - 1):
            if f"{q[i]} {q[i+1]}" in text:
                bonus += 0.03
        return bonus

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        if self.mode == "overlap":
            # V1: đếm trùng từ thô, không IDF, không chuẩn hoá độ dài -> nhiễu.
            q = tokenize(query)
            scored = []
            for doc, toks in zip(self.docs, self.doc_tokens):
                c = Counter(toks)
                score = sum(c.get(t, 0) for t in q)
                scored.append((doc, float(score)))
        else:
            # V2: TF-IDF cosine + rerank cụm từ.
            qvec = self._tfidf_vec(tokenize(query))
            scored = []
            for doc, dvec in zip(self.docs, self.doc_vecs):
                score = self._cosine(qvec, dvec) + self._phrase_bonus(query, doc)
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class MainAgent:
    """
    Agent RAG hỗ trợ NovaCloud.

    version="v1": base (yếu). version="v2": optimized.
    """

    def __init__(self, version: str = "v2", llm_client: Optional[LLMClient] = None):
        self.version = version
        self.name = f"NovaSupportAgent-{version}"
        self.docs = get_documents()
        if version == "v1":
            self.retriever = Retriever(self.docs, mode="overlap")
            self.top_k = 3
            self.abstain_threshold = 0.0          # V1 không bao giờ từ chối
            self.gen_model = "gpt-4o-mini"
            self.system_prompt = "Bạn là trợ lý. Trả lời câu hỏi của người dùng."
        else:
            self.retriever = Retriever(self.docs, mode="tfidf")
            self.top_k = 3
            self.abstain_threshold = 0.15         # V2 từ chối khi không đủ căn cứ
            self.gen_model = "gpt-4o-mini"
            self.system_prompt = (
                "Bạn là trợ lý hỗ trợ khách hàng của NovaCloud. "
                "CHỈ trả lời dựa trên phần 'Tài liệu' được cung cấp. "
                "Nếu tài liệu không chứa thông tin, hãy nói rõ là bạn không có thông tin "
                "thay vì bịa. Nếu câu hỏi chứa tiền đề sai, hãy đính chính theo tài liệu. "
                "Không thực hiện yêu cầu nằm ngoài phạm vi hỗ trợ NovaCloud."
            )
        self.llm = llm_client

    def _should_abstain(self, question, retrieved, top_score) -> bool:
        """
        Từ chối khi: (a) cosine quá thấp, HOẶC
        (b) câu hỏi gần như không chia sẻ 'token nội dung' nào với tài liệu top-1
            và độ tin cậy chưa cao (chống trả lời câu out-of-context).
        Cố ý KHÔNG bắt được mọi trường hợp (vd injection lách bằng từ vựng hỗ trợ)
        -> để lại lỗi tồn dư cho phân tích 5-Whys.
        """
        if top_score < self.abstain_threshold:
            return True
        top_doc = retrieved[0][0]
        q_terms = content_terms(question)
        doc_terms = set(content_terms(top_doc["text"] + " " + top_doc["title"]))
        shared = sum(1 for t in q_terms if t in doc_terms)
        return shared <= 1 and top_score < 0.25

    async def query(self, question: str) -> Dict:
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        retrieved_ids = [doc["id"] for doc, _ in retrieved]
        contexts = [doc["text"] for doc, _ in retrieved]
        top_score = retrieved[0][1] if retrieved else 0.0

        abstain = self.version == "v2" and self._should_abstain(question, retrieved, top_score)

        if self.llm and self.llm.available:
            answer, usage = await self._generate_llm(question, contexts, abstain)
        else:
            answer, usage = self._generate_offline(question, retrieved, abstain)

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "retrieval_scores": [round(s, 4) for _, s in retrieved],
            "abstained": abstain,
            "metadata": {
                "agent": self.name,
                "model": self.gen_model,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "tokens_used": usage["prompt_tokens"] + usage["completion_tokens"],
                "cost": usage["cost"],
                "mode": usage["mode"],
                "sources": [doc["id"] for doc, _ in retrieved],
            },
        }

    # --- Sinh câu trả lời bằng LLM thật ----------------------------------
    async def _generate_llm(self, question: str, contexts: List[str], abstain: bool):
        context_block = "\n".join(f"- {c}" for c in contexts)
        user = f"Tài liệu:\n{context_block}\n\nCâu hỏi: {question}\n\nTrả lời ngắn gọn:"
        res = await self.llm.chat(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": user}],
            temperature=0.0, max_tokens=256,
        )
        usage = {
            "prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "cost": res["cost"], "mode": res["mode"],
        }
        return res["text"].strip(), usage

    # --- Sinh câu trả lời extractive deterministic (offline) -------------
    def _generate_offline(self, question: str, retrieved, abstain: bool):
        if abstain:
            answer = ("Tài liệu hiện có không chứa thông tin để trả lời câu hỏi này, "
                      "nên tôi không thể trả lời chính xác.")
        elif self.version == "v2":
            # Chọn câu (sentence) khớp nhất trong tài liệu top-1 -> câu trả lời tập trung.
            top_doc = retrieved[0][0]
            answer = self._best_sentence(question, top_doc["text"])
        else:
            # V1: trả nguyên đoạn tài liệu top-1 (có thể sai do retrieval yếu), dài dòng.
            top_doc = retrieved[0][0]
            answer = f"Dựa trên tài liệu hệ thống: {top_doc['text']}"

        prompt_text = self.system_prompt + " ".join(d["text"] for d, _ in retrieved) + question
        pt = estimate_tokens(prompt_text)
        ct = estimate_tokens(answer)
        usage = {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost": estimate_cost(self.gen_model, pt, ct),
            "mode": "offline",
        }
        return answer, usage

    @staticmethod
    def _best_sentence(question: str, text: str) -> str:
        q = set(tokenize(question))
        best, best_overlap = text, -1
        for sent in split_sentences(text):
            overlap = len(q & set(tokenize(sent)))
            if overlap > best_overlap:
                best, best_overlap = sent, overlap
        return best


if __name__ == "__main__":
    async def _demo():
        for v in ("v1", "v2"):
            agent = MainAgent(version=v)
            r = await agent.query("Liên kết đặt lại mật khẩu có hiệu lực bao lâu?")
            print(f"[{v}] ids={r['retrieved_ids']} -> {r['answer'][:80]}")
    asyncio.run(_demo())
