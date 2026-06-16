"""
Synthetic Data Generation (SDG) — sinh Golden Dataset cho benchmark.

Nguồn: data/knowledge_base.py (KNOWLEDGE_BASE + RED_TEAM_CASES).
Mỗi test case có Ground Truth Retrieval IDs -> cho phép tính Hit Rate & MRR.

Chạy:  python data/synthetic_gen.py   -> tạo data/golden_set.jsonl (60+ cases)

Nếu có API key (LLM khả dụng) sẽ paraphrase thêm 1 biến thể câu hỏi cho mỗi doc
để tăng tính đa dạng; nếu không, vẫn sinh đủ >50 cases deterministic (tái lập 100%).
"""

import asyncio
import json
import os
import sys

# Cho phép chạy trực tiếp `python data/synthetic_gen.py` từ thư mục gốc repo.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.knowledge_base import KNOWLEDGE_BASE, RED_TEAM_CASES  # noqa: E402
from engine.llm_client import LLMClient  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")


def build_fact_cases():
    """Sinh case fact/reasoning từ các cặp QA mầm trong KB."""
    cases = []
    n = 0
    for doc in KNOWLEDGE_BASE:
        for qa in doc["qa"]:
            n += 1
            cases.append({
                "id": f"case_{n:03d}",
                "question": qa["q"],
                "expected_answer": qa["a"],
                "context": doc["text"],
                "expected_retrieval_ids": [doc["id"]],
                "metadata": {
                    "category": doc["category"],
                    "difficulty": qa["difficulty"],
                    "type": qa["type"],
                    "source": "kb_seed",
                },
            })
    return cases


def build_redteam_cases(start_idx):
    cases = []
    for i, rc in enumerate(RED_TEAM_CASES, start=start_idx):
        cases.append({
            "id": f"case_{i:03d}",
            "question": rc["question"],
            "expected_answer": rc["expected_answer"],
            "context": "",  # red-team: không gắn 1 context cố định
            "expected_retrieval_ids": rc["expected_retrieval_ids"],
            "metadata": {**rc["metadata"], "source": "red_team"},
        })
    return cases


async def maybe_augment(cases, client: LLMClient):
    """(Tuỳ chọn) Dùng LLM paraphrase câu hỏi fact đầu tiên của mỗi category."""
    if not client.available:
        return []
    seen, extra, n = set(), [], len(cases)
    for c in cases:
        cat = c["metadata"].get("category")
        if c["metadata"].get("source") != "kb_seed" or cat in seen:
            continue
        seen.add(cat)
        res = await client.chat(
            [{"role": "system", "content": "Bạn paraphrase câu hỏi tiếng Việt, giữ nguyên ý."},
             {"role": "user", "content": f"Viết lại câu hỏi sau theo cách khác, ngắn gọn:\n{c['question']}"}],
            temperature=0.7, max_tokens=60,
        )
        if res and res["text"].strip():
            n += 1
            nc = json.loads(json.dumps(c))  # deep copy
            nc["id"] = f"case_{n:03d}"
            nc["question"] = res["text"].strip().strip('"')
            nc["metadata"]["source"] = "llm_paraphrase"
            extra.append(nc)
    return extra


async def main():
    fact_cases = build_fact_cases()
    redteam_cases = build_redteam_cases(start_idx=len(fact_cases) + 1)
    cases = fact_cases + redteam_cases

    client = LLMClient(model=os.environ.get("SDG_MODEL", "gpt-4o-mini"))
    if client.available:
        print("🔌 LLM khả dụng -> augment paraphrase...")
        cases += await maybe_augment(fact_cases, client)
    else:
        print("📴 Không có API key -> sinh deterministic (đủ >50 cases).")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Thống kê nhanh
    by_type = {}
    for c in cases:
        t = c["metadata"].get("type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"✅ Đã tạo {len(cases)} test cases -> {OUT_PATH}")
    print(f"   - Fact/Reasoning: {len(fact_cases)} | Red-team/Edge: {len(redteam_cases)}")
    print(f"   - Phân loại theo type: {by_type}")


if __name__ == "__main__":
    asyncio.run(main())
