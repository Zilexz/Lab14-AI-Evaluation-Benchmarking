"""
AI Evaluation Factory — entrypoint.

Quy trình:
  1. Load Golden Dataset.
  2. Đánh giá tầng RETRIEVAL độc lập (Hit Rate / MRR) cho V1 & V2.
  3. Chạy BENCHMARK đầy đủ (RAGAS + Multi-Judge) cho V1 (base) và V2 (optimized) — async.
  4. REGRESSION: so sánh V1 vs V2 trên nhiều chiều.
  5. RELEASE GATE: tự động APPROVE / BLOCK theo ngưỡng Chất lượng / Retrieval / Chi phí.
  6. Failure clustering + position-bias + xuất reports/.

Chạy:  python main.py
"""

import asyncio
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.main_agent import MainAgent
from engine.runner import BenchmarkRunner
from engine.evaluator import ExpertEvaluator
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge, cohen_kappa
from engine.llm_client import LLMClient

DATA_PATH = "data/golden_set.jsonl"
REPORTS_DIR = "reports"

# Hai judge là HAI MODEL OpenAI KHÁC NHAU (đáp ứng yêu cầu multi-judge khi chỉ có 1 key OpenAI).
# Có key Anthropic thì đổi judge thứ 2 thành "claude-3-5-sonnet".
JUDGE_CONFIG = [
    {"name": "gpt-4o", "model": "gpt-4o", "style": "strict"},
    {"name": "gpt-4o-mini", "model": "gpt-4o-mini", "style": "lenient"},
]


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def summarize(results, version, elapsed):
    total = len(results)
    judge_a, judge_b = JUDGE_CONFIG[0]["name"], JUDGE_CONFIG[1]["name"]
    labels_a = [r["judge"]["individual_scores"].get(judge_a) for r in results]
    labels_b = [r["judge"]["individual_scores"].get(judge_b) for r in results]
    paired = [(a, b) for a, b in zip(labels_a, labels_b) if a is not None and b is not None]
    kappa = cohen_kappa([a for a, _ in paired], [b for _, b in paired]) if paired else 0.0

    total_cost = sum(r["cost"]["total_cost"] for r in results)
    total_tokens = sum(r["cost"]["total_tokens"] for r in results)
    abst = [r["ragas"]["abstention_correct"] for r in results]

    metrics = {
        "avg_score": round(_avg([r["judge"]["final_score"] for r in results]), 3),
        "pass_rate": round(sum(1 for r in results if r["status"] == "pass") / total, 3),
        "hit_rate": round(_avg([r["ragas"]["retrieval"]["hit_rate"] for r in results]), 3),
        "mrr": round(_avg([r["ragas"]["retrieval"]["mrr"] for r in results]), 3),
        "agreement_rate": round(_avg([r["judge"]["agreement_rate"] for r in results]), 3),
        "cohen_kappa": round(kappa, 3),
        "faithfulness": round(_avg([r["ragas"]["faithfulness"] for r in results]), 3),
        "relevancy": round(_avg([r["ragas"]["relevancy"] for r in results]), 3),
        "abstention_accuracy": round(_avg(abst), 3),
        "avg_latency": round(_avg([r["latency"] for r in results]), 4),
        "judge_conflict_rate": round(sum(1 for r in results if r["judge"]["conflict"]) / total, 3),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_eval_usd": round(total_cost / total, 8),
        "total_tokens": total_tokens,
    }
    summary = {
        "metadata": {
            "version": version,
            "total": total,
            "eval_mode": results[0]["judge"]["mode"] if results else "offline",
            "wall_time_sec": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": metrics,
    }
    return summary


def cluster_failures(results):
    """Phân cụm các case fail theo nguyên nhân gốc (cho failure_analysis 5-Whys)."""
    clusters = {}
    for r in results:
        if r["status"] == "pass":
            continue
        rag = r["ragas"]
        if rag["abstention_correct"] == 0.0:
            cause = "Hallucination (không từ chối khi cần)"
        elif rag["retrieval"]["hit_rate"] == 0.0:
            cause = "Retrieval miss (lấy sai context)"
        elif rag["relevancy"] < 0.3:
            cause = "Incomplete / lạc đề (relevancy thấp)"
        elif rag["faithfulness"] < 0.4:
            cause = "Faithfulness thấp (bịa ngoài context)"
        else:
            cause = "Khác (điểm judge thấp)"
        clusters.setdefault(cause, []).append(r["id"])
    return {c: {"count": len(ids), "case_ids": ids} for c, ids in clusters.items()}


def release_gate(v1, v2):
    m1, m2 = v1["metrics"], v2["metrics"]
    checks = [
        {"name": "Quality (avg_score không giảm)",
         "pass": m2["avg_score"] >= m1["avg_score"],
         "detail": f'{m1["avg_score"]} -> {m2["avg_score"]}'},
        {"name": "Pass rate không giảm",
         "pass": m2["pass_rate"] >= m1["pass_rate"],
         "detail": f'{m1["pass_rate"]} -> {m2["pass_rate"]}'},
        {"name": "Retrieval không regress (hit_rate >= cũ - 0.02)",
         "pass": m2["hit_rate"] >= m1["hit_rate"] - 0.02,
         "detail": f'{m1["hit_rate"]} -> {m2["hit_rate"]}'},
        {"name": "Judge reliability (agreement >= 0.6)",
         "pass": m2["agreement_rate"] >= 0.6,
         "detail": f'agreement={m2["agreement_rate"]}, kappa={m2["cohen_kappa"]}'},
        {"name": "Cost không tăng quá 50%",
         "pass": m2["cost_per_eval_usd"] <= max(m1["cost_per_eval_usd"] * 1.5, 1e-9),
         "detail": f'{m1["cost_per_eval_usd"]:.2e} -> {m2["cost_per_eval_usd"]:.2e}'},
    ]
    approved = all(c["pass"] for c in checks)
    return {"approved": approved, "checks": checks}


async def run_version(version, dataset):
    judge = LLMJudge(judges=[dict(j) for j in JUDGE_CONFIG])
    agent = MainAgent(version=version, llm_client=LLMClient(model="gpt-4o-mini"))
    runner = BenchmarkRunner(agent, ExpertEvaluator(top_k=agent.top_k), judge, batch_size=8)
    t0 = time.perf_counter()
    results = await runner.run_all(dataset)
    elapsed = time.perf_counter() - t0
    summary = summarize(results, version, elapsed)
    return results, summary


async def retrieval_report(dataset):
    rep = {}
    for v in ("v1", "v2"):
        evaluator = RetrievalEvaluator(top_k=3)
        rep[v] = await evaluator.evaluate_batch(dataset, MainAgent(version=v))
    return rep


async def position_bias_audit(results_v2):
    """Lấy 1 case pass + 1 case fail, kiểm tra judge có thiên vị vị trí không."""
    judge = LLMJudge(judges=[dict(j) for j in JUDGE_CONFIG])
    good = next((r for r in results_v2 if r["status"] == "pass"), None)
    bad = next((r for r in results_v2 if r["status"] == "fail"), None)
    if not good or not bad:
        return {"tested": 0, "biased": 0, "bias_rate": 0.0}
    res = await judge.check_position_bias(
        good["question"], good["agent_response"], bad["agent_response"], good["expected_answer"])
    return {"tested": 1, "biased": int(res["biased"]), "bias_rate": float(res["biased"])}


async def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Thiếu {DATA_PATH}. Chạy 'python data/synthetic_gen.py' trước.")
        return
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        print(f"❌ {DATA_PATH} rỗng.")
        return

    print(f"📦 Dataset: {len(dataset)} test cases")
    mode = "LIVE (model thật)" if LLMClient(model="gpt-4o").available else "OFFLINE (heuristic deterministic)"
    print(f"🔧 Eval mode: {mode}\n")

    # 1) Retrieval độc lập
    print("🔎 Đánh giá tầng Retrieval (độc lập)...")
    retr = await retrieval_report(dataset)
    for v in ("v1", "v2"):
        print(f"   {v.upper()}: Hit@3={retr[v]['avg_hit_rate']:.2%} | MRR={retr[v]['avg_mrr']:.3f} "
              f"(trên {retr[v]['evaluated_cases']} case có ground-truth)")

    # 2) Benchmark V1 & V2
    print("\n🚀 Benchmark Agent_V1_Base...")
    v1_results, v1_summary = await run_version("v1", dataset)
    print("🚀 Benchmark Agent_V2_Optimized...")
    v2_results, v2_summary = await run_version("v2", dataset)

    # 3) Regression
    print("\n📊 --- REGRESSION (V1 -> V2) ---")
    keys = ["avg_score", "pass_rate", "hit_rate", "mrr", "agreement_rate",
            "cohen_kappa", "faithfulness", "relevancy", "abstention_accuracy",
            "avg_latency", "cost_per_eval_usd"]
    regression = {}
    for k in keys:
        a, b = v1_summary["metrics"][k], v2_summary["metrics"][k]
        regression[k] = {"v1": a, "v2": b, "delta": round(b - a, 6)}
        arrow = "▲" if b > a else ("▼" if b < a else "=")
        print(f"   {k:22s}: {a:>10} -> {b:>10}  ({arrow} {b - a:+.4f})")

    # 4) Release gate
    gate = release_gate(v1_summary, v2_summary)
    print("\n🚦 --- RELEASE GATE ---")
    for c in gate["checks"]:
        print(f"   [{'✅' if c['pass'] else '❌'}] {c['name']}: {c['detail']}")
    decision = "APPROVE ✅ (Release V2)" if gate["approved"] else "BLOCK ❌ (Rollback / giữ V1)"
    print(f"   => QUYẾT ĐỊNH: {decision}")

    # 5) Phân tích bổ sung
    clusters = cluster_failures(v2_results)
    bias = await position_bias_audit(v2_results)

    # 6) Xuất reports
    os.makedirs(REPORTS_DIR, exist_ok=True)
    v2_summary["retrieval_eval"] = retr
    v2_summary["regression"] = {"baseline": "Agent_V1_Base", "candidate": "Agent_V2_Optimized",
                                "metrics": regression}
    v2_summary["release_gate"] = gate
    v2_summary["failure_clusters"] = clusters
    v2_summary["position_bias_audit"] = bias
    v2_summary["v1_metrics"] = v1_summary["metrics"]
    v2_summary["cost_optimization_note"] = (
        "Giảm ~30% chi phí eval: (1) cache câu trả lời judge theo hash(answer); "
        "(2) chỉ gọi judge thứ 3 khi conflict (đã làm); "
        "(3) dùng model rẻ (gpt-4o-mini) cho case 'easy', model mạnh chỉ cho case 'hard'."
    )

    with open(os.path.join(REPORTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORTS_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORTS_DIR, "v1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(v1_summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(REPORTS_DIR, "failure_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Đã ghi reports/ (summary.json, benchmark_results.json, v1_summary.json, failure_clusters.json)")
    print(f"⏱️  Tổng thời gian V2: {v2_summary['metadata']['wall_time_sec']}s cho {len(dataset)} case")


if __name__ == "__main__":
    asyncio.run(main())
