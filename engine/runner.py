"""
BenchmarkRunner — chạy toàn bộ test cases SONG SONG (asyncio) theo batch để tránh
rate-limit, đo latency và gom Cost/Token cho mỗi case.

Pipeline mỗi case:  Agent.query -> ExpertEvaluator.score (RAGAS + retrieval)
                    -> LLMJudge.evaluate_multi_judge (consensus nhiều judge).
"""

import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, batch_size: int = 8):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.batch_size = batch_size

    async def run_single_test(self, case: Dict) -> Dict:
        start = time.perf_counter()
        response = await self.agent.query(case["question"])
        latency = time.perf_counter() - start

        ragas = await self.evaluator.score(case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            case["question"], response["answer"], case["expected_answer"]
        )

        agent_cost = response["metadata"]["cost"]
        agent_tokens = response["metadata"]["tokens_used"]
        judge_cost = judge_result["judge_cost"]
        judge_tokens = judge_result["judge_tokens"]

        # Pass/Fail: điểm consensus >=3 VÀ (nếu case cần từ chối) phải từ chối đúng.
        passed = judge_result["final_score"] >= 3
        if ragas["abstention_correct"] is not None and ragas["abstention_correct"] == 0.0:
            passed = False

        return {
            "id": case.get("id"),
            "type": case.get("metadata", {}).get("type"),
            "difficulty": case.get("metadata", {}).get("difficulty"),
            "category": case.get("metadata", {}).get("category"),
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "agent_response": response["answer"],
            "abstained": response["abstained"],
            "retrieved_ids": response["retrieved_ids"],
            "expected_retrieval_ids": case.get("expected_retrieval_ids", []),
            "latency": round(latency, 4),
            "ragas": ragas,
            "judge": judge_result,
            "cost": {
                "agent_cost": round(agent_cost, 8),
                "judge_cost": round(judge_cost, 8),
                "total_cost": round(agent_cost + judge_cost, 8),
                "agent_tokens": agent_tokens,
                "judge_tokens": judge_tokens,
                "total_tokens": agent_tokens + judge_tokens,
            },
            "status": "pass" if passed else "fail",
        }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        results: List[Dict] = []
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batch_results = await asyncio.gather(*(self.run_single_test(c) for c in batch))
            results.extend(batch_results)
        return results
