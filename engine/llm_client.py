"""
LLM Client tương thích OpenAI (OpenAI / FPT AI Marketplace / OpenRouter / mọi endpoint /v1).

- Nếu có API key (.env hoặc biến môi trường) và đã cài `openai`  -> gọi model THẬT (mode="live").
- Nếu không  -> trả về None để Agent/Judge dùng nhánh heuristic deterministic (mode="offline").

Luôn hạch toán token & chi phí (USD) để phục vụ báo cáo Cost của lab,
kể cả khi chạy offline (ước lượng token từ độ dài text + bảng giá model).
"""

import os
from typing import List, Dict, Optional

# Giá tham khảo (USD / 1 triệu token): (input, output)
PRICING = {
    "gpt-4o-mini":          (0.15, 0.60),
    "gpt-4o":               (2.50, 10.00),
    "gpt-4.1-mini":         (0.40, 1.60),
    "claude-3-5-sonnet":    (3.00, 15.00),
    "claude-3-5-haiku":     (0.80, 4.00),
    "glm-4-flash":          (0.00, 0.00),
    "glm-4-plus":           (0.70, 0.70),
    "default":              (0.15, 0.60),
}


def estimate_tokens(text: str) -> int:
    """Ước lượng số token ~ ký tự / 4 (heuristic đủ tốt cho báo cáo chi phí)."""
    if not text:
        return 1
    return max(1, len(text) // 4)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, PRICING["default"])
    return (prompt_tokens / 1_000_000) * inp + (completion_tokens / 1_000_000) * out


def load_dotenv(path: str = ".env") -> None:
    """Parser .env tối giản (không phụ thuộc python-dotenv)."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
    except Exception:
        pass


class LLMClient:
    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY",
                 base_url: Optional[str] = None):
        load_dotenv()
        self.model = model
        self.api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None
        if self.api_key:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def chat(self, messages: List[Dict], temperature: float = 0.0,
                   max_tokens: int = 512) -> Optional[Dict]:
        """Trả về {text, prompt_tokens, completion_tokens, cost, mode} hoặc None nếu offline."""
        if not self.available:
            return None
        resp = await self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        pt = getattr(usage, "prompt_tokens", estimate_tokens(str(messages)))
        ct = getattr(usage, "completion_tokens", estimate_tokens(text))
        return {
            "text": text,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost": estimate_cost(self.model, pt, ct),
            "mode": "live",
        }
