"""Locust load test for the chat gateway: mixed realistic traffic.

Run: ``locust -f scripts/load_test.py --host http://localhost:8000 -u 50 -r 5 -t 5m --headless``
Watch vLLM metrics (KV-cache usage, waiting requests) in Grafana while ramping users up to find the
``max-num-seqs`` / ``max-model-len`` combination that stays below OOM and the p95 latency target.
"""

from __future__ import annotations

import random
import uuid

from locust import HttpUser, between, task

ORDER_IDS = [str(100000 + i * 7) for i in range(0, 3000, 13)]
FAQ = ["Сколько идёт доставка в Самарканд?", "Как оформить рассрочку?", "Qanday to'lov usullari bor?",
       "Сколько хранится заказ в пункте выдачи?", "Можно ли вернуть косметику?"]  # fmt: skip
SEARCH = ["есть наушники JBL?", "ищу кроссовки Nike до 1 500 000 сум", "Samsung televizor bormi?"]


class Shopper(HttpUser):
    """One customer session."""

    wait_time = between(1, 4)

    def on_start(self) -> None:
        self.session_id = uuid.uuid4().hex

    def _chat(self, message: str, name: str) -> None:
        self.client.post("/chat", json={"session_id": self.session_id, "message": message}, name=name, timeout=90)

    @task(4)
    def order_status(self) -> None:
        self._chat(f"где мой заказ {random.choice(ORDER_IDS)}?", "order_status")

    @task(3)
    def faq(self) -> None:
        self._chat(random.choice(FAQ), "faq")

    @task(2)
    def search(self) -> None:
        self._chat(random.choice(SEARCH), "search")

    @task(1)
    def chitchat(self) -> None:
        self._chat(random.choice(["привет", "salom", "спасибо!"]), "chitchat")
