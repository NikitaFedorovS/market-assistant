"""Agent tools: Pydantic argument schemas + implementations over the marketplace SQLite DB.

The same implementations are used to build synthetic training dialogs, so tool results the model
sees in training match what it gets in production.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from market_assistant.data.policies import CATEGORY_BY_SLUG, ORDER_STATUSES, REFUND_DAYS, RETURN_DAYS


class GetOrderStatusArgs(BaseModel):
    """Get the current status and delivery estimate of an order."""

    order_id: str = Field(..., pattern=r"^\d{5,7}$", description="Order number, 5–7 digits")


class CreateReturnArgs(BaseModel):
    """Create a return request for a delivered order. Call only after the user confirmed."""

    order_id: str = Field(..., pattern=r"^\d{5,7}$", description="Order number, 5–7 digits")
    reason: str = Field(..., min_length=2, max_length=300, description="Return reason in user's words")


class SearchProductsArgs(BaseModel):
    """Search the catalog. Use filters when the user mentions category, price or color."""

    query: str = Field(..., min_length=1, max_length=200, description="Search query (ru or uz)")
    category: str | None = Field(None, description=f"One of: {', '.join(CATEGORY_BY_SLUG)}")
    max_price: int | None = Field(None, gt=0, description="Max price, UZS")
    min_price: int | None = Field(None, ge=0, description="Min price, UZS")
    color: str | None = Field(None, description="Color in Russian, e.g. 'чёрный'")


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "get_order_status": GetOrderStatusArgs,
    "create_return": CreateReturnArgs,
    "search_products": SearchProductsArgs,
}


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style ``tools`` list (used by vLLM tool calling and the chat template)."""
    schemas = []
    for name, model in TOOL_ARGS.items():
        params = model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        schemas.append(
            {
                "type": "function",
                "function": {"name": name, "description": (model.__doc__ or "").strip(), "parameters": params},
            }
        )
    return schemas


class ToolError(Exception):
    """Raised for invalid tool name or arguments (reported back to the LLM)."""


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


class MarketTools:
    """Tool implementations bound to a DB and a (simulated) current date."""

    def __init__(self, db_path: str | Path, today: date, dry_run: bool = False) -> None:
        """Args:
        db_path: SQLite DB created by ``make synth``.
        today: current date for return-window checks (simulated in the demo).
        dry_run: don't persist returns (used when generating training dialogs).
        """
        self.db_path = Path(db_path)
        self.today = today
        self.dry_run = dry_run
        self._bm25: Any = None
        self._product_rows: list[sqlite3.Row] = []

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ---------------------------------------------------------------- dispatch

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments and run a tool. Raises :class:`ToolError` on bad input."""
        if name not in TOOL_ARGS:
            raise ToolError(f"unknown tool {name!r}")
        try:
            args = TOOL_ARGS[name].model_validate(arguments)
        except ValidationError as e:
            raise ToolError(f"invalid arguments for {name}: {e.errors(include_url=False)}") from e
        handler: Callable[[Any], dict[str, Any]] = getattr(self, name)
        return handler(args)

    # ---------------------------------------------------------------- tools

    def get_order_status(self, args: GetOrderStatusArgs) -> dict[str, Any]:
        """Order status, product, ETA and pickup point."""
        with self._connect() as con:
            row = con.execute(
                "SELECT o.*, p.title_ru, p.title_uz, p.category FROM orders o "
                "JOIN products p USING(product_id) WHERE order_id = ?",
                (args.order_id,),
            ).fetchone()
        if row is None:
            return {"error": "not_found", "order_id": args.order_id}
        status_ru, status_uz = ORDER_STATUSES[row["status"]]
        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "status_ru": status_ru,
            "status_uz": status_uz,
            "product_ru": row["title_ru"],
            "product_uz": row["title_uz"],
            "eta": row["eta"],
            "delivered_at": row["delivered_at"],
            "pickup_point": row["pickup_point"],
        }

    def check_return(self, order_id: str) -> dict[str, Any]:
        """Eligibility check without side effects (used to explain refusals before confirming)."""
        with self._connect() as con:
            row = con.execute(
                "SELECT o.status, o.delivered_at, p.category FROM orders o JOIN products p USING(product_id) "
                "WHERE order_id = ?",
                (order_id,),
            ).fetchone()
            existing = con.execute("SELECT return_id FROM returns WHERE order_id = ?", (order_id,)).fetchone()
        if row is None:
            return {"eligible": False, "reason": "not_found"}
        if existing is not None:
            return {"eligible": False, "reason": "already_requested", "return_id": existing["return_id"]}
        if row["status"] != "delivered" or not row["delivered_at"]:
            return {"eligible": False, "reason": "not_delivered", "order_status": row["status"]}
        if not CATEGORY_BY_SLUG[row["category"]].returnable:
            return {"eligible": False, "reason": "non_returnable_category", "category": row["category"]}
        days = (self.today - date.fromisoformat(row["delivered_at"])).days
        if days > RETURN_DAYS:
            return {"eligible": False, "reason": "period_expired", "days_since_delivery": days}
        return {"eligible": True, "days_left": RETURN_DAYS - days}

    def create_return(self, args: CreateReturnArgs) -> dict[str, Any]:
        """Create a return if the order is eligible."""
        check = self.check_return(args.order_id)
        if not check["eligible"]:
            return {"error": "not_eligible", **check}
        return_id = "R" + uuid.uuid5(uuid.NAMESPACE_URL, args.order_id).hex[:8].upper()
        if not self.dry_run:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO returns VALUES (?,?,?,?)",
                    (return_id, args.order_id, args.reason, self.today.isoformat()),
                )
        lo, hi = REFUND_DAYS
        return {"return_id": return_id, "status": "created", "refund_business_days": f"{lo}-{hi}"}

    def search_products(self, args: SearchProductsArgs) -> dict[str, Any]:
        """BM25 over ru/uz titles + SQL-style filters; top-5 in-stock items."""
        from rank_bm25 import BM25Okapi

        if self._bm25 is None:
            with self._connect() as con:
                self._product_rows = con.execute("SELECT * FROM products").fetchall()
            corpus = [_tokens(f"{r['title_ru']} {r['title_uz']} {r['brand']}") for r in self._product_rows]
            self._bm25 = BM25Okapi(corpus)
        scores = self._bm25.get_scores(_tokens(args.query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        items = []
        for i in ranked:
            r = self._product_rows[i]
            if scores[i] <= 0 or len(items) >= 5:
                break
            if not r["in_stock"]:
                continue
            if args.category and r["category"] != args.category:
                continue
            if args.max_price and r["price"] > args.max_price:
                continue
            if args.min_price and r["price"] < args.min_price:
                continue
            if args.color and args.color.lower() not in r["color"]:
                continue
            items.append(
                {
                    "product_id": r["product_id"],
                    "title_ru": r["title_ru"],
                    "title_uz": r["title_uz"],
                    "price": r["price"],
                    "rating": r["rating"],
                }
            )
        return {"items": items, "total": len(items)}
