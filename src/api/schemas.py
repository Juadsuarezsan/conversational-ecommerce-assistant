"""Public Pydantic schemas used across the API + agent."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["product_search", "cart_op", "order_status", "refund", "escalate", "greeting"]


class Product(BaseModel):
    product_id: int
    product_name: str
    aisle: str
    department: str
    price_usd: float | None = None
    avg_rating: float | None = None
    rating_count: int = 0
    in_stock: bool = True


class RetrievedProduct(Product):
    score: float
    source: Literal["bm25", "dense", "hybrid", "rerank"] = "hybrid"


class CartItem(BaseModel):
    product_id: int
    product_name: str
    quantity: int = Field(..., gt=0)
    unit_price_usd: float


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    user_id: str = "demo-user"
    message: str = Field(..., min_length=1, max_length=2000)


class IntentDecision(BaseModel):
    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_filters: dict[str, str | float | int | bool] = Field(default_factory=dict)
    rationale: str = ""


class ChatResponse(BaseModel):
    session_id: str
    turn: int
    intent: Intent
    confidence: float
    response: str
    retrieved_products: list[RetrievedProduct] = Field(default_factory=list)
    cart_snapshot: list[CartItem] = Field(default_factory=list)
    escalated: bool = False
    latency_ms: int
    cost_usd: float = 0.0
    trace_id: str | None = None
