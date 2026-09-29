"""Public Pydantic schemas shared by the API, the agent and the eval harness."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["product_search", "cart_op", "order_status", "refund", "escalate", "greeting"]
CartAction = Literal["view", "add", "remove", "update_quantity", "clear"]
RetrievalSource = Literal["bm25", "dense", "hybrid", "rerank", "structured"]


class Product(BaseModel):
    """A catalog row (Instacart-like schema with price/rating enrichments)."""

    product_id: int
    product_name: str
    aisle: str
    department: str
    price_usd: float | None = None
    avg_rating: float | None = None
    rating_count: int = 0
    in_stock: bool = True


class RetrievedProduct(Product):
    """A product returned by a retriever, annotated with its score and origin."""

    score: float
    source: RetrievalSource = "hybrid"


class CartItem(BaseModel):
    """One line of a shopping cart."""

    product_id: int
    product_name: str
    quantity: int = Field(..., gt=0)
    unit_price_usd: float


class CartOperation(BaseModel):
    """A cart mutation parsed from a user message."""

    action: CartAction
    product_id: int | None = None
    quantity: int = Field(default=1, ge=0)
    reference: str = ""


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat``."""

    session_id: str = Field(..., min_length=1, max_length=128)
    user_id: str = Field(default="demo-user", max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)


class IntentDecision(BaseModel):
    """Structured output of the Intent Router."""

    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    extracted_filters: dict[str, str | float | int | bool] = Field(default_factory=dict)
    rationale: str = ""


class Usage(BaseModel):
    """Token usage and cost accumulated during one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def add(self, other: Usage) -> Usage:
        """Return the element-wise sum of two usages."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
            llm_calls=self.llm_calls + other.llm_calls,
        )


class ChatResponse(BaseModel):
    """Body returned by ``POST /api/chat``."""

    session_id: str
    turn: int
    intent: Intent
    confidence: float
    response: str
    retrieved_products: list[RetrievedProduct] = Field(default_factory=list)
    cart_snapshot: list[CartItem] = Field(default_factory=list)
    cart_action: CartAction | None = None
    escalated: bool = False
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_used: bool = False
    trace_id: str | None = None


class CartView(BaseModel):
    """Body returned by the cart endpoints."""

    session_id: str
    items: list[CartItem]
    total_usd: float


class CartAddRequest(BaseModel):
    """Body of ``POST /api/cart/{session_id}/items``."""

    product_id: int = Field(..., ge=1)
    quantity: int = Field(default=1, ge=1, le=99)


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: Literal["ok"]
    version: str
    model: str
    llm_enabled: bool
    embedding_backend: str
    rerank_backend: str
    vector_store: str
    cart_backend: str
    session_backend: str
    catalog_size: int
