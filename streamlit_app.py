"""Streamlit front-end for the assistant (talks to the FastAPI backend over HTTP).

    pip install -e ".[ui]"
    API_URL=http://localhost:8000 streamlit run streamlit_app.py

The page never imports the agent: it only calls ``POST /api/chat`` and the cart
endpoints, so it deploys to Streamlit Community Cloud pointing at any backend.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT_SECONDS", "20"))
PREBAKED = [
    "lactose free milk gallon",
    "add 2 of the second one to my cart",
    "healthy breakfast for kids",
    "top rated cereals under $5",
    "refund my $200 order from last week",
    "where is my order #1001",
]

st.set_page_config(page_title="Grocery Assistant", page_icon="🛒", layout="wide")
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2f6; font-size:12px; margin-right:4px; }
    .esc { background:#fff4e5; color:#b54708; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = f"st-{uuid.uuid4().hex[:8]}"
if "turns" not in st.session_state:
    st.session_state.turns = []


def call_chat(message: str) -> dict[str, Any]:
    """POST one turn to the API."""
    resp = httpx.post(
        f"{API_URL}/api/chat",
        json={
            "session_id": st.session_state.session_id,
            "user_id": "streamlit",
            "message": message,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def fetch_cart() -> dict[str, Any]:
    """GET the cart of the current session."""
    resp = httpx.get(f"{API_URL}/api/cart/{st.session_state.session_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def health() -> dict[str, Any] | None:
    """GET /health; ``None`` when the backend is unreachable."""
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=5.0)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data
    except httpx.HTTPError:
        return None


def send(message: str) -> None:
    """Send a message, store the turn and surface errors in the UI."""
    with st.spinner("Thinking…"):
        try:
            data = call_chat(message)
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text[:300]}")
            return
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the API at {API_URL}: {exc}")
            return
    st.session_state.turns.append({"user": message, "bot": data})


with st.sidebar:
    st.title("Grocery Assistant")
    st.caption("Hybrid retrieval + LangGraph cart agent")
    st.link_button(
        "GitHub repo", "https://github.com/Juadsuarezsan/conversational-ecommerce-assistant"
    )
    info = health()
    if info is None:
        st.error(f"Backend unreachable at {API_URL}")
    else:
        mode = "LLM enabled" if info.get("llm_enabled") else "deterministic fallback (no LLM)"
        st.success(f"API ok · {mode}")
        st.caption(
            f"model {info.get('model')} · embeddings {info.get('embedding_backend')} · "
            f"rerank {info.get('rerank_backend')} · store {info.get('vector_store')}"
        )
    st.subheader("Try one")
    for q in PREBAKED:
        if st.button(q, use_container_width=True):
            send(q)
    if st.button("New session", type="secondary", use_container_width=True):
        st.session_state.session_id = f"st-{uuid.uuid4().hex[:8]}"
        st.session_state.turns = []
        st.rerun()

col_chat, col_cart = st.columns([3, 1])

with col_chat:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.write(turn["user"])
        bot = turn["bot"]
        with st.chat_message("assistant"):
            st.write(bot["response"])
            badges = f"<span class='badge'>intent: {bot['intent']}</span>"
            if bot.get("cart_action"):
                badges += f"<span class='badge'>cart: {bot['cart_action']}</span>"
            if bot.get("escalated"):
                badges += "<span class='badge esc'>escalated to human</span>"
            badges += f"<span class='badge'>{bot['latency_ms']} ms</span>"
            badges += f"<span class='badge'>${bot.get('cost_usd', 0):.4f}</span>"
            st.markdown(badges, unsafe_allow_html=True)
            if bot.get("retrieved_products"):
                st.table(
                    [
                        {
                            "pid": p["product_id"],
                            "product": p["product_name"],
                            "aisle": p["aisle"],
                            "price": p.get("price_usd"),
                            "rating": p.get("avg_rating"),
                        }
                        for p in bot["retrieved_products"][:5]
                    ]
                )
    prompt = st.chat_input("Ask for products, manage your cart, request a refund…")
    if prompt:
        send(prompt)
        st.rerun()

with col_cart:
    st.subheader("Cart")
    try:
        cart = fetch_cart()
    except httpx.HTTPError:
        cart = {"items": [], "total_usd": 0.0}
    if not cart["items"]:
        st.caption("Empty")
    for item in cart["items"]:
        st.write(f"[pid:{item['product_id']}] {item['product_name']} x {item['quantity']}")
    st.metric("Total", f"${cart['total_usd']:.2f}")
