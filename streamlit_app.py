"""Streamlit demo for the e-commerce assistant.

Run locally:
    streamlit run streamlit_app.py
Deploy:
    push to GitHub, connect on streamlit.io/cloud
"""
from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="E-commerce Assistant", page_icon="🛒", layout="wide")

st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg, #06070d 0%, #0a0e1c 100%); color: #e7ecf5; }
  .stChatMessage { background: rgba(18,20,28,0.7) !important; border:1px solid rgba(255,255,255,0.08); border-radius:14px; }
  .brand { font-family: 'JetBrains Mono', monospace; letter-spacing:0.2em;
           background: linear-gradient(90deg, #22d3ee, #7c5cff);
           -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='brand'>// P1 · CONVERSATIONAL E-COMMERCE ASSISTANT</div>", unsafe_allow_html=True)
st.title("🛒 Ask the supermarket")
st.caption("Hybrid retrieval (BM25 + dense + RRF + rerank) over an Instacart-like catalog. Multi-turn cart agent.")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"st-{uuid.uuid4().hex[:8]}"
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    st.code(st.session_state.session_id)
    if st.button("New session"):
        st.session_state.session_id = f"st-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

    st.subheader("Try these")
    examples = [
        "Heinz ketchup 397 gram bottle",
        "healthy breakfast for kids",
        "lactose-free milk alternatives",
        "best rated granola bars",
        "add the first one to my cart",
        "what's in my cart?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

    st.subheader("Backend")
    try:
        h = httpx.get(f"{API_URL}/health", timeout=2).json()
        st.success(f"online · LLM: {h['llm_enabled']}")
        st.caption(f"store: {h['vector_store']} · emb: {h['embedding_backend']} · rerank: {h['rerank_backend']}")
    except Exception as e:
        st.error(f"backend offline at {API_URL}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("products"):
            for p in m["products"]:
                price = f"${p['price_usd']:.2f}" if p.get("price_usd") else "—"
                rating = f"★ {p['avg_rating']}" if p.get("avg_rating") else "—"
                st.markdown(f"- **[pid:{p['product_id']}] {p['product_name']}** · {p['aisle']} · {price} · {rating}")

prompt = st.session_state.pop("pending_query", None) or st.chat_input("Ask me anything about the catalog...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = httpx.post(
                    f"{API_URL}/api/chat", timeout=30,
                    json={"session_id": st.session_state.session_id, "message": prompt},
                ).json()
            except Exception as e:
                st.error(f"Error: {e}")
                r = None
        if r:
            st.markdown(r["response"])
            if r.get("retrieved_products"):
                for p in r["retrieved_products"]:
                    price = f"${p['price_usd']:.2f}" if p.get("price_usd") else "—"
                    rating = f"★ {p['avg_rating']}" if p.get("avg_rating") else "—"
                    st.markdown(f"- **[pid:{p['product_id']}] {p['product_name']}** · {p['aisle']} · {price} · {rating}")
            st.caption(
                f"intent: {r['intent']} (conf {r['confidence']:.2f}) · "
                f"latency {r['latency_ms']} ms · escalated: {r['escalated']}"
            )
            st.session_state.messages.append({
                "role": "assistant", "content": r["response"],
                "products": r.get("retrieved_products", []),
            })
