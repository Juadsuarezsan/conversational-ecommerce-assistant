"""End-to-end API tests through the FastAPI TestClient (offline deterministic mode)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_offline_components(api_client: TestClient) -> None:
    body = api_client.get("/health").json()
    assert body["status"] == "ok" and body["llm_enabled"] is False
    assert body["embedding_backend"] == "hash" and body["rerank_backend"] == "lexical"
    assert body["vector_store"] == "in_memory" and body["catalog_size"] == 213
    assert body["model"] == "claude-sonnet-4-5-20250929"


def test_six_turn_conversation(api_client: TestClient) -> None:
    sid = "e2e-1"

    def chat(msg: str) -> dict:  # type: ignore[type-arg]
        r = api_client.post("/api/chat", json={"session_id": sid, "message": msg})
        assert r.status_code == 200, r.text
        return r.json()  # type: ignore[no-any-return]

    t1 = chat("hi")
    assert t1["intent"] == "greeting" and t1["turn"] == 1 and t1["trace_id"]
    t2 = chat("lactose free milk gallon")
    assert t2["intent"] == "product_search" and t2["retrieved_products"][0]["product_id"] == 6
    second = t2["retrieved_products"][1]["product_id"]
    t3 = chat("add 2 of the second one to my cart")
    assert t3["cart_action"] == "add" and t3["cart_snapshot"] == [
        {
            "product_id": second,
            "product_name": t2["retrieved_products"][1]["product_name"],
            "quantity": 2,
            "unit_price_usd": t2["retrieved_products"][1]["price_usd"],
        }
    ]
    t4 = chat("add the almond milk too")
    assert len(t4["cart_snapshot"]) == 2 and t4["cart_snapshot"][1]["product_id"] in (7, 30)
    t5 = chat("remove the almond milk from my cart")
    assert [c["product_id"] for c in t5["cart_snapshot"]] == [second]
    t6 = chat("refund the milk")
    assert t6["intent"] == "refund" and t6["escalated"] is False and t6["turn"] == 6
    cart = api_client.get(f"/api/cart/{sid}").json()
    assert cart["items"][0]["quantity"] == 2 and cart["total_usd"] > 0
    assert t6["cost_usd"] == 0.0 and t6["llm_used"] is False


def test_determinism_across_sessions(api_client: TestClient) -> None:
    a = api_client.post(
        "/api/chat", json={"session_id": "d1", "message": "healthy breakfast for kids"}
    ).json()
    b = api_client.post(
        "/api/chat", json={"session_id": "d2", "message": "healthy breakfast for kids"}
    ).json()
    assert a["response"] == b["response"]
    assert [p["product_id"] for p in a["retrieved_products"]] == [
        p["product_id"] for p in b["retrieved_products"]
    ]


def test_validation_edge_cases(api_client: TestClient) -> None:
    assert api_client.post("/api/chat", json={"session_id": "", "message": "x"}).status_code == 422
    assert api_client.post("/api/chat", json={"session_id": "s", "message": ""}).status_code == 422
    assert (
        api_client.post("/api/chat", json={"session_id": "s", "message": "x" * 2001}).status_code
        == 422
    )
    assert api_client.post("/api/chat", json={"session_id": "s"}).status_code == 422
    r = api_client.post(
        "/api/chat", content="{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 422 and "detail" in r.json()
    assert api_client.get("/api/products/search", params={"q": ""}).status_code == 422
    assert api_client.get("/api/products/search", params={"q": "milk", "k": 0}).status_code == 422
    assert api_client.post("/api/cart/s/items", json={"product_id": 0}).status_code == 422
    assert api_client.post("/api/cart/s/items", json={"product_id": 999999}).status_code == 404


def test_cart_endpoints_and_search(api_client: TestClient) -> None:
    r = api_client.post("/api/cart/c1/items", json={"product_id": 1, "quantity": 3}).json()
    assert r["items"][0]["product_name"].startswith("Heinz") and r["total_usd"] == 12.87
    r = api_client.delete("/api/cart/c1/items/1").json()
    assert r["items"] == [] and r["total_usd"] == 0.0
    api_client.post("/api/cart/c1/items", json={"product_id": 2})
    assert api_client.post("/api/cart/c1/clear").json() == {"cleared": True, "session_id": "c1"}
    assert api_client.get("/api/cart/c1").json()["items"] == []
    hits = api_client.get("/api/products/search", params={"q": "Heinz ketchup", "k": 3}).json()
    assert len(hits) == 3 and hits[0]["product_id"] in (1, 2) and hits[0]["source"] == "rerank"


def test_cors_headers_and_rate_limit_headers(api_client: TestClient) -> None:
    r = api_client.options(
        "/api/chat",
        headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8501"
    r = api_client.options(
        "/api/chat",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in r.headers


def test_internal_error_becomes_500(api_client: TestClient, mocker) -> None:  # type: ignore[no-untyped-def]
    mocker.patch("src.api.main.run_turn", side_effect=RuntimeError("boom"))
    r = api_client.post("/api/chat", json={"session_id": "x", "message": "hi"})
    assert r.status_code == 500 and "boom" not in r.text
