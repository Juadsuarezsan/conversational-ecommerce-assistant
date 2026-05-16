CREATE EXTENSION IF NOT EXISTS vector;

-- Product catalog (denormalized for query speed). Source: Instacart `products.csv`.
CREATE TABLE IF NOT EXISTS products (
    product_id      INTEGER PRIMARY KEY,
    product_name    TEXT NOT NULL,
    aisle_id        INTEGER NOT NULL,
    aisle           TEXT NOT NULL,
    department_id   INTEGER NOT NULL,
    department      TEXT NOT NULL,
    -- enrichments (filled by ingestion step)
    price_usd       NUMERIC(10, 2),
    avg_rating      NUMERIC(3, 2),
    rating_count    INTEGER DEFAULT 0,
    in_stock        BOOLEAN DEFAULT TRUE,
    embedding       vector(1024)
);

CREATE INDEX IF NOT EXISTS products_aisle_idx ON products (aisle_id);
CREATE INDEX IF NOT EXISTS products_dept_idx  ON products (department_id);
CREATE INDEX IF NOT EXISTS products_name_trgm ON products USING gin (product_name gin_trgm_ops);
-- ivfflat over 1024-dim embeddings. lists=100 is reasonable for 50K rows.
CREATE INDEX IF NOT EXISTS products_embedding_idx
    ON products USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Sessions, carts, orders for the conversational state
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    state           JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS cart_items (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    added_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS cart_items_session_idx ON cart_items (session_id);

CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    total_usd       NUMERIC(10, 2) NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'paid', 'shipped', 'delivered', 'refunded')),
    placed_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    items           JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_user_idx ON orders (user_id, placed_at DESC);

-- Audit log: every agent decision is recorded for replay + analysis
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    turn            INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    intent          TEXT NOT NULL,
    confidence      NUMERIC(4, 3),
    query           TEXT NOT NULL,
    retrieved_ids   INTEGER[] DEFAULT '{}',
    response        TEXT,
    escalated       BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ms      INTEGER,
    cost_usd        NUMERIC(8, 5)
);
CREATE INDEX IF NOT EXISTS audit_log_session_idx ON audit_log (session_id, turn);
