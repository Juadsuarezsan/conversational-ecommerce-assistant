# Resultados de evaluación

Generado por `python -m eval.run` el 2026-09-29T01:32:15+00:00 (commit `01d476d`, Python 3.11.15).

> **Etiqueta de la corrida:** **fallback determinista (embedding por hash / rerank lexico), sin modelo**. Embeddings `hash` y reranker `lexical` son fallbacks deterministas sin modelo neuronal; las filas *dense* y *rerank* miden ese fallback, **no** el sistema de producción (voyage-3 + Cohere Rerank v3 o MiniLM + cross-encoder).
>
> La ground truth está escrita a mano sobre el **catálogo sintético** (213 productos, seed 20260516), no sobre Instacart real (ver `data/eval/README.md`).

## Tabla obligatoria: vector DBs

| Vector DB | Precision@5 | Recall@10 | nDCG@10 | Latencia p95 (dense, ms) | Costo/1K queries |
|---|---|---|---|---|---|
| qdrant | pendiente (requiere Docker: docker compose up) | pendiente (requiere Docker: docker compose up) | pendiente (requiere Docker: docker compose up) | pendiente (requiere Docker: docker compose up) | $0 self-hosted (contenedor local) |
| pgvector | pendiente (requiere Docker: docker compose up) | pendiente (requiere Docker: docker compose up) | pendiente (requiere Docker: docker compose up) | pendiente (requiere Docker: docker compose up) | $0 self-hosted (contenedor local) |
| chroma | 0.398 | 0.874 | 0.792 | 4.0 | $0 (embebido, disco local) |
| in_memory | 0.398 | 0.874 | 0.790 | 17.5 | $0 (proceso local) |

Las filas medidas usan la estrategia *dense* (solo vector) del store indicado, con las mismas embeddings; p95 incluye embed de la query + búsqueda. Costo/1K = infraestructura self-hosted; con `voyage-3` se suman ~$0.12/M tokens de embedding de queries (≈$0.002/1K queries de 15 tokens).

## Ablación de retrieval (store `in_memory`, n=80 queries)

| Estrategia | P@5 | R@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| BM25 solo (baseline sin IA) | 0.383 | 0.861 | 0.825 | 0.773 | 0.2 | 0.3 |
| Dense solo | 0.398 | 0.874 | 0.848 | 0.790 | 17.0 | 17.5 |
| Híbrido BM25 + dense (RRF k=60) | 0.405 | 0.877 | 0.848 | 0.793 | 17.3 | 19.0 |
| Híbrido + rerank + filtros | 0.393 | 0.867 | 0.866 | 0.829 | 0.2 | 0.4 |
| Zero-shot Claude sin retrieval | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | — | — |

### Por categoría (nDCG@10)

| Categoría | BM25 | Dense | Híbrido | + rerank |
|---|---|---|---|---|
| lookup | 0.995 | 0.988 | 0.989 | 0.992 |
| semantic | 0.607 | 0.615 | 0.622 | 0.586 |
| comparative | 0.691 | 0.755 | 0.754 | 0.948 |

## Conversacional (20 conversaciones multi-turn guionizadas)

Etiqueta: **fallback determinista, sin LLM (router heuristico + plantillas)**.

| Métrica | Valor |
|---|---|
| Task completion rate (verificación determinista de carrito/intención) | 1.00 (20/20) |
| Turnos promedio por tarea | 2.70 |
| Latencia p50 / p95 por conversación (ms) | 30.3 / 38.6 |
| Costo promedio por conversación (USD) | 0.0000 (sin LLM: $0) |
| Task completion (LLM-as-judge, 50 conversaciones) | pendiente (requiere ANTHROPIC_API_KEY) |

## RAGAS

| Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---|---|---|
| pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) |

## Los 10 peores casos (híbrido + rerank, por nDCG@10)

| id | query | nDCG@10 | MRR | top-5 devuelto |
|---|---|---|---|---|
| q-semantic-03 | low carb dinner options | 0.000 | 0.000 | 202, 185, 165, 167, 116 |
| q-semantic-10 | dairy free yogurt | 0.000 | 0.000 | 32, 6, 8 |
| q-semantic-25 | things to make guacamole | 0.000 | 0.000 | 208, 209, 41, 42, 191 |
| q-semantic-30 | sandwich supplies for lunch | 0.054 | 0.100 | 9, 97, 70, 158, 156 |
| q-semantic-04 | snacks for a road trip | 0.234 | 0.250 | 3, 129, 152, 128, 127 |
| q-semantic-22 | pantry staples for a beginner cook | 0.272 | 0.333 | 71, 74, 72, 73, 78 |
| q-semantic-12 | high protein snacks | 0.305 | 0.333 | 3, 152, 129, 150, 127 |
| q-semantic-21 | spices for a curry | 0.319 | 1.000 | 71, 72, 74, 73, 176 |
| q-semantic-08 | something to make tacos tonight | 0.400 | 0.333 | 170, 101, 99, 111, 202 |
| q-semantic-06 | quick weekday breakfast | 0.431 | 1.000 | 137, 55, 52, 54, 3 |

El análisis de causas está en `docs/error_analysis.md`.

## Reproducir

```bash
python -m eval.run --stores in_memory,chroma
```

Cada corrida queda en `eval/runs/` y este archivo se regenera por completo.
