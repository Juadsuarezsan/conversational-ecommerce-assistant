# Post de LinkedIn (borrador)

Publiqué el primer proyecto de mi portafolio de AI Engineering: un asistente conversacional de
e-commerce que busca productos con retrieval híbrido (BM25 + vectores + RRF + reranking),
opera el carrito siguiendo la conversación ("agrega dos del segundo", "quita la leche de
almendras", "que sean tres") y aplica una política de reembolsos explícita: más de $100 o
categoría sensible, lo revisa un humano.

Lo que más me importaba no era el chat sino poder medirlo:

- 100 consultas etiquetadas a mano (30 directas, 30 semánticas, 20 comparativas, 20 multi-turno).
- `python -m eval.run` regenera la tabla completa: ablación BM25 / denso / híbrido / +rerank, 20 conversaciones reproducidas por el agente, latencias y costo.
- Cada corrida queda versionada con qué componentes la produjeron. Sin llaves de API el sistema corre con fallbacks deterministas y el informe lo dice en negrita; las celdas que necesitan Claude, Voyage o Docker dicen "pendiente" en vez de inventar un número.

Primera corrida (fallback determinista, catálogo sintético): nDCG@10 0,992 en búsquedas
directas, 0,948 en comparativas y 0,586 en semánticas. Ese último número es el hallazgo: sin
embedding neuronal, "cosas para hacer guacamole" no encuentra el aguacate. El análisis de los
diez peores casos está en el repo, con la causa y la corrección de cada uno.

Stack: Python 3.11, FastAPI, LangGraph, Claude Sonnet 4.5 (id fijado), Qdrant / pgvector /
Chroma detrás de un mismo protocolo, Postgres para carritos, Redis para sesión, pytest con 97 %
de cobertura, mypy estricto, CI con gitleaks.

Repo: https://github.com/Juadsuarezsan/conversational-ecommerce-assistant

#AIEngineering #RAG #LangGraph #Ecommerce #Python
