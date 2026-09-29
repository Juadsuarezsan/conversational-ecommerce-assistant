# Un asistente de compras que sigue la conversación: retrieval híbrido, LangGraph y una evaluación que no miente

*Borrador para Medium / Dev.to. Repositorio:
https://github.com/Juadsuarezsan/conversational-ecommerce-assistant*

## Por qué otro asistente de e-commerce

Casi todo comercio con catálogo grande ya tiene un chat. Casi ninguno resuelve bien tres cosas
a la vez: entender lo que el cliente quiere cuando no usa las palabras del catálogo, mantener
un carrito que siga la conversación ("agrega dos del segundo", "quita la leche de almendras"),
y aplicar reglas de devolución que un equipo de cumplimiento pueda leer. Y casi ninguno puede
demostrar con números que lo hace.

Este proyecto es una implementación de referencia de esas tres piezas, con una cuarta que
considero la más importante: una evaluación reproducible con un comando, cuyos resultados
quedan versionados en el repositorio y etiquetados con qué componentes los produjeron. En este
artículo cuento las decisiones que tomé, lo que midió la primera corrida y, sobre todo, en qué
falla.

## La arquitectura en una frase

Una petición `POST /api/chat` entra a FastAPI, carga el estado de la sesión (historial y
productos mostrados en la última búsqueda) desde Redis o memoria, y ejecuta un grafo de
LangGraph con siete nodos: clasificar intención, recuperar productos, operar el carrito,
evaluar reembolsos, consultar órdenes, escalar a humano y sintetizar la respuesta. Cada nodo
registra su estado de entrada y salida con el `trace_id` de la petición, y cada respuesta
reporta tokens y costo en dólares.

La recuperación es híbrida: BM25 sobre nombre + pasillo + departamento, vectores densos en
Qdrant, pgvector, Chroma o memoria, fusión por Reciprocal Rank Fusion con `k = 60`, reranking
(Cohere v3, un cross-encoder local o un fallback léxico) y, al final, filtros estructurados de
precio, calificación, disponibilidad y dieta.

## Decisión 1: cada dependencia de pago tiene un doble determinista

Claude clasifica intenciones y redacta respuestas; Voyage AI genera embeddings; Cohere
reordena. Ninguna de las tres está disponible en el entorno donde desarrollé y evalué el
proyecto, y no quería que eso me impidiera tener un sistema que arranca, responde y se prueba.

La solución fue disciplinada: cada proveedor vive detrás de un `Protocol` y una variable de
entorno elige la implementación. Sin `ANTHROPIC_API_KEY`, el router de intenciones es un
clasificador por palabras clave y el sintetizador una plantilla que siempre cita productos
como `[pid:N]`. Sin `VOYAGE_API_KEY`, los embeddings son un hash determinista de palabras y
trigramas de caracteres, de 1024 dimensiones como `voyage-3`, para que el esquema de la base
vectorial no cambie. Sin `COHERE_API_KEY`, el reranker mide cobertura de tokens de la consulta.

La regla que acompaña a esa decisión es la que evita el autoengaño: **cada corrida de
evaluación escribe qué componentes la produjeron** y el informe etiqueta en negrita cuando son
fallbacks. Los números del fallback miden el fallback, no el sistema de producción.

## Decisión 2: no instalar torch por defecto

`sentence-transformers` estaba en las dependencias base y el entorno virtual pesaba 6,5 GB.
Moverlo a un extra opcional (`pip install -e ".[ml]"`) e importarlo de forma perezosa dentro de
las clases que lo usan bajó la instalación a 700 MB. CI instala en un minuto y las pruebas
corren en 20 segundos. Cuando hay un modelo local disponible, basta con dos variables de
entorno para activarlo; no cambia una línea de código.

## Decisión 3: el carrito se opera con símbolos, no con embeddings

"Agrega dos del segundo" tiene que resolverse contra la lista que el usuario acaba de ver.
"Quita la leche de almendras" tiene que encontrar la línea correcta del carrito aunque haya
otra leche. "Que sean tres" tiene que actualizar la única línea que hay.

Implementé un analizador determinista (`src/agents/cart_ops.py`) que reconoce el verbo (agregar,
quitar, cambiar cantidad, ver, vaciar), la cantidad, los ordinales, los superlativos ("el más
barato") y los pronombres, y resuelve la referencia contra tres fuentes en orden: los productos
mostrados en el turno anterior, el carrito y el catálogo vía BM25. La coincidencia por nombre
exige que el candidato cubra al menos la mitad de los tokens descriptivos del usuario, de modo
que "leche de almendras" nunca se resuelve a "leche deslactosada" solo porque ambas dicen
leche. Es la parte del sistema que más pruebas unitarias tiene, y también la que más veces
corregí durante el desarrollo: la primera versión eliminaba el producto equivocado y ordenaba
por precio toda la lista fusionada en vez de solo los cafés cuando pedías "el café más barato".

## Decisión 4: filtros después del reranking, con una compuerta léxica

Las consultas comparativas ("cereales mejor calificados por menos de $5") combinan semántica
con restricciones. Aplicar el filtro antes de la búsqueda descarta candidatos relevantes;
ordenar por precio la lista fusionada completa devuelve arena para gatos como el café más
barato. La versión final aplica el filtro sobre los candidatos rerankeados que comparten al
menos un token descriptivo con la consulta, luego sobre la lista fusionada, y si nada
sobrevive consulta el catálogo con SQL parametrizado (o el filtro en memoria) y lo dice en la
respuesta. Ese cambio movió el nDCG@10 de las consultas comparativas de 0,754 a 0,948.

## Decisión 5: una política de escalación que cabe en dos líneas

Un reembolso se aprueba automáticamente solo si el monto es menor o igual a $100 **y** ningún
departamento involucrado está en {alcohol, tabaco, farmacia, bebés}. El monto sale, en orden,
de una cifra explícita en el mensaje, de la orden referida por número, del producto nombrado o
del total del carrito. Cualquier petición explícita de hablar con un humano escala sin mirar el
monto. Está documentada en `docs/decisions.md` y probada con casos de borde. Lo valioso no es
la regla sino que sea legible: un equipo legal puede discutir el umbral sin abrir el código del
agente.

## La evaluación: cien consultas etiquetadas a mano

El spec pedía cien consultas en cuatro categorías: treinta de búsqueda directa ("Heinz ketchup
397 gramos"), treinta semánticas ("desayuno saludable para niños"), veinte comparativas y veinte
multi-turno. Las etiqueté a mano contra el catálogo sintético de 213 productos que el proyecto
usa mientras no haya acceso al dataset de Instacart, con relevancia de 1 a 3 por producto. Las
multi-turno llevan además los turnos previos que hay que reproducir y un bloque de
expectativas verificables: intención esperada, si debió escalar, qué debe haber en el carrito,
con qué cantidad, qué producto debe citar la respuesta.

`python -m eval.run` corre cuatro estrategias (solo BM25, solo denso, híbrido, híbrido con
reranking y filtros) sobre las ochenta consultas de un turno, reproduce las veinte
conversaciones por el agente completo, intenta el baseline zero-shot, RAGAS y el juez LLM (que
quedan marcados como pendientes sin llave), y escribe un JSON en `eval/runs/` y un
`eval/RESULTS.md` regenerado por completo.

## Lo que midió la primera corrida

Con los fallbacks deterministas, sobre el catálogo sintético:

| Estrategia | P@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 solo (baseline sin IA) | 0,383 | 0,861 | 0,825 | 0,773 |
| Denso solo (hash) | 0,398 | 0,874 | 0,848 | 0,790 |
| Híbrido RRF | 0,405 | 0,877 | 0,848 | 0,793 |
| Híbrido + rerank + filtros | 0,393 | 0,867 | 0,866 | 0,829 |

Por categoría, el sistema completo alcanza 0,992 de nDCG@10 en búsquedas directas, 0,948 en
comparativas y 0,586 en semánticas. Las veinte conversaciones multi-turno pasan sus veinte
verificaciones, con 2,7 turnos promedio y 39 ms de p95 sin LLM.

Hay que leer esa tabla con cuidado. Precision@5 baja de 0,405 a 0,393 con el reranking mientras
MRR y nDCG suben: el filtro recorta la lista a los productos que cumplen la restricción, así que
"cereales por menos de $5" devuelve un solo producto (el correcto) y P@5 lo castiga por no
devolver cinco. El mejor resumen es nDCG@10, y ahí cada capa aporta.

## En qué falla, y por qué es lo más útil del informe

Los diez peores casos son todos semánticos. "Cosas para hacer guacamole" devuelve cervezas y
huevos: no hay ningún token compartido entre la consulta y "Avocado Hass", y un hash de
palabras no sabe qué es el guacamole. "Yogur sin lácteos" devuelve leches deslactosadas porque
"free" aparece en su nombre y el yogur de coco no dice "dairy" por ningún lado. "Algo para tacos
esta noche" pone una pasta de dientes de primero por una colisión de trigramas entre "tonight"
y "toothpaste whitening".

Cada uno de esos casos tiene una hipótesis y una corrección en `docs/error_analysis.md`: unos
necesitan un embedding neuronal (que está a un `pip install` de distancia), otros un atributo
de dieta en el catálogo que el filtro estructurado ya sabe consumir, otros una reescritura de
consulta por el LLM. Lo importante es que el informe distingue lo que falla por diseño del
fallback de lo que fallaría igual con modelos reales.

## Lo que no está hecho

Sin llaves ni Docker en el entorno de desarrollo quedan pendientes, y así aparecen en el
informe: las filas de Qdrant y pgvector en la tabla comparativa, RAGAS, el juez LLM sobre
conversaciones simuladas, el baseline zero-shot con Claude, la latencia y el costo reales por
conversación, las trazas públicas de LangSmith y el despliegue en Streamlit Cloud. El código de
cada una existe, está probado con mocks, y se activa con la llave correspondiente sin cambiar
una línea. Prefiero una celda que diga "pendiente" a un número que no pueda regenerar.

## Qué me llevo

Tres cosas. Primero, que el fallback determinista no es un parche sino una configuración de
primera clase: es lo que corre en CI, lo que hornea el demo estático y lo que el proyecto de
red teaming del portafolio ataca como sistema bajo prueba. Segundo, que la evaluación debe
escribir su propia procedencia; una tabla sin etiqueta es una tabla que en tres semanas nadie
sabrá interpretar. Tercero, que las operaciones de carrito merecen un analizador simbólico
propio: es la parte que más se prueba, la que menos depende del proveedor de turno y la que
más confianza le da al usuario cuando dice "ese" y el sistema entiende cuál.

El repositorio, con el eval set, las corridas y los documentos de decisiones, está en
https://github.com/Juadsuarezsan/conversational-ecommerce-assistant.
