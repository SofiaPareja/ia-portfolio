---
title: "De prompts sueltos a bots estructurados: LLMs con LangChain, RAG y salidas en JSON"
date: 2025-10-18
---

# De prompts sueltos a bots estructurados: LLMs con LangChain, RAG y salidas en JSON

## Contexto

En esta tarea trabajé con **LangChain** y modelos de OpenAI para pasar de interacciones simples tipo *“prompt suelto → respuesta”* a **pipelines estructurados**, reutilizables y observables.  
El foco estuvo en:

- Entender el impacto de **parámetros de decodificación** (temperature, max_tokens, top_p).
- Diseñar **plantillas de prompt** con `ChatPromptTemplate` y el operador `|` (LCEL).
- Obtener **salidas estructuradas** usando `with_structured_output(...)` y modelos Pydantic.
- Medir **latencia/tokens** como base de observabilidad.
- Implementar un **chatbot de soporte** que combina **FAQ local + WebSearch** (Tavily) y devuelve una respuesta **tipada en JSON**.

La consigna se enmarca en la Unidad 4 (LLMs & LangChain) y prepara el terreno para agentes más complejos con RAG y tools.

---

## Objetivos

- Instanciar un modelo de chat con `ChatOpenAI` y explorar sus parámetros (`temperature`, `max_tokens`, `top_p`).
- Diseñar **prompts reutilizables** con `ChatPromptTemplate` y encadenarlos con el LLM usando LCEL (`prompt | llm`).
- Utilizar `with_structured_output(...)` para obtener **respuestas validables** vía Pydantic (sin parseo frágil).
- Implementar mini-tareas guiadas: traducción determinista, resúmenes y clasificación de sentimiento (zero-shot vs few-shot).
- Construir un **RAG básico con FAISS** y embeddings de OpenAI.
- Desarrollar un **chatbot de soporte “FAQ + WebSearch”** con salida estructurada (`answer`, `sources`, `confidence`).

---

## Actividades (con tiempos estimados)

| Actividad                                            | Tiempo | Resultado esperado                                                |
|------------------------------------------------------|:------:|-------------------------------------------------------------------|
| Setup de LangChain, LangSmith y modelo base          |  20m   | LLM funcional y entorno de tracing configurado                   |
| Experimentos con `temperature`, `max_tokens`, `top_p`|  30m   | Observaciones sobre claridad vs creatividad                      |
| Diseño de plantillas con `ChatPromptTemplate`        |  35m   | Prompts reutilizables (zero-shot y few-shot)                      |
| Salida estructurada con Pydantic                     |  30m   | Resúmenes y traducciones en JSON tipado                           |
| Experimentos de resúmenes (single-doc y map-reduce)  |  30m   | Cadena de summary chunk → reduce y análisis de calidad           |
| RAG básico con FAISS y prompts “context-only”        |  40m   | Pipeline de recuperación + generación con control de alucinación |
| Implementación del chatbot “FAQ + WebSearch”         |  60m   | Función `answer_question` con contrato Pydantic                  |
| Documentación y reflexión                            |  30m   | Entrada de portafolio coherente y alineada a la rúbrica          |

---

## Desarrollo

### 1. Primeros pasos con ChatOpenAI y parámetros de decodificación

Comencé instanciando un `ChatOpenAI` con el modelo `gpt-5-mini` y probé prompts simples (“Definí ‘Transformer’ en 1 vs 3 oraciones”) para observar:

- **1 oración** → respuesta más condensada, menor cantidad de tokens.
- **3 oraciones** → explicación más técnica y detallada, mayor coste en tokens.

Luego varié la **temperature** (`0.0`, `0.5`, `0.9`) sobre prompts como:

- “Escribí un tuit (<=20 palabras) celebrando un paper de IA.”
- “Dame 3 bullets concisos sobre ventajas de los Transformers.”
- “Escribí un haiku sobre evaluación de modelos.”

Conclusiones:

- `temperature=0.0`: máximo **determinismo**, estilo más rígido y técnico. Ideal para tareas *cerradas* (definiciones, formatos estrictos).
- `temperature≈0.5`: buen compromiso entre **claridad y creatividad**, mantiene estructura pero introduce variación razonable.
- `temperature≈0.9`: estilo más creativo y metafórico; útil para textos expresivos, pero menos adecuado cuando hay una única respuesta correcta.

Reflexión técnica: en tareas de evaluación o clasificación, temperaturas altas pueden introducir ruido o desvíos del formato deseado; para producción, conviene **temperaturas bajas y `max_tokens` razonables**.

---

### 2. De texto suelto a plantillas con `ChatPromptTemplate` + LCEL

Para mejorar la **reutilización** y el control de formato, reemplacé prompts “en bruto” por plantillas:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Sos un asistente conciso, exacto y profesional."),
    ("human",  "Explicá {tema} en <= 3 oraciones, con un ejemplo real.")
])

chain = prompt | llm  # LCEL: prompt → LLM
respuesta = chain.invoke({"tema": "atención multi-cabeza"}).content
```
Luego incorporé few-shot:

```python
    prompt_fewshot = ChatPromptTemplate.from_messages([
        ("system", "Sos un asistente conciso, exacto y profesional."),
        ("human", "Explicá regularización L2 en <= 3 oraciones, con un ejemplo real."),
        ("ai",    "La regularización L2 penaliza pesos grandes para reducir sobreajuste. "
                "Agrega λ·||w||² al loss, empujando los parámetros hacia cero sin anularlos. "
                "Ejemplo: en una regresión de precios de casas, L2 evita que un feature ruidoso domine el modelo."),
        ("human", "Explicá {tema} en <= 3 oraciones, con un ejemplo real.")
    ])
    chain_fewshot = prompt_fewshot | llm

```

!!! note "Observaciones"
    - Zero-shot fue suficiente para explicaciones generales, pero la forma de la respuesta variaba más.
    - Few-shot fijó el estilo (estructura “definición + mecanismo + ejemplo”) y generó salidas más homogéneas entre temas.

**Criterio general:**

- Zero-shot → cuando la tarea está bien definida y necesito flexibilidad.
- Few-shot → cuando necesito copiar un formato/tono muy específico o la tarea es ambigua.

### 3. Salida estructurada con Pydantic (with_structured_output)

Para evitar parsear JSON “a mano”, utilicé Pydantic como contrato de salida. Ejemplo: resumen de riesgos de prompt injection en bullets:
```python
    from typing import List
    from pydantic import BaseModel, Field

    class Resumen(BaseModel):
        title: str
        bullets: List[str] = Field(..., min_items=3, max_items=3)

    llm_json = llm.with_structured_output(Resumen)

    pedido = "Resumí en 3 bullets los riesgos de la 'prompt injection' en LLM apps."
    res = llm_json.invoke(pedido)
```

**Ventajas frente al parseo manual:**

- El modelo devuelve directamente un objeto tipado (Resumen), sin tener que hacer json.loads(...) sobre un string potencialmente inválido.
- Se validan tipos y cardinalidades (exactamente 3 bullets).
- En producción, esto se traduce en contratos claros de integración (APIs, pipelines de datos).

Más adelante reutilicé el patrón para un traductor determinista:
```python 
    class Traduccion(BaseModel):
        text: str
        lang: str

    llm_det = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    traductor = llm_det.with_structured_output(Traduccion)

    salida = traductor.invoke("Traducí al portugués: 'Excelente trabajo del equipo'.")
```
Aquí, la combinación `temperature=0` + esquema Pydantic ayudó a minimizar “alucinaciones” de formato y contenido.

---
### 4. Zero-shot vs Few-shot en clasificación de sentimiento

Implementé dos plantillas:

- Zero-shot:
> “Clasificá el sentimiento de este texto como POS, NEG o NEU: …”

- Few-shot con 2 ejemplos anotados (POS, NEG) y luego el texto objetivo.

Resultados:

- En zero-shot, el modelo a veces respondía con frases (“El sentimiento es positivo”) o mezclaba explicación + etiqueta, lo que complica el uso automático.
- En few-shot, la salida se estabilizó en etiquetas cortas (POS, NEG, NEU), siguiendo el patrón de los ejemplos.

Conclusión: para tareas de clasificación en producción, conviene:

- Few-shot bien diseñado.
- Temperature baja (≈ 0.0–0.2).
- Posiblemente combinarlo con salida estructurada (Literal["POS","NEG","NEU"]).

### 5. Resúmenes: single-doc vs map-reduce
Probé la estrategia map-reduce sobre un texto largo:

1. Split en chunks con RecursiveCharacterTextSplitter.
2. Resumen parcial de cada chunk en 2–3 bullets.
3. Reducción final con otro prompt que consolida bullets redundantes:

```python
    chunk_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Resumí el siguiente fragmento en 2–3 bullets, claros y factuales."),
        ("human", "{input}")
    ])

    bullets = [ (chunk_summary_prompt | llm).invoke({"input": c}).content for c in chunks ]

    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", "Consolidá bullets redundantes y producí un resumen único y breve."),
        ("human", "Bullets:\n{bullets}\n\nResumen final (<=120 tokens):")
    ])

    final = (reduce_prompt | llm).invoke({"bullets": "\n".join(bullets)}).content
```
Comparación:

- Resumen directo (sin split) funciona si el texto entra en el límite de contexto.
- Map-reduce escala mejor para textos largos y permite controlar mejor el coste en tokens, a costa de perder algo de continuidad global si los chunks son muy pequeños o el chunk_overlap es bajo.

---

### 6. Extracción de información estructurada (entidades y campos clave)
Definí un esquema para extraer entidades de un texto:
```python
    from typing import List, Optional
    from pydantic import BaseModel

    class Entidad(BaseModel):
        tipo: str   # 'ORG', 'PER', 'LOC'
        valor: str

    class ExtractInfo(BaseModel):
        titulo: Optional[str]
        fecha: Optional[str]
        entidades: List[Entidad]

    extractor = llm.with_structured_output(ExtractInfo)
    texto = "OpenAI anunció una colaboración con la Universidad Católica del Uruguay en Montevideo el 05/11/2025."
    extractor.invoke(f"Extraé titulo, fecha y entidades (ORG/PER/LOC) del siguiente texto:\n\n{texto}")
```
Esto permitió discutir:

- Cómo flexibilizar el esquema (campos opcionales) vs forzar al modelo a inventar (campos obligatorios).
- Fallos típicos: ambigüedad en fechas, clasificación errónea de nombres propios, etc.

### 7. RAG básico con FAISS y contexto controlado
Para experimentar con RAG, creé un mini-corpus de 3 frases:
```python
    docs_raw = [
        "LangChain soporta structured output con Pydantic.",
        "RAG combina recuperación + generación para mejor grounding.",
        "OpenAIEmbeddings facilita embeddings para indexar textos."
    ]
```
Pasos:

1. Transformar a Document.
2. Split + FAISS con OpenAIEmbeddings.
3. Crear retriever (k=4).
4. Construir un prompt estricto:
```python 
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Respondé SOLO con el contexto. Si no alcanza, decí 'No suficiente contexto'."),
        ("human",  "Contexto:\n{context}\n\nPregunta: {input}")
    ])
```
5. Combinar con create_stuff_documents_chain y create_retrieval_chain.

Esto ayudó a:

- Reducir alucinaciones (“solo con el contexto”).
- Ver el efecto de k: valores muy bajos pueden dejar fuera info relevante; valores muy altos introducen ruido y coste en tokens.

---
### 8. Desafío integrador: Chatbot de soporte “FAQ + WebSearch”
La parte final fue diseñar un chatbot de soporte para un “Producto X” que:

- Consulta primero un corpus local de FAQs (RAG).
- Si el contexto es insuficiente, llama a TavilySearchResults para buscar en la web.
- Devuelve siempre una respuesta estructurada con:
    - `answer`: texto al usuario.
    - `sources`: lista de `{title, url}`.
    - `confidence`: `"low" | "medium" | "high"`.

**8.1 Datos y RAG local**

Construí un mini-corpus de 5 FAQs:

- Reinicio de contraseña.
- Planes y precios.
- Soporte técnico y horarios.
- Integraciones (Slack, Teams, Google Drive).
- Seguridad y backups.

Cada entrada se modeló como:
```python
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=entry["content"],
            metadata={"title": entry["title"], "url": entry["url"]}
        )
        for entry in faq_texts
    ]
```
Luego:

- Split con RecursiveCharacterTextSplitter.
- Indexación con FAISS + OpenAIEmbeddings.
- retriever = vectorstore.as_retriever(search_kwargs={"k": 4}).

**8.2 WebSearch con Tavily**
```python
    from langchain_community.tools.tavily_search import TavilySearchResults

    web_search = TavilySearchResults(max_results=3)

    def search_web(question: str):
        results = web_search.invoke({"query": question})
        # Normalizo a {title, url, content}
```
**8.3 Contrato de salida en Pydantic**
```python
    from typing import List, Literal
    from pydantic import BaseModel

    class Source(BaseModel):
        title: str
        url: str

    class SupportResponse(BaseModel):
        answer: str
        sources: List[Source]
        confidence: Literal["low", "medium", "high"]
```
La función principal:
```python
    def answer_question(question: str) -> SupportResponse:
    # 1) RAG local
    local_context, local_sources = build_local_context(question)

    # 2) Heurística: ¿necesito WebSearch?
    use_web = len(local_context.strip()) < 80

    web_context, web_sources = "", []
    if use_web:
        web_context, web_sources = build_web_context(question)

    # 3) Sin contexto útil → respuesta low confidence
    if not local_context.strip() and not web_context.strip():
        return SupportResponse(
            answer="No encontré suficiente información para responder con confianza a tu pregunta.",
            sources=[],
            confidence="low"
        )

    # 4) Prompt de soporte (FAQ + WebSearch)
    final_context_local = local_context if local_context.strip() else "No hay contexto local relevante."
    final_context_web = web_context if web_context.strip() else "No se realizaron búsquedas web o no hubo resultados."

    llm_resp = support_chain.invoke({
        "question": question,
        "local_context": final_context_local,
        "web_context": final_context_web,
    }).content

    all_sources = local_sources + web_sources

    if local_sources and not web_sources:
        confidence = "high"
    elif local_sources and web_sources:
        confidence = "medium"
    else:
        confidence = "medium"

    return SupportResponse(
        answer=llm_resp,
        sources=all_sources,
        confidence=confidence
    )
```

Probé el bot con preguntas como:

- “¿Cómo restablezco mi contraseña?”
- “¿El producto X ofrece cifrado de datos y backups?”
- “¿Producto X tiene integración con Slack?”

En todos los casos, la respuesta viene empaquetada como SupportResponse, lista para serializar a JSON.

---
### Evidencias
- Notebook del assignment con todas las secciones implementadas (setup, parámetros, plantillas, structured output, RAG).
- Código de plantillas (ChatPromptTemplate) y cadenas LCEL (prompt | llm).
- Objetos Pydantic (Resumen, Traduccion, Entidad, ExtractInfo, Source, SupportResponse) como contratos de salida.
- Implementación funcional del chatbot de soporte “FAQ + WebSearch” con la función answer_question.
- Ejemplos de llamadas a answer_question mostrando:
    - Respuestas en lenguaje natural.
    - Fuentes (títulos + URLs) combinando local://... y resultados web.
    - Nivel de confianza (high para solo FAQ, medium cuando entra web search, low cuando no hay contexto).

--- 

### Reflexión

A nivel técnico, la tarea me permitió pasar de un uso ad hoc de LLMs a un enfoque mucho más ingenieril:
- Descubrí el valor de separar prompting, modelo y post-procesado en componentes reutilizables (LCEL).
- `with_structured_output` y Pydantic resultan fundamentales para tener contratos de salida robustos, especialmente en contextos de integración con otros servicios.
- El pequeño pipeline de RAG + WebSearch mostró cómo orquestar múltiples fuentes manteniendo un control razonable sobre alucinaciones y nivel de confianza.

A nivel conceptual, los experimentos con temperature, zero-shot vs few-shot y resúmenes map-reduce reforzaron ideas clave:
- Las decisiones de decodificación y formato son tan importantes como la elección del modelo.
- La explicabilidad y la estructura (JSON, contratos) reducen el “magia negra” alrededor de los LLMs y facilitan su uso en aplicaciones reales.

Si tuviera más tiempo, extendería el chatbot de soporte con:
- Métricas de latencia y token usage registradas sistemáticamente (LangSmith).
- Clasificadores previos para decidir mejor cuándo usar RAG, cuándo WebSearch y cuándo responder directo.
- Una pequeña UI (Gradio/Streamlit) para probar el bot con usuarios reales y ver logs de contexto y fuentes.

### Checklist
- [x] Instanciación de ChatOpenAI y experimentos con temperature, max_tokens, top_p
- [x] Diseño de plantillas con ChatPromptTemplate (zero-shot y few-shot)
- [x] Uso de with_structured_output con Pydantic para salidas en JSON
- [x] Experimentos de resúmenes (single-doc vs map-reduce)
- [x] Clasificación de sentimiento zero-shot vs few-shot
- [x] RAG básico con FAISS y prompt “solo contexto”
- [x] Implementación del chatbot “FAQ + WebSearch” con contrato SupportResponse
- [x] Documentación y reflexión alineadas a la rúbrica del curso