---
title: "Orquestando un mini-agente de soporte: LangGraph + RAG + Tools + Memoria"
date: 2025-11-30
---

# Orquestando un mini-agente de soporte: LangGraph + RAG + Tools + Memoria

## Contexto

En esta tarea construí un **agente conversacional con LangGraph** que combina:

- un **LLM** (ChatOpenAI) como motor de razonamiento,
- un **RAG** pequeño sobre documentación de un curso de IA,
- una **tool de estado** para consultar el plan/suscripción de un estudiante,
- y una **memoria ligera** basada en resúmenes de la conversación.

El objetivo fue pasar de un simple `llm.invoke()` de un turno a un **grafo de agentes** que mantiene estado, llama tools cuando las necesita y soporta conversaciones multi-turn.

---

## Objetivos

- Definir un **AgentState** adecuado para conversaciones con memoria resumida.
- Construir un **RAG mínimo** (FAISS + embeddings OpenAI) y exponerlo como tool.
- Implementar una **tool de estado de usuario** (`get_user_plan`) para consultas personalizadas.
- Orquestar todo en un **grafo LangGraph** con nodos `assistant`, `tools` y `memory`.
- Probar **conversaciones de soporte** que combinen RAG + tools en varios turnos.
- Reflexionar sobre **ventajas, limitaciones y extensiones** del diseño.

---

## Actividades (con tiempos estimados)

| Actividad                                         | Tiempo | Resultado esperado                                          |
|--------------------------------------------------|:------:|-------------------------------------------------------------|
| Definir `AgentState` y primer grafo mínimo       |  20m   | Agente que responde usando solo el LLM                      |
| Construcción del RAG (docs IA + FAISS)           |  30m   | `retriever` listo y tool `rag_search` funcional             |
| Implementación de tool de estado `get_user_plan` |  20m   | Consulta ficticia de plan de usuario                        |
| Creación del grafo `assistant ↔ tools ↔ memory`  |  45m   | LangGraph con routing condicional y actualización de summary|
| Pruebas multi-turn y análisis de comportamiento  |  40m   | 3 conversaciones de soporte con uso de distintas tools      |
| (Opcional) UI en Gradio                          |  35m   | Interfaz de chat para probar el agente sin tocar código     |

---

## Desarrollo

### 1. Diseño del estado del agente

Definí un estado explícito para el agente usando `TypedDict`:

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    summary: Optional[str]
```

- `messages`: historial completo de mensajes (HumanMessage, AIMessage, tool calls, etc.).
- `summary`: resumen corto de la conversación, pensado como memoria comprimida para no pasar todo el historial al modelo.

Ventaja: el grafo puede mutar el estado (agregar mensajes, actualizar el resumen) sin que yo tenga que reconstruirlo manualmente en cada llamada.
<!-- TODO Arreglar el ventaja -->

### 2. RAG sobre documentación del curso de IA
Construí un RAG pequeño con 10 textos que describen el curso de IA: duración, unidades, evaluación, proyecto final, carga horaria, requisitos, uso de librerías, etc.

Pipeline:
```python
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    raw_docs = [
        "El curso de IA dura 12 semanas...",
        "La evaluación del curso de IA incluye assignments...",
        ...
    ]
    docs = [Document(page_content=t) for t in raw_docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    emb = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embedding=emb)
    retriever = vs.as_retriever(search_kwargs={"k": 3})
```

Luego lo expuse como tool reutilizable:
```python
    @tool
    def rag_search(question: str) -> str:
        """
        Busca en la base de documentación del curso de IA y devuelve
        los fragmentos más relevantes como contexto en texto plano.
        """
        docs = retriever.vectorstore.similarity_search(
            question,
            k=retriever.search_kwargs.get("k", 3),
        )
        context = "\n\n".join(d.page_content for d in docs)
        if not context:
            return "No encontré documentación relevante para esta pregunta en el RAG."
        return context
```
### 3. Tool de estado: get_user_plan
Para simular lógica de negocio agregué una tool que devuelve el plan de suscripción de un estudiante:

```python
    FAKE_USERS = {
        "sofia": "Plan Pro – acceso completo al curso de IA, materiales y proyecto final.",
        "juan": "Plan Free – acceso limitado a algunos videos y notebooks.",
        "maria": "Plan Estudiante – acceso al curso completo con descuento."
    }

    @tool
    def get_user_plan(username: str) -> str:
        """
        Devuelve el plan/suscripción actual de un estudiante ficticio.
        """
        key = username.strip().lower()
        plan = FAKE_USERS.get(key)
        if not plan:
            return f"No encontré información de suscripción para el usuario '{username}'."
        return f"Estado de suscripción de {username}: {plan}"
```
Esta tool representa el típico acceso a un backend de cuentas, pero con datos fake.
---
### 4. Nodos de LangGraph: assistant, tools y memory
**LLM con tools**

```python
    tools = [rag_search, get_user_plan]
    llm_with_tools = ChatOpenAI(model="gpt-5-mini", temperature=0).bind_tools(tools)
```

**Nodo `assistant`**

El nodo de reasoning decide si responde directo o llama tools. Si hay un `summary`, lo inyecto como `SystemMessage` para dar contexto comprimido:
```python
    def assistant_node(state: AgentState) -> AgentState:
    msgs = list(state["messages"])
    if state.get("summary"):
        msgs = [SystemMessage(
            content=f"Resumen de la conversación hasta ahora:\n{state['summary']}"
        )] + msgs

    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}
```
**Nodo `tools`**

Uso el `ToolNode` preconstruido:
```python
    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(tools)
```
**Router `assistant → tools/END`**

```python
    def route_from_assistant(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END
```

**Nodo `memory`**

```python
    def memory_node(state: AgentState) -> AgentState:
        summary_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        prev_summary = state.get("summary", None)
        messages = state["messages"]
        last_msgs = messages[-6:]

        # Construyo prompt con resumen previo (si existe) + últimos mensajes
        ...
        response = summary_llm.invoke(
            [
                SystemMessage(content="Sos un asistente que resume conversaciones sin incluir datos personales."),
                HumanMessage(content=full_prompt),
            ]
        )
        return {"summary": response.content}
```
El nodo resume en 3 bullets lo acordado hasta ahora, evitando datos sensibles.

**Grafo completo**
```python
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", tool_node)
    builder.add_node("memory", memory_node)

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        route_from_assistant,
        {"tools": "tools", END: END}
    )

    # Circuito cuando hay tools:
    builder.add_edge("tools", "memory")
    builder.add_edge("memory", "assistant")

    graph = builder.compile()
```
El flujo típico es:
`START → assistant → (tools?) → memory → assistant → END`

---
### 5. Conversaciones de prueba

Usé un helper para ver qué tools se activaban:
```python
    def get_tools_used_from_result(result_state: dict):
    last_msg = result_state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return [call.name for call in last_msg.tool_calls]
    return []
```

**Conversación 1 — Pregunta solo de documentación**

Input:
> “¿Cómo es la evaluación del curso de IA?”
- El LLM decide llamar rag_search.
- El RAG devuelve fragmentos sobre assignments, proyecto final y combinación de nota final.
- El asistente responde con una explicación sintetizada usando ese contexto.
- El nodo memory actualiza el summary con 3 bullets sobre cómo se evalúa el curso.

**Conversación 2 — Solo tool de estado**

Input:
> “¿Qué plan tengo yo si soy la usuaria sofia?”
- El LLM detecta intención de estado de cuenta y llama get_user_plan("sofia").
- La tool devuelve el texto asociado al usuario sofia (Plan Pro).
- El asistente responde en lenguaje natural indicando su plan.
- El summary se actualiza reflejando que la usuaria sofia tiene cierto plan.

**Conversación 3 — Multi-turn mezclando estado + RAG**

Turno 1:
> “Hola, soy sofia. ¿Qué plan tengo en el curso de IA?”

Turno 2 (reutilizando estado):
> “Y también explicame brevemente cómo funciona el proyecto final.”

Comportamiento esperado:

- T1: el agente usa get_user_plan para responder sobre el plan de sofia y memory_node guarda que la conversación es con esa usuaria.
- T2: ahora la intención cambia a documentación; el LLM llama rag_search para buscar información del proyecto final y responde usando el contexto del RAG, manteniendo coherencia con el turno anterior.

Esta conversación muestra que el grafo soporta interacciones combinadas (estado + conocimiento) dentro de un mismo hilo.


### 6. Interfaz en Gradio
Implementé una pequeña UI con gr.Blocks:

- gr.Chatbot para visualizar el diálogo.
- gr.State que almacena el AgentState.
- run_agent() que:
    - agrega el HumanMessage,
    - llama graph.invoke(state),
    - devuelve el historial y un log de tools usadas.

Esto permite probar el agente como un chat real sin modificar código en cada iteración.

## Reflexión
### ¿Qué aporta LangGraph frente a un simple `llm.invoke`?
- Me obliga a explicitar un estado estructurado y a pensar en términos de flujo (nodos, edges).
- Integrar tools deja de ser “llamar funciones sueltas” y pasa a ser un workflow orquestado.
- El grafo hace más fácil:
    - insertar nodos extra (memoria, logging, filtrado),
    - controlar dónde ocurre el razonamiento (`assistant`) y dónde sólo se ejecutan acciones (`tools`).

### RAG + tools: trade-offs
- El RAG mejora el grounding en preguntas sobre el curso, pero requiere:
    - cuidar el tamaño del contexto,
    - ajustar k para no devolver texto irrelevante.
- La tool de estado ilustra el patrón típico de “consultar backend”, pero muestra riesgos de una implementación naive: datos hardcodeados, sin auth ni errores de red.

### Memoria ligera: ventajas y cuidados
- El summary permite mantener contexto sin cargar todo el historial.
- Es importante excluir información sensible y evitar que el resumen se convierta en un “log eterno” con datos personales.
- Una mejora futura sería disparar `memory_node` solo cada N turnos o cuando el historial supere cierto tamaño.

---
## Checklist
- [x] Definí un AgentState con messages y summary.
- [x] Construí un RAG con documentación del curso de IA y lo expuse como tool rag_search.
- [x] Implementé una tool de estado get_user_plan para usuarios ficticios.
- [x] Creé un grafo LangGraph con nodos assistant, tools y memory y routing condicional.
- [x] Probé al menos 3 conversaciones cubriendo solo RAG, solo estado y combinación multi-turn.
- [x] Analicé el comportamiento del agente, el uso de tools y el rol de la memoria ligera.
- [x] Implementé una UI en Gradio para probar el agente de forma interactiva.


