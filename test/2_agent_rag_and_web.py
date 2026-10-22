"""
2_agent_rag_and_web.py

Agente Router Multimodal: Razona entre buscar en el RAG (Reglamento en Pinecone)
o buscar en la Web (Tavily), descartando herramientas irrelevantes.

Demuestra:
1. Elección fundamentada de herramientas (RAG vs Web vs Herramienta vacía).
2. Trazabilidad: Muestra en pantalla el plan y los llamados que realiza el agente.
3. Respuestas fundamentadas en evidencia.
"""

import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch

# Cargar variables de entorno desde la raíz de LangCHain
load_dotenv(Path(__file__).parent.parent / ".env")

INDEX_NAME = os.environ.get("INDEX_NAME", "student-governance")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# ==========================================
# 1. Configuración del Vector Store (RAG)
# ==========================================
embeddings = OllamaEmbeddings(model="bge-large")
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})


# ==========================================
# 2. Definición de Herramientas (Tools)
# ==========================================

@tool
def search_student_regulations(query: str) -> str:
    """Busca en el Reglamento General de Alumnos (RGA) del Tecnológico de Monterrey en Pinecone.
    USA ESTA HERRAMIENTA OBLIGATORIAMENTE para preguntas sobre:
    - Normas y reglas de estudiantes, faltas disciplinarias o integridad académica.
    - Bajas de materias, requisitos de graduación, faltas de asistencia, calificaciones.
    - Derechos, deberes y sanciones a estudiantes del Tec.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No se encontró evidencia relevante en el reglamento estudiantil."

    results = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "rga.pdf")
        results.append(f"[FUENTE: {source}, Página {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


@tool
def search_web(query: str) -> str:
    """Busca información pública y actual en Internet sobre el Tec de Monterrey mediante Tavily.
    USA ESTA HERRAMIENTA OBLIGATORIAMENTE para preguntas sobre:
    - Oferta académica: qué carreras profesionales, maestrías o preparatorias ofrece el Tec.
    - Campus, fechas de admisión, costos de colegiatura, eventos o noticias públicas.
    - Información general de la institución que NO son normas ni reglamentos disciplinarios.
    """
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY no configurada en .env"

    tavily = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)
    results = tavily.invoke(query)
    return str(results)


@tool
def empty_weather_tool(location: str) -> str:
    """Herramienta señuelo para consultar el clima en campus universitarios.
    NO USAR para temas académicos, carreras ni reglamentos.
    """
    return f"Información meteorológica no disponible para {location}."


# ==========================================
# 3. Creación del Agente y System Prompt
# ==========================================

llm = ChatOllama(model="qwen2.5:3b", temperature=0)

system_prompt = """
Eres un Asistente Académico y Consejero de Estudiantes del Tecnológico de Monterrey.

Tu objetivo principal es responder con precisión eligiendo la herramienta correcta según la intención del usuario:

REGLAS DE DECISIÓN DE HERRAMIENTAS:
1. SI LA PREGUNTA ES SOBRE REGLAS, POLÍTICAS O NORMAS:
   - Temas: Bajas de materias, integridad académica, sanciones, faltas, derechos estudiantiles.
   - DECISIÓN: Debes llamar a 'search_student_regulations'. NUNCA uses búsqueda web para normas internas oficiales.
2. SI LA PREGUNTA ES SOBRE CARRERAS O INFORMACIÓN PÚBLICA GENERAL:
   - Temas: Qué carreras ofrece el Tec, planes de estudio, campus, admisiones, noticias.
   - DECISIÓN: Debes llamar a 'search_web', ya que el reglamento NO contiene el catálogo de carreras.
3. SI LA PREGUNTA CONTIENE AMBOS TEMAS:
   - Llama a ambas herramientas ('search_student_regulations' y 'search_web') y sintetiza la respuesta.
4. DESCARTA herramientas irrelevantes como 'empty_weather_tool' a menos que pidan el clima explícitamente.

REQUISITO DE TRANSPARENCIA:
Antes de dar la respuesta final, indica brevemente qué herramienta consultaste y por qué.
Basa tus respuestas estrictamente en la evidencia obtenida y menciona las fuentes consultadas.
"""

agent = create_agent(
    model=llm,
    tools=[search_student_regulations, search_web, empty_weather_tool],
    system_prompt=system_prompt,
)


# ==========================================
# 4. Función de Trazabilidad y Visualización del Plan
# ==========================================

def run_agent_with_trace(user_question: str):
    """Ejecuta el agente mostrando paso a paso el plan y las herramientas elegidas."""
    print("\n" + "=" * 80)
    print(f"PREGUNTA DEL USUARIO: {user_question}")
    print("=" * 80)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question,
                }
            ]
        }
    )

    # Inspeccionamos los mensajes para mostrar los llamados realizados (El Plan)
    print("\n--- SEGUIMIENTO DE LLAMADOS Y DECISIONES (PLAN DEL AGENTE) ---")
    tool_called = False
    for msg in result["messages"]:
        # Si el modelo decidió llamar a una herramienta:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                tool_called = True
                name = call.get("name")
                args = call.get("args")
                print(f"[DECISIÓN]: El agente eligió la herramienta -> '{name}'")
                print(f"[PARÁMETROS]: {args}")
        elif msg.type == "tool":
            snippet = msg.content[:160].replace("\n", " ") if msg.content else ""
            print(f"[EVIDENCIA RECIBIDA]: {snippet}...")

    if not tool_called:
        print("El agente respondió directamente sin requerir herramientas.")

    # Imprimir la respuesta final
    print("\n--- RESPUESTA FINAL DEL AGENTE ---")
    final_message = result["messages"][-1]
    if isinstance(final_message.content, str):
        print(final_message.content)
    elif isinstance(final_message.content, list):
        for block in final_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))


# ==========================================
# 5. Demostración y Pruebas Contrastantes
# ==========================================

if __name__ == "__main__":
    test_queries = [
        # Caso 1: Debe ir al RAG (Reglamento estudiantil)
        "¿Qué dice el reglamento de alumnos sobre las faltas a la integridad académica y deshonestidad?",
        
        # Caso 2: Debe ir a la Web (Carreras / Oferta no está en el reglamento)
        "¿Cuáles son las carreras de ingeniería en tecnologías computacionales que ofrece el Tec de Monterrey?",
    ]

    for q in test_queries:
        run_agent_with_trace(q)