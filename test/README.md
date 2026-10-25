# Agente Data Steward & Router Académico (RAG + Web Search)

Proyecto desarrollado para la evaluación de Agentes Inteligentes y Gobernanza de Datos (Tecnológico de Monterrey).

---

## Integrantes del Equipo
* **David Isaí Díaz Avilés**
* **Ian Paulo García González**
* **Luis Enrique Hernández Torres**
* **Pablo Alberto Ramos Roldán**

---

## Objetivo del Data Steward
El objetivo principal del agente es actuar como un **custodio y consejero de gobernanza de datos estudiantiles**, enfrentándose al reto de **discernir y razonar autónomamente** entre:
1. **Consultar la base de datos vectorial (Pinecone)** para extraer normas, políticas y lineamientos oficiales e internos de los alumnos.
2. **Realizar búsquedas en Internet (Tavily)** cuando la consulta corresponde a información pública, dinámica o externa (como oferta académica o carreras).
3. **Descartar herramientas irrelevantes** que no aportan a la consulta, garantizando respuestas precisas sin inventar ni alucinar permisos ni información.

---

## Arquitectura General

```
                          [ Usuario ]
                              │
                              ▼
            [ LLM Router: Ollama (qwen2.5:3b) ]
                   (Razonamiento y Decisión)
                     /        │        \
                    /         │         \
                   ▼          ▼          ▼
     [ Tool 1: RAG Pinecone ] [ Tool 2: Web Search ] [ Tool 3: Dummy Tool ]
      (bge-large Embeddings)      (Tavily API)        (Herramienta señuelo)
               │                        │                       │
      [ rga.pdf Chunks ]      [ Carreras y Campus ]          [ Ignorada ]
```

### Componentes Tecnológicos:
* **Modelo LLM Local:** Ollama `qwen2.5:3b` (temperatura 0 para respuestas deterministas y confiables).
* **Modelo de Embeddings:** `bge-large` (vía Ollama, vectores densos de 1024 dimensiones).
* **Base de Datos Vectorial:** **Pinecone** (índice en la nube: `student-governance`, métrica `cosine`).
* **Búsqueda Web:** **Tavily Search API** para recuperación de información pública en tiempo real.

### Herramientas del Agente (`Tools`):
1. **`search_student_regulations` (RAG):** Busca en la base de datos vectorial en Pinecone los fragmentos semánticos más afines al query para responder con fundamento normativo acotado y certero.
2. **`search_web` (Búsqueda Web):** Busca en internet mediante Tavily si la consulta trata sobre oferta académica, campus o información pública que no corresponde al reglamento.
3. **`empty_weather_tool` (Herramienta Señuelo):** Herramienta vacía diseñada para evidenciar y evaluar la capacidad del agente para filtrar y descartar herramientas innecesarias.

---

## Archivos de Gobernanza Utilizados
* **`test/data/rga.pdf`:** **Reglamento General de Alumnos (RGA)** del Tecnológico de Monterrey (normas de integridad académica, bajas de materias, sanciones, derechos y deberes).
* **Metadatos asociados:** Cada fragmento vectorizado conserva su origen (`source: rga.pdf`) y su **número de página** para garantizar trazabilidad y auditoría.

---

## Cómo Instalar Dependencias

El proyecto utiliza **`uv`** para garantizar un entorno determinista y reproducible:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/pabloramosr/ai-langchain.git
   cd ai-langchain
   git checkout mcp
   ```

2. **Instalar dependencias del proyecto:**
   ```bash
   uv sync
   ```

3. **Descargar modelos en Ollama:**
   Asegúrate de que el servicio de Ollama esté corriendo y descarga los modelos requeridos:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull bge-large
   ```

4. **Configurar variables de entorno (`.env`):**
   Crea o edita el archivo `.env` en la raíz del proyecto:
   ```dotenv
   PINECONE_API_KEY=tu_api_key_de_pinecone
   INDEX_NAME=student-governance
   TAVILY_API_KEY=tu_api_key_de_tavily
   ```

---

## Cómo Ejecutar la Beta

Los scripts de prueba y ejecución se encuentran dentro de la carpeta `test/` debidamente enumerados:

### Paso 1: Ingesta del Reglamento a Pinecone
Procesa el documento `rga.pdf`, genera los chunks (1200 caracteres, overlap 150), calcula los embeddings con `bge-large` y los almacena en Pinecone:
```bash
uv run test/1_ingest_reglamento.py
```

### Paso 2: Ejecución del Agente Multi-Herramienta (Beta)
Ejecuta el agente interactivo con trazabilidad en consola:
```bash
uv run test/2_agent_rag_and_web.py
```
**En la terminal se mostrará:**
* La herramienta elegida por el agente (`search_student_regulations` vs `search_web`).
* Los parámetros/query con los que invocó la herramienta.
* La evidencia recuperada.
* La respuesta final fundamentada.

---

## Ejemplos de Preguntas que Puede Responder

| Tipo de Consulta | Pregunta de Ejemplo | Herramienta Seleccionada | Justificación |
| :--- | :--- | :--- | :--- |
| **Normativa / RAG** | *"¿Qué dice el reglamento de alumnos sobre las faltas a la integridad académica y deshonestidad?"* | `search_student_regulations` (Pinecone) | La respuesta se encuentra en los estatutos y capítulos disciplinarios del `rga.pdf`. |
| **Oferta Académica / Web** | *"¿Cuáles son las carreras de ingeniería en tecnologías computacionales que ofrece el Tec de Monterrey?"* | `search_web` (Tavily) | El reglamento no contiene catálogos de carreras; el agente acude a internet para obtener la oferta vigente. |

---

## Limitaciones Actuales
1. **Tamaño del Modelo Local (3B):** Al utilizar `qwen2.5:3b`, la capacidad de razonamiento en consultas altamente complejas o prompts muy extensos es más acotada en comparación con modelos de frontera (como Gemini 1.5 Pro o GPT-4o).
2. **Capacidad Computacional:** La inferencia y generación de embeddings se realizan localmente sobre CPU/RAM, lo que puede incrementar la latencia si el equipo no cuenta con aceleración por GPU.
3. **Dependencia de Conectividad Cloud:** Requiere conexión activa a Internet para comunicarse con las APIs de Pinecone y Tavily Search.
