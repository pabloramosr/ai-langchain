from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
import re

# 1. Conexión nativa de LangChain con PostgreSQL
# Incluimos los esquemas 'dim' y 'cfg' en el search_path para que detecte todas las tablas
db_uri = "postgresql+psycopg2://postgres:banamex_database@localhost:5432/reto_banamex?options=-csearch_path=dim,cfg,public"
db = SQLDatabase.from_uri(db_uri)

print("=" * 60)
print("Tablas detectadas en la base de datos:")
print(db.get_usable_table_names())
print("=" * 60)

# 2. Inicializar el modelo con OllamaLLM (Modo Completado / Completion)
# Defog SQLCoder fue entrenado para completado de código, no para chat conversacional
llm = OllamaLLM(
    model="sqlcoder:7b",
    temperature=0
)

# 3. Pregunta de prueba de negocio
pregunta = "¿Cuántos clientes tenemos en el segmento PREMIUM?"

# 4. Obtener el esquema de las tablas relevantes para darle contexto al LLM
esquema = db.get_table_info(["dim_cliente", "dim_producto", "dim_txn"])

# 5. Formato de Prompt oficial de Defog SQLCoder
prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{pregunta}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{esquema}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{pregunta}[/QUESTION]
[SQL]
"""

print(f"\nPregunta del usuario: {pregunta}")
print("Generando consulta SQL con sqlcoder:7b...")

# 6. Invocar al modelo
respuesta = llm.invoke(prompt)
texto_crudo = (respuesta if isinstance(respuesta, str) else respuesta.content).strip()

def extract_sql(text: str) -> str:
    # 1. Remover etiquetas [SQL] si vienen al inicio o al final
    text = text.replace("[SQL]", "").strip()
    
    # 2. Si el modelo usó bloques de código markdown ```sql ... ```
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        
    # 3. Buscar la sentencia SELECT o WITH
    match = re.search(r"\b(SELECT|WITH)\b.*?(?:;|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    # 4. Limpieza de encabezados de texto si no hubo coincidencia
    lineas = [l for l in text.splitlines() if not l.strip().startswith("###") and not l.strip().startswith("Given the")]
    return "\n".join(lineas).strip()

sql_generado = extract_sql(texto_crudo)

print("\nConsulta SQL generada por el LLM:")
print(sql_generado)

# 7. Ejecutar la consulta en la base de datos PostgreSQL
print("\nResultado directo de la base de datos:")
try:
    if sql_generado:
        resultado = db.run(sql_generado)
        print(resultado)
    else:
        print("No se pudo extraer una consulta SQL válida de la respuesta.")
        print(f"Respuesta cruda del modelo:\n{texto_crudo}")
except Exception as e:
    print(f"Error al ejecutar la consulta: {e}")
    print(f"Respuesta cruda recibida:\n{texto_crudo}")
print("=" * 60)

# Liberar y cerrar todas las conexiones activas hacia PostgreSQL
db._engine.dispose()
print("Conexiones a la base de datos cerradas correctamente.")