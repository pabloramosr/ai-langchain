"""
1_ingest_reglamento.py

Script para cargar y vectorizar el Reglamento General de Alumnos (rga.pdf) en Pinecone.
Ubicación del PDF: test/data/rga.pdf
Pipeline: PDF -> Chunks -> Embeddings (bge-large) -> Pinecone Index
"""

import sys
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8")
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Cargar variables de entorno desde la raíz de LangCHain
load_dotenv(Path(__file__).parent.parent / ".env")

INDEX_NAME = os.environ.get("INDEX_NAME", "student-governance")
PDF_PATH = Path(__file__).parent / "data" / "rga.pdf"


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """Extrae el texto página por página del PDF del reglamento."""
    print(f"Leyendo archivo PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "rga.pdf",
                        "document_type": "reglamento_estudiantil",
                        "page": page_num,
                    },
                )
            )
    return documents


def generate_stable_id(doc: Document) -> str:
    """Genera un hash determinista para evitar chunks duplicados al reejecutar."""
    identity = f"{doc.metadata.get('source')}|page_{doc.metadata.get('page')}|{doc.page_content[:100]}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {PDF_PATH}")

    # 1. Cargar páginas
    raw_docs = load_pdf_pages(PDF_PATH)
    print(f"Se cargaron {len(raw_docs)} páginas con texto útil.")

    # 2. Dividir en fragmentos (Chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Se generaron {len(chunks)} chunks.")

    # 3. Inicializar embeddings locales con bge-large
    print("Inicializando modelo de embeddings (bge-large)...")
    embeddings = OllamaEmbeddings(model="bge-large")

    # 4. Conectar a Pinecone
    print(f"Conectando al índice de Pinecone: '{INDEX_NAME}'...")
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    # 5. Subir a Pinecone en lotes para monitorear el progreso
    ids = [generate_stable_id(c) for c in chunks]
    batch_size = 50
    total_chunks = len(chunks)

    print(f"Subiendo {total_chunks} chunks a Pinecone...")
    for i in range(0, total_chunks, batch_size):
        end = min(i + batch_size, total_chunks)
        batch_chunks = chunks[i:end]
        batch_ids = ids[i:end]
        
        vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
        print(f"   Subidos {end}/{total_chunks} chunks...")

    print(f"¡Éxito! Reglamento indexado correctamente en Pinecone ('{INDEX_NAME}').")


if __name__ == "__main__":
    main()