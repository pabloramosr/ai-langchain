"""
10_agentic_rag_pinecone.py

Agentic RAG over a persistent Pinecone student-data-governance knowledge base.

At runtime the agent does NOT reopen the local source files.
It decides when to search Pinecone and uses the retrieved evidence to answer.
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama

load_dotenv()

INDEX_NAME = os.environ["INDEX_NAME"]

embeddings = OllamaEmbeddings(model="bge-large")

vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5},
)


@tool
def search_governance_knowledge(query: str) -> str:
    """Search student-data definitions, privacy documents, AI rules, ownership, and retention policies."""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant governance evidence was found."

    evidence = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        row = doc.metadata.get("row")

        location = source
        if page is not None:
            location += f", page {page}"
        if row is not None:
            location += f", row {row}"

        evidence.append(
            f"SOURCE: {location}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(evidence)


'''llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)'''

llm = ChatOllama(model="qwen2.5:3b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[search_governance_knowledge],
    system_prompt="""
You are a Data Steward for student data.

Use the governance retrieval tool whenever a question depends on:
- privacy requirements
- data classification
- data ownership
- approved uses
- retention
- AI or analytics use of student data

Base governance claims only on retrieved evidence.
Distinguish between official/public documents and synthetic teaching policies
when the source information makes that distinction clear.

If the retrieved evidence is insufficient, conflicting, or does not authorize
the proposed use, say so. Do not invent permission.

When possible, mention the source filename that supports the answer.
""",
)


def print_final_text(result):
    message = result["messages"][-1]

    if isinstance(message.content, str):
        print(message.content)
        return

    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block.get("text", ""))


if __name__ == "__main__":
    questions = [
        "Can we use face embeddings to personalize student marketing?",
        "Who owns student_gpa and how long should it be retained?",
        "Can student location data be used to train an attendance prediction model?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("QUESTION:", question)

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        print("\nANSWER:")
        print_final_text(result)