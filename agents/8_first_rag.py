"""
8_first_rag.py

First RAG example for the course.

Pipeline:
documents -> embeddings -> vector store -> retrieval
-> context -> prompt -> LLM answer
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings


documents = [
    Document(
        page_content=(
            "customer_income is the estimated annual income of a customer. "
            "It is classified as Sensitive financial information."
        ),
        metadata={"source": "business_glossary"},
    ),
    Document(
        page_content=(
            "Sensitive financial attributes must not be used for marketing "
            "campaigns unless explicit governance approval has been granted."
        ),
        metadata={"source": "marketing_policy"},
    ),
    Document(
        page_content=(
            "customer_income is owned by the Retail Risk Data Office. "
            "Access to Sensitive data requires approval from the responsible data owner."
        ),
        metadata={"source": "ownership_policy"},
    ),
    Document(
        page_content=(
            "Personal and Sensitive information must only be accessed for approved business purposes."
        ),
        metadata={"source": "access_policy"},
    ),
]

# 1. Embeddings
embeddings = OllamaEmbeddings(model="bge-large")

# 2. Local in-memory vector store
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=documents)

# 3. Retriever
retriever = vector_store.as_retriever(search_kwargs={"k":1})

# 4A. Generic RAG prompt from LangChain Hub
#prompt = hub.pull("rlm/rag-prompt")

# 4B. Data Steward-specific prompt
#
prompt = ChatPromptTemplate.from_template("""
You are a Data Steward assistant.
Use only the following retrieved governance context to answer the question.
Do not invent policies, ownership information, definitions, or governance rules.
If the retrieved context does not contain enough information, say that the
information is not available.
Question: {question}
Context:
{context}
Answer:
""")

# 5. LLM
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# 6. Question
question = (
    "Can customer_income be used for a marketing campaign? "
    "Who owns this field?"
)

# 7. RETRIEVE
retrieved_docs = retriever.invoke(question)

print("\n--- RETRIEVED DOCUMENTS ---")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"\nDocument {i}")
    print("Source:", doc.metadata.get("source", "unknown"))
    print(doc.page_content)

# 8. AUGMENT
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

messages = prompt.invoke(
    {
        "question": question,
        "context": context,
    }
)

# 9. GENERATE
response = llm.invoke(messages)

print("\n--- FINAL ANSWER ---")
print(response.content)


# Suggested student experiments:
# 1. Change k from 3 to 1.
# 2. Ask a question for which the corpus has no answer.
# 3. Switch from the Hub prompt to the Data Steward prompt.
# 4. Add a new governance document and ask a question that retrieves it.