# load from disk
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
embed_model = HuggingFaceEmbedding(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)

db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("RD-RAG_1.0")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# Query Data from the persisted index
query_engine = index.as_retriever(similarity_top_k=2)
response = query_engine("What is x linked dominant?")
display(Markdown(f"{response}"))