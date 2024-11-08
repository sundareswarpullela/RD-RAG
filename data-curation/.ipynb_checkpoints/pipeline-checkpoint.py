from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import chromadb


# config = {
#         "chunk_size":100, 
#         "chunk_overlap":0,
#         "keywords":10,
#         "num_workers":16
#          }

# files_path = "data/files"
# documents = SimpleDirectoryReader(files_path).load_data(num_workers=16)
# print("Loaded documents into memory")

# nodes = SentenceSplitter(chunk_size=config['chunk_size'], chunk_overlap=config['chunk_overlap']).get_nodes_from_documents(documents)
# print("Nodes (Chunks) generated")

# embed_model = HuggingFaceEmbedding(
#     "nomic-ai/nomic-embed-text-v1.5",
#     trust_remote_code=True
# )

# print("Embedder loaded")

# KEYWORD_EXTRACT_PROMPT = '{context_str}. Generate {keywords} keywords from the text. The general theme is X-linked dominant rare diseases. Format as comma separated. Keywords:'

# from llama_index.core.extractors import TitleExtractor, KeywordExtractor
# from llama_index.core.ingestion import IngestionPipeline

# # create the pipeline with transformations
# pipeline = IngestionPipeline(
#     transformations=[
#         TitleExtractor(),
#         KeywordExtractor(keywords=10, prompt_template = KEYWORD_EXTRACT_PROMPT), 
#         embed_model,
#     ]
# )

# # run the pipeline
# nodes = pipeline.run(nodes=nodes, num_workers=12)
# print("Nodes generated")


# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("RD-RAG_1.1")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext(vector_store=vector_store)

# index = VectorStoreIndex.from_documents(
#     documents, 
#     storage_context=storage_context, 
#     embed_model=embed_model,
#     show_progress=True
# )



db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("RD-RAG_1.1_1000_nodes")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# Query Data from the persisted index
retriever = index.as_retriever(similarity_top_k=2)
response = retriever.retrieve("What is x linked dominant?")
display(Markdown(f"{response}"))
