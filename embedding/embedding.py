from embedding.cohere_embed import CohereEmbedder
from embedding.nv_embed import NVEmbedder
from embedding.titan_embed import TitanEmbedder
from embedding.openai_embed import OpenAIEmbedder
from embedding.gte_large_embed import GTEEmbedder
from embedding.bge_en_embed import BGEEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import argparse
import logging
from PyPDF2 import PdfReader
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient

CHARACTERS_SIZE = 2048

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

joson_file_path = "embedding/embedder.json"
pdfs_path = "data_curation/data/files"

embedder_map = {
    "cohere": CohereEmbedder,
    "titan": TitanEmbedder,
    "openai": OpenAIEmbedder,
    "nv": NVEmbedder,
    "gte": GTEEmbedder,
    "bge": BGEEmbedder
}


class Embedder(EmbeddingFunction):
    embedder_map = embedder_map
    def __init__(self, model, is_chroma = False, device="cpu"):
        self.device = device
        self.embedder_name = model
        self.embedder = embedder_map[model]()
        self.is_chroma = is_chroma
        
    def __call__(self, input: Documents) -> Embeddings:

        return  self.embedder.embed_passages(input)


    def embed_query(self, text):
        return self.embedder.embed_query(text)
    
    def embed_passage(self, text):
        chunks = self.chunk_text(text)
        embeddings = []
        for chunk in chunks:
            embeddings.append(self.embedder.embed_passage(chunk))
        return embeddings
    


    def chunk_text(self, text):
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=CHARACTERS_SIZE,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = []
        for chunk in text_splitter.create_documents(text):
            chunks.append(chunk)
        
        return chunks

    def split_bioasq_article(self, data):
        split_articles = []
        for doc in data:
                for article in doc["articles"]:
                    if article:
                        if len(article) >= CHARACTERS_SIZE:
                            chunked_article = self.chunk_text(article)
                            split_articles.extend(chunked_article)
                        else:
                            split_articles.append(split_articles)
        return split_articles
            

        
def embed_bioasq(embedder, data_path):
    with open(data_path, "r") as f:
        data = json.load(f)

    chroma_client = PersistentClient(path ="vectordb")

    collection = chroma_client.get_or_create_collection(
        name=f"bioasq_{embedder.embedder_name}",
        embedding_function=embedder,
        metadata={
            "hnsw:space": "cosine",
            "description": f"BioASQ data embedded using {embedder.embedder_name}",
            }
        )
   
    print("Created ChromaDB collection:", collection.name)
    
    id_idx = 0
    split_articles = embedder.split_bioasq_article(data["data"])
    print("Articles split")
    batch_size=10000
    for i in tqdm(range(0, len(split_articles), batch_size), desc="Adding to ChromaDB"):
        batch = split_articles[i:i+batch_size]
        ids = ["id_" + str(id_idx + i) for i in range(0, batch_size)]
    

        # Add to ChromaDB
        collection.add(ids=ids, documents=batch)
        collection.add(
            documents= batch,
            ids=ids,
            
        )
        id_idx += batch_size
                  
    print(f"Embedded documents using {embedder.embedder_name} and saved to ChromaDB collection {collection.name}")
