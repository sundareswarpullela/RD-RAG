from embedding.cohere_embed import CohereEmbedder
from embedding.nv_embed import NVEmbedder
from embedding.titan_embed import TitanEmbedder
from embedding.openai_embed import OpenAIEmbedder
from embedding.gte_large_embed import GTEEmbedder
from embedding.bge_en_embed import BGEEmbedder
from langchain_text_splitters import CharacterTextSplitter

import json
import argparse
import logging
from PyPDF2 import PdfReader
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient

CHUNK_SIZE = 250

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
        self.embedder = embedder_map[model]()
        self.is_chroma = is_chroma
        
    def __call__(self, input: Documents) -> Embeddings:

        return  self.embedder.embed_passage(input)


    def embed_query(self, text):
        return self.embedder.embed_query(text)
    
    def embed_passage(self, text):
        chunks = self.chunk_text(text)
        embeddings = []
        for chunk in chunks:
            embeddings.append(self.embedder.embed_passage(chunk))
        return embeddings
    


    def chunk_text(self, text):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=CHUNK_SIZE, chunk_overlap=0)
        chunks = []
        for chunk in text_splitter.split(text):
            chunks.append(chunk)
        
        return chunks

    def embed_bioasq(self, data_path):
        with open(data_path, "r") as f:
            data = json.load(f)

        chroma_client = PersistentClient(path ="vectordb")

        collection = chroma_client.create_collection(
            name=f"bioasq_{self.embedder}",
            embeddiong_function=self,
            metadata={
                "hnsw:space": "cosine",
                "description": f"BioASQ data embedded using {self.embedder}",
                }
            )
        
        print("Created ChromaDB collection:", collection.name)
        
        id = 0
        for doc in data:
            for article in doc["articles"]:
                if len(article) >= CHUNK_SIZE:
                    chunked_article = self.chunk_text(article)

                    collection.add(
                        documents=chunked_article,
                        ids=[f"{id + i}" for i in range(len(chunked_article))],
                        metadata={"sources": doc["sources"]}
                    )
                    
                    id += len(chunked_article)

                else:
                    collection.add(
                        documents=[article],
                            ids=[f"{id}"],
                            metadata={"sources": doc["sources"]}    
                        )
                    id += 1
                print(f"Embedded {id} documents using {self.embedder} and saved to ChromaDB collection {collection.name}")
        
        print(f"Embedded {id} documents using {self.embedder} and saved to ChromaDB collection {collection.name}")
