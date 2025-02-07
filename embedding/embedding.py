from cohere_embed import CohereEmbedder
from nv_embed import NVEmbedder
from titan_embed import TitanEmbedder
from openai_embed import OpenAIEmbedder
from gte_large_embed import GTEEmbedder
from bge_en_embed import BGEEmbedder
from langchain_text_splitters import CharacterTextSplitter


import argparse
import logging
from PyPDF2 import PdfReader
from chromadb import Documents, EmbeddingFunction, Embeddings



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
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0)
        chunks = []
        for chunk in text_splitter.split(text):
            chunks.append(chunk)
        
        return chunks


            


    

if __name__ == "__main__":
    # Testing code
    embedder = Embedder("cohere")
    pages = embedder.embed_pdf("/Users/sundar/Projects/RD-RAG/data-curation/data/files/10.1055_s-0042-1759881.pdf10.1055_s-0042-1759881.pdf")
    for p in pages:
        print(p)
        print(embedder.embed_passage(p[:2048]))
        print("\n\n\n")
        break


