from embedding.cohere_embed import CohereEmbedder
from embedding.nv_embed import NVEmbedder
from embedding.titan_embed import TitanEmbedder
from embedding.openai_embed import OpenAIEmbedder
from embedding.gte_large_embed import GTEEmbedder
from embedding.bge_en_embed import BGEEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json
import logging
from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient
import time

CHARACTERS_SIZE = 2048

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


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

        return  self.embed(input)


    def embed_query(self, text):
        return self.embedder.embed_query(text)
    
    def embed(self, texts):
        return self.embedder.embed(texts)
        

    


    def chunk_text(self, text):
        # print(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHARACTERS_SIZE,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,

        )
        chunks = []
        for chunk in text_splitter.split_text(text):
            chunks.append(chunk)
        return chunks

    def split_bioasq_article(self, data):

        split_articles = []
        for doc in data:
                for article in doc["articles"]:
                    if len(article["article"]) >= CHARACTERS_SIZE:
                        chunked_article = self.chunk_text(article["article"])
                        chunked_article = [{"pmid": article["PMID"], "article": chunk} for chunk in chunked_article]
                        split_articles.extend(chunked_article)
                    else:
                        split_articles.append({"pmid": article["PMID"], "article": article["article"]})

        return split_articles
            

        
def embed_bioasq(embedder, data_path):
    with open(data_path, "r") as f:
        data = json.load(f)


    

    split_articles = embedder.split_bioasq_article(data["data"])
    

    with open(f"split_articles_{embedder.embedder_name}.json", "w") as f:
        json.dump(split_articles, f)

    print(f"Successfully split articles into splits of {CHARACTERS_SIZE} characters. Total splits: {len(split_articles)}")

    batch_size=20

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

    for i in tqdm(range(0, len(split_articles), batch_size), desc="Adding to ChromaDB"):
        batch = split_articles[i:i+batch_size]
        ids = ["id_" + str(id_idx + i) for i in range(0, len(batch))]
        id_idx += batch_size
        


        # Add to ChromaDB
        collection.add(
            documents= [doc["article"] for doc in batch],
            ids=ids,
            metadatas=[{"pmid": doc["pmid"]} for doc in batch]
        )

    print(f"Embedded documents using {embedder.embedder_name} and saved to ChromaDB collection {collection.name}")
