from nv_embed import NVEmbedder
from cohere_embed import CohereEmbedder
from titan_embed import TitanEmbedder
from openai_embed import OpenAIEmbedder
from gte_large_embed import GTEEmbedder
from bge_en_embed import BGEEmbedder


embedder_map = {
    "cohere": CohereEmbedder,
    "titan": TitanEmbedder,
    "openai": OpenAIEmbedder,
    "nv": NVEmbedder,
    "gte": GTEEmbedder,
    "bge": BGEEmbedder
}
class Embedder:
    def __init__(self, model, device):
        self.device = device
        self.embedder = embedder_map[model]()

    def embed_query(self, text):
        return self.embedder.embed_query(text)
    
    def embed_passage(self, text):
        return self.embedder.embed_passage(text)
    
if __name__ == "__main__":
    embedder = Embedder("titan", "cuda")
    query = "What is the capital of France?"
    passage = "The capital of France is Paris."
    query_embedding = embedder.embed_query(query)
    passage_embedding = embedder.embed_passage(passage)
    print(query_embedding)
    print(passage_embedding)