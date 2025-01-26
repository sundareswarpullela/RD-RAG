from nv_embed import NVEmbedder
from cohere_embed import CohereEmbed
from titan_embed import TitanEmbed
from openai_embed import OpenAIEmbedder
from gte_embed import GTEEmbedder
from bge-en-embed import BGEEmbedder


embedder_map = {
    "cohere": CohereEmbed,
    "titan": TitanEmbed,
    "openai": OpenAIEmbedder,
    "nv": NVEmbedder,
    "gte": GTEEmbedder,
    "bge": BGEEmbedder  
}
class Embedder:
    def __init__(self, model, device):
        self.device = device
        self.embedder = embedder_map[model]

    def embed_query(self, text):
        return self.embedder.embed_query(text, self.device)
    
    def embed_passage(self, text):
        return self.embedder.embed_passage(text, self.device)
    
