from sentence_transformers import SentenceTransformer



class GTEEmbedder:
    def __init__(self, model_path ="Alibaba-NLP/gte-large-en-v1.5"):
        self.model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        self.max_length = 8192

    def __embed_text__(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings
    
    def embed_query(self, query):
        instruction = ""
        return self.__embed_text__(instruction + query)
    
    def embed(self, texts):
        return self.__embed_text__(texts)