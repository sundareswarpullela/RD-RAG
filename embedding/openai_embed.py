from openai import OpenAI
class OpenAIEmbedder:
    def __init__(self, model_path = "text-embedding-3-large"):
        self.client = OpenAI()
        self.model = model_path
        self.max_length = 8191

    def __embed_text__(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )

        return response.data[0].embedding
    
    def embed_query(self, query):
        return self.__embed_text__(query)   
    
    def embed(self, texts):
        embeddings = []
        for text in texts:
            embeddings.append(self.__embed_text__(text))
        return embeddings