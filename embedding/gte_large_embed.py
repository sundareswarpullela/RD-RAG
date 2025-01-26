# Requires transformers>=4.36.0
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class GTEEmbedder:
    def __init__(self, model_path ="Alibaba-NLP/gte-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)


    def __embed_text__(self, text):
        encoded_input = self.tokenizer([text], max_length=8192, padding=True, truncation=True, return_tensors='pt')
        model_output = self.model(**encoded_input)
        embeddings = model_output[:, 0]
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def embed_query(self, query):
        instruction = ""
        return self.__embed_text__(instruction + query)
    
    def embed_passage(self, passage):
        return self.__embed_text__(passage)