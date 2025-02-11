from transformers import AutoTokenizer, AutoModel
import torch


class BGEEmbedder:
    def __init__(self, model_path = "BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.max_length = 512

    
    def __embed_text__(self, texts):
        tokenized_input = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**tokenized_input)
        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings
    
    def embed_query(self, query):
        instruction = ""
        return self.__embed_text__(instruction + query)
    
    def embed(self, texts):
        embeddings = self.__embed_text__(texts)
        return embeddings
    