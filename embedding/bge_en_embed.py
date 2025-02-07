from transformers import AutoTokenizer, AutoModel
import torch


class BGEEmbedder:
    def __init__(self, model_path = "BAAI/bge-large-en-v1.5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
        self.max_length = 512

    
    def __embed_text__(self, text):
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            embeddings = model_output[0][:, 0]
        # normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def embed_query(self, query):
        instruction = ""
        return self.__embed_text__(instruction + query)
    
    def embed_passage(self, passage):
        return self.__embed_text__(passage)
    