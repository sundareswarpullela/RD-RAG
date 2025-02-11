import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.nn import DataParallel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



class NVEmbedder:
    def __init__(self, model_path="nvidia/NV-Embed-v2"):
        self.max_length = 32768
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = self.model.to(self.device)
        # self.model = DataParallel(self.model).to(self.device)


        
    def embed_query(self, query):
        task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

        query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
        query_embeddings = self.model.encode([query], instruction=query_prefix, max_length=self.max_length)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings

    
    def embed(self, texts):    
        embeddings = self.model.encode(texts, instruction="", max_length=self.max_length)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings