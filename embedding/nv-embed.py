import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from embedding import Embedder


class NVEmbedder:
    def __init__(self, model_path, device):
        self.device = device
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)

    def embed_query(self, sentences):
        task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
        instruction = "Instruct: " + task_name_to_instruct["example"] + "\n"
        return self._encode(sentences, instruction)

    def embed_doc(self, sentences):
        return self._encode(sentences)

    def _encode(self, sentences, max_length=32768, instruction=None):
        if instruction:
            sentences = [instruction + sentence for sentence in sentences]
            embeddings = self.model.encode(sentences, max_length=max_length, instruction=instruction)
        else:
            embeddings = self.model.encode(sentences, max_length=max_length)
        return F.normalize(embeddings, p=2, dim=1)


