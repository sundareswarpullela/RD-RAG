from sentence_transformers import SentenceTransformer
class Embedder:
    def __init__(model_path, device):
        device = device
        model = SentenceTransformer(model_path).to(device)
        model.eval()

    def embed_query(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True).to(self.device)

    def embed_doc(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True).to(self.device)
