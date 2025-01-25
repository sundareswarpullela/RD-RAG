from nv-embed import NVEmbedder

class Embedder:
    def __init__(self, model_path, device):
        self.device = device
        self.embedder = get_embedder(model_path)


def get_embedder(model):
    if model == 'nvidia/NV-Embed-v2':
        return NVEmbedder
    else:
        raise ValueError(f'Unknown model: {model}')