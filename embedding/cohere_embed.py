import boto3
import json 



# "{\"texts\":[\"Hello world\",\"This is a test\"],\"input_type\":\"search_document\"}"
class CohereEmbedder:
    def __init__(self):
        self.client = boto3.client('bedrock-runtime')
        self.modelId = 'cohere.embed-english-v3'
        self.contentType = 'application/json'
        self.accept = 'application/json'
        self.max_length = 512
    def embed(self, text):
        body = json.dumps(
            {"texts":[text],
             "input_type":"search_document"
            }
        )
        
        response = self.client.invoke_model(body=body, modelId=self.modelId, accept=self.accept, contentType=self.contentType)
        
        return json.loads(response['body'].read())
    
    def embed_query(self, query):
        instruction = ""
        return self.embed(instruction + query)
    
    def embed_passages(self, texts):
        embeddings = []
        for text in texts:
             embeddings.append(self.embed(text)["embeddings"])
        return embeddings



# embb = CohereEmbedder()
# print(embb.embed(["abc" for x in range(10)]))
    