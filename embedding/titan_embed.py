import boto3
import json 


class TitanEmbedder:
    def __init__(self):
        self.client = boto3.client('bedrock-runtime')
        self.modelId = 'amazon.titan-embed-text-v2:0'
        self.contentType = 'application/json'
        self.accept = '*/*'
    def embed(self, text):

        body = json.dumps({"inputText": text ,"dimensions": 512, "normalize": True})
        response = self.client.invoke_model(body=body, modelId=self.modelId, accept=self.accept, contentType=self.contentType)
        
        return response['Body'].read().decode('utf-8')
    
    def embed_query(self, query):
        instruction = ""
        return self.embed(instruction + query)
    
    def embed_passage(self, passage):
        return self.embed(passage)
    