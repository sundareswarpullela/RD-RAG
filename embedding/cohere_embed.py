import boto3
import json 



# "{\"texts\":[\"Hello world\",\"This is a test\"],\"input_type\":\"search_document\"}"
class CohereEmbed:
    def __init__(self):
        self.client = boto3.client('bedrock-runtime')
        self.modelId = 'meta.llama3-1-70b-instruct-v1:0'
        self.contentType = 'application/json'
        self.accept = 'application/json'
    def embed(self, text):

        body = json.dumps({"texts":[text],"input_type":"search_document"})
        response = self.client.invoke_model(body=body, modelId=self.modelId, accept=self.accept, contentType=self.contentType)
        
        return response['Body'].read().decode('utf-8')
    
    def embed_query(self, query):
        instruction = ""
        return self.embed(instruction + query)
    
    def embed_passage(self, passage):
        return self.embed(passage)
    