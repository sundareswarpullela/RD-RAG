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

    def embed(self, texts):
        # print(texts)
        print(len(texts))


        body = json.dumps(
            {"texts":texts,
             "input_type":"search_document"
            }
        )
        try:
            response = self.client.invoke_model(body=body, 
                                                modelId=self.modelId, 
                                                accept=self.accept, 
                                                contentType=self.contentType
                                                )
            return json.loads(response['body'].read())["embeddings"]
        except Exception as e:
            print("ERROR: ", e)

        
    
    def embed_query(self, query):
        instruction = ""
        return self.embed(instruction + query)
    
    def embed_passages(self, texts):
        print(len(texts))
        # return self.embed(texts)



# embb = CohereEmbedder()
# print(embb.embed_passage(["abc" for _ in range(10)]))
    