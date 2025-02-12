import chromadb
import boto3
import json
import uuid
from embedding.embedding import Embedder
import os


# Initialize AWS Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

def retrieve_context(query, embedder, collection, top_k=5):
    """
    Retrieve relevant context from ChromaDB for the given query.
    """
    results = collection.query(query_embeddings=embedder.embed([query]), n_results=top_k)

    retrieved_contexts = []
    for doc_id, text in zip(results["ids"][0], results["documents"][0]):
        retrieved_contexts.append({"doc_id": doc_id, "text": text})

    return retrieved_contexts

def generate_response(query, retrieved_context):
    """
    Query Llama 3 on AWS Bedrock and generate a response.
    """
    context_text = "\n".join([ctx["text"] for ctx in retrieved_context])
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

    response = bedrock_client.invoke_model(
        modelId="meta.llama3-1-70b-instruct-v1:0",
        body=json.dumps({"prompt": prompt, "max_tokens": 512})
    )

    response_body = json.loads(response["body"].read())
    return response_body.get("outputText", "").strip()

def rag_pipeline(queries, ground_truths, embedder, collection):
    """
    Perform retrieval-augmented generation (RAG) for multiple queries.
    """
    results = []

    for query, gt_answer in zip(queries, ground_truths):
        query_id = str(uuid.uuid4())  # Generate unique query ID
        retrieved_context = retrieve_context(query, embedder, collection)
        response = generate_response(query, retrieved_context)

        results.append({
            "query_id": query_id,
            "query": query,
            "gt_answer": gt_answer,
            "response": response,
            "retrieved_context": retrieved_context
        })

    return {"results": results}


def generate_responses(queries, ground_truths, emb_model):
    """
    Generate responses for the given queries and ground truth answers.
    """

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path="vectordb")
    collection = chroma_client.get_collection(f"bioasq_{emb_model}")
    embedder = Embedder(emb_model)

    rag_results = rag_pipeline(queries, ground_truths, embedder, collection)

    # Save to JSON
    with open("rag_results.json", "w") as f:
        json.dump(rag_results, f, indent=4)

    print("RAG process completed. Results saved in 'rag_results.json'.")



generate_responses(
    ["What is x linked dominant?"], 
    ["X-linked dominant inheritance is a mode of genetic inheritance \
     by which a dominant gene is carried on the X chromosome."], 
    "nv")
