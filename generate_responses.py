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
        body=json.dumps({"prompt": prompt, "max_gen_len": 512})
    )

    response_body = json.loads(response["body"].read())
    return response_body.get("generation", "").strip()

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

def load_data(data_path):
    """
    Load data from the given JSON file.
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    queries = [item["question"] for item in data]
    ground_truths = [item["answer"] for item in data]

    return queries, ground_truths


def generate_responses(emb_model, data_path):
    """
    Generate responses for the given queries and ground truth answers.
    """

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path="vectordb")
    collection = chroma_client.get_collection(f"bioasq_{emb_model}")
    embedder = Embedder(emb_model)
    queries, ground_truths = load_data(data_path)

    rag_results = rag_pipeline(queries, ground_truths, embedder, collection)

    # Save to JSON
    with open(f"rag_results_{emb_model.name}.json", "w") as f:
        json.dump(rag_results, f, indent=4)

    print(f"RAG process completed. Results saved in 'rag_results_{emb_model.name}.json'.")


