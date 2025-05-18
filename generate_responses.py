import chromadb
import boto3
import json
import uuid
from embedding.embedding import Embedder
import os
from tqdm import tqdm

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
    prompt = f"""You are an expert in answering Questions about Rare Diseases based on the given context. 
    Only use the given context to answer the questions.
    Here is an example: 
    Question: Which is the genetic defect causing Neurofibromatosis type 1?
    Context: 
    Neurofibromatosis type 1 (NF1), characterized by skin neurofibromas and an excess of caf\u00e9-au-lait spots, is due to mutations in the neurofibromin (NF1) gene. Identifying the genetic defect in individuals with the disease represents a significant challenge because the gene is extremely large with a high incidence of sporadic mutations across the entire gene ranging from single nucleotide substitutes to large deletions. In the present study, we have used a combination of techniques (heteroduplex analysis, sequencing, loss of heterozygosity and quantification of gene dosage) to define the genetic defect in 68 individuals from a cohort of 107 NF1 Taiwanese patients of Chinese origin. Fifty-eight were initially identified using heteroduplex analytical techniques and confirmed by sequence analysis. A further five were identified by direct sequence analysis alone. The reminders were shown to carry large deletions in the NF1 gene by demonstrating loss of heterozygosity that was confirmed by gene dosage measurements using quantitative-PCR techniques. Mis-sense, non-sense, frame-shift or splice-site mutations were identified across the entire gene of which the majority (45/68) were novel in nature. The detection rate with the various analytical techniques and the types of mutation detected are consistent with published data involving both individuals and large cohort studies from other ethnic backgrounds.
    The locus for the gene causing neurofibromatosis type 1 (NF1) was bracketed to a region on the long arm of chromosome 17 by means of genetic linkage analysis. When the limits of resolution for genetic mapping were reached physical mapping methods were used to map the NF1 gene precisely, with reference to translocation breakpoints in NF1 affected individuals who harboured constitutional chromosomal translocations on chromosome 17. The region of DNA located between two translocation breakpoints has been cloned and a DNA sequence encoding a 11-13 kb mRNA identified. That this sequence shows deletions and point mutations in NF1 affected individuals and not in normal controls provides strong evidence that it is indeed the NF1 gene. The genetic defect in NF2 has been mapped to chromosome 22 by studies of chromosomal loss in tumours associated with this disease. Subsequent linkage analysis of NF2 pedigrees has confirmed this location. DNA markers that bracket the NF2 locus to a region of 5-10 Mb have been identified.
    Answer: Neurofibromatosis type 1 (NF1) is due to all types of mutations in the neurofibromin (NF1) gene.

    
    Now answer the following question using the given context:
    
    Question: {query}\n\n
    Context:\n{context_text}\n\n
    Answer:"""
    try: 
        response = bedrock_client.invoke_model(
            modelId="meta.llama3-1-70b-instruct-v1:0",
            body=json.dumps({"prompt": prompt, "max_gen_len": 512})
        )
    
        response_body = json.loads(response["body"].read())
        return response_body.get("generation", "").strip()
        
    except Exception as e:
        print(f"Unable to generate response. {e}")
        
        return ""
    

def rag_pipeline(queries, ground_truths, embedder, collection):
    """
    Perform retrieval-augmented generation (RAG) for multiple queries.
    """
    results = []

    for query, gt_answer in tqdm(zip(queries, ground_truths), total=len(queries), desc="Processing Queries"):

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
    queries = [item['question'] for item in data['data']]
    ground_truths = [item["answer"][0] for item in data['data']]

    return queries, ground_truths


def generate_responses(embedder, data_path):
    """
    Generate responses for the given queries and ground truth answers.
    """

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path="vectordb")
    collection = chroma_client.get_collection(f"bioasq_{embedder.embedder_name}")
    queries, ground_truths = load_data(data_path)


    rag_results = rag_pipeline(queries, ground_truths, embedder, collection)

    # Save to JSON
    with open(f"rag_results_{embedder.embedder_name}.json", "w") as f:
        json.dump(rag_results, f, indent=4)

    print(f"RAG process completed. Results saved in 'rag_results_{embedder.embedder_name}.json'.")


