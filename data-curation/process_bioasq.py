import json
import os
import pubmed_parser as pp
from concurrent.futures import ThreadPoolExecutor
rare_diseases = [ 
    "Cystic Fibrosis", 
    "Huntington Disease", 
    "Marfan Syndrome", 
    "Sickle Cell Anemia",
    "Hemophilia", 
    "Gaucher Disease", 
    "Phenylketonuria", 
    "Tay-Sachs Disease", 
    "Alpha-1 Antitrypsin Deficiency",
    "Duchenne Muscular Dystrophy",
    "Amyotrophic Lateral Sclerosis",
    "Fabry Disease", 
    "Pompe Disease", 
    "Wilson Disease", 
    "Spinal Muscular Atrophy",
    "Thalassemia", 
    "Neurofibromatosis Type 1", 
    "Hereditary Angioedema", 
    "X-linked Adrenoleukodystrophy",
    "Ehlers-Danlos Syndrome", 
    "Alport Syndrome", 
    "Friedreich Ataxia", 
    "Rett Syndrome",
    "Prader-Willi Syndrome", 
    "Usher Syndrome", 
    "Von Hippel-Lindau Disease", 
    "Tuberous Sclerosis",
    "Batten Disease",
    "Krabbe Disease", 
    "Leigh Syndrome"
 ]

source_files_directory = "data-curation/data/source_files"
data_set_json = f"{source_files_directory}/training13b.json"


source_data = dict()
source_data["data"] = []
files = [f"{source_files_directory}/{filename}" for filename in os.listdir(source_files_directory) if filename.endswith(".json")]
total = 0
questions = set()
data_set_json = f"{source_files_directory}/training13b.json"

with open(data_set_json, "r") as f:
    print(f"Processing file: {data_set_json.split('/')[-1]}")
    data = json.load(f)
    for q in data["questions"]:
            for disease in rare_diseases:
                if disease.lower() in q["body"].lower():
                    total += 1
                    questions.add(q["body"])
                    articles = []
                    pmids = []
                    for doc in q["documents"]:
                        pmids.append(doc.split("/")[-1])
                    source_data["data"].append({
                                                # "disease": disease,
                                                "question":q["body"], 
                                                "answer": q["ideal_answer"], 
                                                "sources": q["documents"],
                                                "PMIDs": pmids
                                            })
                    
print(total)
print(len(questions))
            

# print(source_data)
pmids = []
for data in source_data["data"]:
    pmids.extend(data["PMIDs"])

pmids = list(set(pmids))
with ThreadPoolExecutor(max_workers=7) as executor:
    pmids_and_articles = executor.map(pp.fetch_pubmed_article, pmids)
pmid_to_article = dict(pmids_and_articles)
print(pmid_to_article)

for data in source_data["data"]:
    articles = []
    for pmid in data["PMIDs"]:
        if pmid in pmid_to_article:
            articles.append({"PMID": pmid, "article":pmid_to_article[pmid]})
    data["articles"] = articles





with open(f"{source_files_directory}/filtered_rare_disease_data.json", "w") as f:
     json.dump(source_data, f, indent=4)
