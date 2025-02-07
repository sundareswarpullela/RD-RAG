import json
import os
import pubmed_parser as pp
from concurrent.futures import ThreadPoolExecutor
from util.RLThreadPool import RateLimitedThreadPoolExecutor
rare_diseases = [ 
    "Catecholaminergic polymorphic Ventricular Tachycardia",
    "Mucolipidosis",
    "Aicardi-Goutieres syndrome",
    "Ataxia telangiectasia",
    "Inherited arrhythmogenic cardiomyopathy",
    "Medium chain acyl-CoA dehydrogenase deficiency",
    "Noonans syndrome",
    "Primary ciliary dyskinesia",
    "Williams syndrome",
    "Charcot Marie Tooth",
    "Silver Russell",
    "Osteogenesis imperfecta",
    "Hereditary hemorrhagic telangiectasia",
    "Romano-Ward syndrome",
    "Fragile X syndrome",
    "Adams Oliver syndrome",
    "Neurofibromatosis",
    "Cystic fibrosis",
    "Tay-sachs disease",
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
            

# print(source_data)
pmids = []
for data in source_data["data"]:
    pmids.extend(data["PMIDs"])

pmids = list(set(pmids))
print(len(pmids))
with ThreadPoolExecutor(max_workers=5) as executor:
    pmids_and_articles = executor.map(pp.fetch_pubmed_article, pmids)
pmid_to_article = dict(pmids_and_articles)
print(pmid_to_article)

for data in source_data["data"]:
    articles = []
    for pmid in data["PMIDs"]:
        if pmid in pmid_to_article:
            articles.append(pmid_to_article[pmid])
    data["articles"] = articles





with open(f"{source_files_directory}/filtered_data.json", "w") as f:
     json.dump(source_data, f, indent=4)
