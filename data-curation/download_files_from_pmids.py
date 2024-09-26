import pandas as pd
import os
import requests
import xml.etree.ElementTree as ET


#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8658462/


def download_file_from_pmc(pmc_id):
    url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/' + pmc_id + '/'
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi" + "?id=" + pmc_id
    locator_response = requests.get(oa_url)
    root = ET.fromstring(locator_response.text) 
    pdf_link = root.find(".//link[@format='pdf']")
    print(pdf_link)
    if pdf_link is not None:
        pdf_url = pdf_link.get('href')
        print(f"PDF URL: {pdf_url}")
    else:
        print("PDF link not found")

    

source_file_directory = "data-curation/data/source_files"

csv_files = [f for f in os.listdir(source_file_directory) if f.endswith('.csv')]
print(csv_files)

all_pmcids = set()
for csv_file in csv_files:
    df = pd.read_csv(f"{source_file_directory}/{csv_file}")
    pmcids = df["PMCID"].tolist()
    all_pmcids.update(pmcids)

print(len(all_pmcids))

all_pmcids = list(all_pmcids)
print(all_pmcids[:5])

for pmcid in all_pmcids:
    print(pmcid)
    download_file_from_pmc(pmcid)
    break