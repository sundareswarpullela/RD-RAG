import pandas as pd
import os
import requests
import time
import xml.etree.ElementTree as ET
import urllib.request
from concurrent.futures import ThreadPoolExecutor


source_file_directory = "data-curation/data/source_files"
pdf_files_location = "data-curation/data/files/"
download_manifest_location = "data-curation/data/local_files.json"

    
def download_file_from_pmc(pmc_id):
    url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/' + pmc_id + '/'
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi" + "?id=" + pmc_id
    locator_response = requests.get(oa_url)
    root = ET.fromstring(locator_response.text) 
    pdf_link = root.find(".//link[@format='pdf']")
    if pdf_link is not None:
        pdf_url = pdf_link.get('href')
        print(f"PDF URL: {pdf_url}")
        pdf_file_name = f"{pmc_id}.pdf"
        
        try:
            urllib.request.urlretrieve(pdf_url, pdf_files_location + pdf_file_name)
            print(f"Downloaded {pdf_file_name}")
            
        except e:
            print(e)
    else:
        print("PDF link not found")
        pdf_link = root.find(".//link[@format='']")
        

def download_multiple_pmc_files(pmc_ids):
    with ThreadPoolExecutor() as executor:
        executor.map(download_file_from_pmc, pmc_ids)




csv_files = [f for f in os.listdir(source_file_directory) if f.endswith('.csv')]
print(csv_files)

all_pmcids = []
for csv_file in csv_files:
    df = pd.read_csv(f"{source_file_directory}/{csv_file}")
    pmcids = df["PMCID"].tolist()
    all_pmcids.extend(pmcids)

all_unique_pmcids = list(set(all_pmcids))
# List of PMC IDs you want to download
download_multiple_pmc_files(all_unique_pmcids)