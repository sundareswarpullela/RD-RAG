import pandas as pd
import os
import requests
import xml.etree.ElementTree as ET
import urllib.request
from concurrent.futures import ThreadPoolExecutor



#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8658462/


def download_file_from_pmc(pmc_id):
    url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/' + pmc_id + '/'
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi" + "?id=" + pmc_id
    locator_response = requests.get(oa_url)
    root = ET.fromstring(locator_response.text) 
    pdf_link = root.find(".//link[@format='pdf']")
    if pdf_link is not None:
        pdf_url = pdf_link.get('href')
        print(f"PDF URL: {pdf_url}")
        file_location = f"data-curation/data/files/{pmc_id}.pdf"


        urllib.request.urlretrieve(pdf_url, file_location)
        print(f"Downloaded {pdf_file_name}")
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
print(len(set(all_pmcids)))

all_pmcids = list(all_pmcids)
print(all_pmcids[:5])

def download_multiple_pmc_files(pmc_ids):
    with ThreadPoolExecutor() as executor:
        executor.map(download_file_from_pmc, pmc_ids)

# List of PMC IDs you want to download

# Run the download in parallel
# download_multiple_pmc_files(all_pmcids[:100])