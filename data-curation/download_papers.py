import pandas as pd
import os
import requests
import time
import xml.etree.ElementTree as ET
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import logging

log = logging.getLogger()

logging.basicConfig(
    filename="data-curation/data/download_papers.log",
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    )
log.propagate = False
log.setLevel(logging.INFO)


source_file_directory = "data-curation/data/source_files"
pdf_files_location = "data-curation/data/files/"

UNPAYWALL_API = "https://api.unpaywall.org/v2/"

EMAIL = "psundareswar@gmail.com"


def get_pdf_from_unpaywall(doi):
    """Attempts to retrieve an open-access PDF via Unpaywall API."""
    try:
        response = requests.get(f"{UNPAYWALL_API}{doi}?email={EMAIL}")
        data = response.json()
        
        if "best_oa_location" in data and data["best_oa_location"]:
            return data["best_oa_location"]["url_for_pdf"]

        return None
    except Exception as e:
        log.warning(f"Failed to fetch from Unpaywall: {e}")
        return None
    

def get_pdf_from_pmc(pmc_id):
    url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/' + pmc_id + '/'
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi" + "?id=" + pmc_id
    locator_response = requests.get(oa_url)
    root = ET.fromstring(locator_response.text) 
    pdf_link = root.find(".//link[@format='pdf']")
    if pdf_link is not None:
        pdf_url = pdf_link.get('href')
        return pdf_url
    return None



def download_pdf(pdf_url, filename, save_path):
    """Downloads and saves the PDF file."""
    
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as pdf_file:
                for chunk in response.iter_content(1024):
                    pdf_file.write(chunk)
            log.info(f"Downloaded successfully: {filename}")
        else:

            print(f"Failed to download {filename}, HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")

def fetch_research_paper(identifiers, save_folder="data-curation/data/files"):
    """Fetches and downloads the research paper given a DOI."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename = f"{identifiers[0].replace('/', '_')}.pdf"
    save_path = f"{save_folder}/{filename}"
    if os.path.exists(save_path):
        log.info(f"File already exists: {filename}")
        return
    pdf_url = get_pdf_from_unpaywall(identifiers[0])
    if not pd.isna(identifiers[1]) and not pdf_url:
        # Fallback to pubmed_central
        pdf_url = get_pdf_from_pmc(identifiers[1])
    log.debug(f"DOI: {identifiers[0]}, PDF URL: {pdf_url}")

    if pdf_url:
        log.info(f"Downloading PDF from: {pdf_url}")
        download_pdf(pdf_url, filename, save_path)
    else:
        log.warning(f"Could not find an open-access version for DOI: {identifiers[0]}")



def download_multiple_files(identifiers):
    with ThreadPoolExecutor() as executor:
        executor.map(fetch_research_paper, identifiers)


csv_files = [f for f in os.listdir(source_file_directory) if f.endswith('.csv')]


for csv_file in csv_files:
    df = pd.read_csv(f"{source_file_directory}/{csv_file}")
    dois = df["DOI"].tolist()
    pmcids = df["PMCID"].tolist()
    identifiers = list(zip(dois, pmcids))
    download_multiple_files(identifiers)


        
