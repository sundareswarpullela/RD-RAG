import requests
import os
from Bio import Entrez
import xml.etree.ElementTree as ET
from metapub import PubMedFetcher
from ratelimit import limits, sleep_and_retry


os.system('export NCBI_API_KEY="0eb218445608ec87b3f19375b576fe023708"')

# os.environ['NCBI_API_KEY'] = "0eb218445608ec87b3f19375b576fe023708"

@sleep_and_retry
@limits(calls=7, period=1)
def fetch_pubmed_article(PMID):
    print("Fetching article for PMID: ", PMID)
    fetch = PubMedFetcher()
    try:
        article = fetch.article_by_pmid(PMID)
        print("Fetched article for PMID: ", PMID)
        return (PMID, article.to_dict()['abstract'])
    except Exception as e:
        print("Failed to fetch article for PMID: ", PMID)
        print(e)
        return (PMID, "")


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



# article = fetch_pubmed_article("PMC7159299")
# print(article)