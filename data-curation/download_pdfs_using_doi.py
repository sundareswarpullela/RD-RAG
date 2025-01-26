import os
import requests

# # SciHub base URL (for academic access)
# SCIHUB_URL = "https://sci-hub.se/"

# Unpaywall API URL
UNPAYWALL_API = "https://api.unpaywall.org/v2/"

# Your email (required for Unpaywall API)
EMAIL = "psundareswar@gmail.com"

# def get_pdf_from_scihub(doi):
#     """Attempts to retrieve a PDF from Sci-Hub."""
#     try:
#         response = requests.get(f"{SCIHUB_URL}{doi}")
#         soup = BeautifulSoup(response.text, "html.parser")
#         pdf_url = soup.find("iframe")["src"]
        
#         if pdf_url.startswith("//"):
#             pdf_url = "https:" + pdf_url

#         return pdf_url
#     except Exception as e:
#         print(f"Failed to fetch from Sci-Hub: {e}")
#         return None

def get_pdf_from_unpaywall(doi):
    """Attempts to retrieve an open-access PDF via Unpaywall API."""
    try:
        response = requests.get(f"{UNPAYWALL_API}{doi}?email={EMAIL}")
        data = response.json()
        
        if "best_oa_location" in data and data["best_oa_location"]:
            return data["best_oa_location"]["url_for_pdf"]

        return None
    except Exception as e:
        print(f"Failed to fetch from Unpaywall: {e}")
        return None

def download_pdf(pdf_url, save_path):
    """Downloads and saves the PDF file."""
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as pdf_file:
                for chunk in response.iter_content(1024):
                    pdf_file.write(chunk)
            print(f"Downloaded successfully: {save_path}")
        else:
            print(f"Failed to download PDF, HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")

def fetch_research_paper(doi, save_folder="data-curation/data/files"):
    """Fetches and downloads the research paper given a DOI."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filename = f"{save_folder}/{doi.replace('/', '_')}.pdf"

    # Try Unpaywall first (legal open access)
    pdf_url = get_pdf_from_unpaywall(doi)
    # if not pdf_url:
    #     # Fallback to Sci-Hub
    #     pdf_url = get_pdf_from_scihub(doi)

    if pdf_url:
        download_pdf(pdf_url, filename)
    else:
        print(f"Could not find an open-access version for DOI: {doi}")

# Example usage
doi = "10.1146/annurev-med-042921-021447"  # Replace with your DOI
fetch_research_paper(doi, save_folder="data-curation/data/files")
