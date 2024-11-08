pmid = "26849231"

import requests
def download_file_from_pmc(pmc_id):
    url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/' + pmc_id + '/'
    oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi" + "?id=" + pmc_id
    locator_response = requests.get(oa_url)
    print(locator_response.text)
    # root = ET.fromstring(locator_response.text) 
    # pdf_link = root.find(".//link[@format='pdf']")
    # if pdf_link is not None:
    #     pdf_url = pdf_link.get('href')
    #     print(f"PDF URL: {pdf_url}")
    #     pdf_file_name = f"{pmc_id}.pdf"
        
    #     try:
    #         urllib.request.urlretrieve(pdf_url, pdf_files_location + pdf_file_name)
    #         print(f"Downloaded {pdf_file_name}")
            
    #     except e:
    #         print(e)
    # else:
    #     print("PDF link not found")
    #     pdf_link = root.find(".//link[@format='']")

download_file_from_pmc(pmid)