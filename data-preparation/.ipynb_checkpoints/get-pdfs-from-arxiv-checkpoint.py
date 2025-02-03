
import pandas as pd
import requests
import os
import re
from tqdm import tqdm

# File and directory
csv_file = "arxiv_2024.csv"
output_dir = "arxiv_pdfs"


os.makedirs(output_dir, exist_ok=True)
output_dir_path = os.path.abspath(output_dir)
print(f"Saving PDFs to directory: {output_dir_path}")
try:
    df = pd.read_csv(csv_file)
    if 'Item' not in df.columns:
        raise ValueError("no column named 'arxiv_id'")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()


def sanitize_filename(arxiv_id):
    sanitized_id = re.sub(r'[:<>"/\\|?*]', '_', arxiv_id)  # Replace invalid characters
    return sanitized_id

# Function download PDF and verify size
def download_and_verify(arxiv_id):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    safe_arxiv_id = sanitize_filename(arxiv_id)  # Ensure
    pdf_path = os.path.join(output_dir, f"{safe_arxiv_id}.pdf")

    print(f"Saving to: {pdf_path}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(pdf_url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an error for HTTP

        # save
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify if file exists and is not empty
        if os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            if file_size > 0:
                print(f"Successfully downloaded: {pdf_path} (Size: {file_size} bytes)")
                return True
            else:
                print(f"File is empty: {pdf_path}")
                os.remove(pdf_path)  # Remove empty file
                return False
        else:
            print(f"File not found after download: {pdf_path}")
            return False

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error for {arxiv_id}: {http_err}")
    except requests.exceptions.RequestException as err:
        print(f"Request error for {arxiv_id}: {err}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return False


for arxiv_id in tqdm(df['Item'].astype(str), desc="Processing arXiv PDFs"):
    if download_and_verify(arxiv_id):
        print(f" File ok: {arxiv_id}")
    else:
        print(f"Skipping: {arxiv_id}")


downloaded_files = os.listdir(output_dir)
print(f"\nDownload complete. Total PDFs successfully downloaded: {len(downloaded_files)}")
print(f"Files in directory: {downloaded_files}")
