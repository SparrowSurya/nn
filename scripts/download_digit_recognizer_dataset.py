import os
import gzip
import shutil
import urllib.request

def download_file(url: str, dest_folder: str) -> str:
    """Downloads a file from a URL to the target destination folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)

    if os.path.exists(file_path):
        print(f"{filename} already exists, skipping download.")
        return file_path

    print(f"Downloading {filename}...")
    try:
        # Standard user-agent to prevent HTTP 403 Forbidden responses
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded {filename}.")
        return file_path
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return ""

def decompress_gzip(file_path: str):
    """Decompresses a .gz file to its raw binary equivalent."""
    if not file_path or not file_path.endswith('.gz'):
        return

    decompressed_path = file_path[:-3] # Remove '.gz' suffix

    if os.path.exists(decompressed_path):
        print(f"Decompressed file {os.path.basename(decompressed_path)} already exists, skipping.")
        return

    print(f"Decompressing {os.path.basename(file_path)}...")
    try:
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Successfully decompressed to {os.path.basename(decompressed_path)}.")
    except Exception as e:
        print(f"Failed to decompress {os.path.basename(file_path)}: {e}")

def main():
    # Reliable AWS mirror hosted by PyTorch/OSSCI
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    # Resolve target directory relative to the scripts folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.abspath(os.path.join(current_dir, "..", "dataset", "digit-recogniser"))

    print(f"Downloading and decompressing MNIST digit dataset to: {target_dir}")
    for file in files:
        download_url = base_url + file
        file_path = download_file(download_url, target_dir)
        if file_path:
            decompress_gzip(file_path)

if __name__ == "__main__":
    main()
