"""
python -m src.utils.google_cloud_storage.google_cloud_storage
"""

from pathlib import Path

import gslib.commands.ls as gslib_ls
import gslib.context_config as context_config
from google.cloud import storage
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.cs_api_map import ApiMapConstants, ApiSelector
from gslib.gcs_json_api import GcsJsonApi


def list_gcs_subfolders(gcs_path: str) -> list[str]:
    path = gcs_path.removeprefix("gs://")
    bucket_name, _, prefix = path.partition("/")
    prefix = prefix.strip("/") + "/"

    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    list(blobs)

    return [f"gs://{bucket_name}/{p}" for p in blobs.prefixes]


def download_gcs_path(gcs_path: str, local_dir: str = ".") -> list[str]:
    """
    Download a file or entire directory from GCS.

    Args:
        gcs_path: GCS path to a file or directory, e.g.:
                  'gs://arxiv-dataset/arxiv/arxiv/pdf/2601/2601.00001.pdf'
                  'gs://arxiv-dataset/arxiv/arxiv/pdf/2601'
        local_dir: Local destination directory (default: current directory)

    Returns:
        List of local file paths that were downloaded
    """
    path = gcs_path.removeprefix("gs://")
    bucket_name, _, prefix = path.partition("/")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    # Check if it's a single file (prefix matches an exact blob)
    blob = bucket.blob(prefix)
    if blob.exists(client):
        local_path = Path(local_dir) / Path(prefix).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        print(f"Downloaded: {gcs_path} → {local_path}")
        return [str(local_path)]

    # Otherwise treat as a directory — list all blobs recursively
    prefix = prefix.strip("/") + "/"
    blobs = client.list_blobs(bucket_name, prefix=prefix)

    downloaded = []
    for blob in blobs:
        # Reconstruct relative path and mirror directory structure locally
        relative_path = blob.name.removeprefix(prefix)
        local_path = Path(local_dir) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        print(f"Downloaded: gs://{bucket_name}/{blob.name} → {local_path}")
        downloaded.append(str(local_path))

    return downloaded


# Example usage
if __name__ == "__main__":

    path = "gs://arxiv-dataset/arxiv/arxiv"
    subfolders = list_gcs_subfolders(path)
    for folder in subfolders:
        print(folder)

    # Download a single file
    download_gcs_path(
        "gs://arxiv-dataset/arxiv/arxiv/pdf/2601/2601.03210v1.pdf",
        local_dir="./test_output/google_cloud_storage_download",
    )

    # # Download an entire directory
    # download_gcs_path(
    #     "gs://arxiv-dataset/arxiv/arxiv/pdf/2601", local_dir="./downloads"
    # )
