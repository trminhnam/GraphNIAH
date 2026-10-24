from pathlib import Path
from typing import Any

from endpoints import download_semantic_scholar_datasets as downloader


def test_download_dataset_reverses_files_without_changing_indices(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    files = [
        "https://example.com/first",
        "https://example.com/second",
        "https://example.com/third",
    ]
    downloaded_files = []

    monkeypatch.setattr(
        downloader,
        "fetch_dataset_details",
        lambda *args: {"files": files},
    )
    monkeypatch.setattr(downloader, "tqdm", None)

    def record_download(
        download_session: Any,
        url: str,
        index: int,
        dataset_dir: Path,
        overwrite: bool,
        show_progress: bool,
        progress_callback: Any = None,
    ) -> Path:
        downloaded_files.append((index, url))
        return dataset_dir / Path(url).name

    monkeypatch.setattr(downloader, "download_dataset_file", record_download)

    downloader.download_dataset(
        api_session=None,
        download_session=None,
        release_id="2026-05-21",
        dataset={"name": "papers"},
        output_root=tmp_path,
        overwrite=False,
        num_workers=1,
        reverse_files=True,
    )

    assert downloaded_files == [
        (3, "https://example.com/third"),
        (2, "https://example.com/second"),
        (1, "https://example.com/first"),
    ]
