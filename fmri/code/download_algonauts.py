"""Download Algonauts 2023 data; omit images unless --images is supplied.

Requires numpy and gdown==6.4.1 (uses its folder-listing helper).
"""

import argparse
from math import prod
from pathlib import Path

import gdown
import numpy as np
import requests
from gdown.download_folder import _parse_embedded_folder_view


SOURCE = "1DUf3nGNNFk6YjRjQtZPfAY5N105GoGJb"
IMAGE_FOLDERS = {"training_images", "test_images"}


def validate_file(path):
    size = path.stat().st_size
    assert size > 0, f"Empty file: {path}"
    if path.suffix == ".npy":
        with path.open("rb") as file:
            version = np.lib.format.read_magic(file)
            assert version in {(1, 0), (2, 0)}, f"Unsupported NPY version: {version}"
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, _, dtype = read_header(file)
            if dtype.hasobject:
                # The official ROI mapping files contain pickled dictionaries.
                file.seek(0)
                np.load(file, allow_pickle=True)
                expected_size = file.tell()
            else:
                expected_size = file.tell() + prod(shape) * dtype.itemsize
            assert size == expected_size, f"Incomplete or extra data: {path}"
    elif path.suffix == ".png":
        with path.open("rb") as file:
            assert file.read(8) == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {path}"
            file.seek(-12, 2)
            assert file.read() == b"\x00\x00\x00\x00IEND\xaeB`\x82", f"Incomplete PNG: {path}"


def download_folder(folder_id, destination, session, images=False):
    _, entries = _parse_embedded_folder_view(
        sess=session, folder_id=folder_id, verify=True, timeout=30
    )
    assert entries, f"Empty folder listing: {folder_id}"
    destination.mkdir(parents=True, exist_ok=True)
    for file_id, name, mime_type in entries:
        assert Path(name).name == name and name not in {".", ".."}, name
        path = destination / name
        if mime_type == "application/vnd.google-apps.folder":
            if images or name not in IMAGE_FOLDERS:
                download_folder(file_id, path, session, images)
        else:
            if path.is_file():
                validate_file(path)
                continue
            print(path, flush=True)
            result = gdown.download(
                id=file_id, output=str(path), resume=True,
                use_cookies=False, quiet=True,
                user_agent=requests.utils.default_user_agent(),
            )
            assert result is not None and path.is_file(), f"Download failed: {path}"
            validate_file(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Destination directory")
    parser.add_argument("--images", action="store_true", help="Also download images")
    args = parser.parse_args()
    with requests.Session() as session:
        download_folder(SOURCE, args.folder.expanduser().resolve(), session, args.images)
    print("Download complete.")
