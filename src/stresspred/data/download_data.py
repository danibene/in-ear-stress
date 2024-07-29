import urllib.request
import pathlib
import zipfile
import os


URL_AUDACE_ONLY_STRESS_CLASS = "REMOVED"
URL_AUDACE_ONLY_DB8K = "REMOVED"
URL_P5_STRESS_ONLY_DB8K = "REMOVED"
URL_P5_STRESS_ONLY_STRESS_CLASS = "REMOVED"


def download_from_url(source_url, out_path, unzip=True):
    # if parent path does not exist, create it
    pathlib.Path(out_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(source_url, out_path)
    if unzip and pathlib.Path(out_path).suffix == ".zip":
        with zipfile.ZipFile(out_path, "r") as z:
            out_dir = pathlib.Path(out_path).parent
            z.extractall(out_dir)
        os.remove(out_path)
