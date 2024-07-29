import importlib.resources as pkg_resources
from pathlib import Path

import test_resources


def get_resource_path(file_name):
    root_path = pkg_resources.path(test_resources, "files")
    return Path(root_path, file_name)
