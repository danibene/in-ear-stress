import importlib.resources as pkg_resources
from pathlib import Path

import test_resources


def get_resource_path(file_name):
    with pkg_resources.path(test_resources, "files") as root_path:
        return Path(root_path) / file_name
