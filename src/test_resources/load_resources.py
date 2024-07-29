import importlib.resources as pkg_resources

import test_resources


def get_resource_path(file_name):
    return pkg_resources.path(test_resources, file_name)
