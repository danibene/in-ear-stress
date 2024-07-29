import inspect
import numpy as np
import os
import re


def get_func_kwargs(func, exclude_keys=[], **kwargs):
    # https://stackoverflow.com/questions/61805344/recursive-function-with-kwargs
    func_args = list(inspect.signature(func).parameters)
    func_kwargs = {
        k: kwargs.pop(k)
        for k in dict(kwargs)
        if k in func_args and k not in exclude_keys
    }
    return func_kwargs


def argtop_k(a, k=1, **kwargs):
    # See https://github.com/numpy/numpy/issues/15128
    # return np.argpartition(a, -k, **kwargs)[-k:]
    if type(a) is list:
        a = np.array(a)
    if k > len(a):
        k = len(a)
    return a.argsort()[-k:][::-1]


def top_k(a, k=1, **kwargs):
    # See https://github.com/numpy/numpy/issues/15128
    return a[argtop_k(a, k=k, **kwargs)]


def find_files_with_string(string, file_dir=None):
    list_file_paths = []
    if file_dir is None:
        file_dir = os.getcwd()
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            try:
                with open(os.path.join(root, file), "r") as auto:
                    data = auto.read()
                    if string in data:
                        list_file_paths.append(os.path.join(root, file))
            except:
                pass
    return list_file_paths


def get_camel_case(s, first_upper=False):
    # https://www.w3resource.com/python-exercises/string/python-data-type-string-exercise-96.php
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
    if first_upper:
        return s
    return "".join([s[0].lower(), s[1:]])


def get_snake_case(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
