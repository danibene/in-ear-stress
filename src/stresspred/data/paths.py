import os
from pathlib import Path

code_paths = {}
repo_name = Path(os.getcwd()).parents[2].name
code_paths["repo_name"] = repo_name
code_paths["package_parent_dir"] = "lib"


code_paths["repo_path"] = os.getcwd()
base_dir = os.path.basename(code_paths["repo_path"])
while base_dir != code_paths["repo_name"]:
    code_paths["repo_path"] = os.path.dirname(os.path.abspath(code_paths["repo_path"]))
    base_dir = os.path.basename(code_paths["repo_path"])