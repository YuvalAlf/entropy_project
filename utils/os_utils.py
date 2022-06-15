import json
import os
from typing import Any


def join_create_dir(*paths: str) -> str:
    joined_path = os.path.join(*paths)
    os.makedirs(joined_path, exist_ok=True)
    return joined_path


def encode_json(value: Any, path: str) -> None:
    with open(path, 'w') as file:
        json.dump(value, file, indent=True)
