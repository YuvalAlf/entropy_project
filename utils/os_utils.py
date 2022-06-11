import os


def join_create_dir(*paths: str) -> str:
    joined_path = os.path.join(*paths)
    os.makedirs(joined_path, exist_ok=True)
    return joined_path
